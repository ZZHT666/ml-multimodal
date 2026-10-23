"""Flickr30k ingestion: annotations CSV + images -> validated metadata -> GCS/BigQuery -> DVC snapshot.

Layout this produces (DATASET_VERSION = v1):

  gs://<bucket>/raw/flickr30k/v1/images/<filename>.jpg      immutable raw images
  gs://<bucket>/raw/flickr30k/v1/annotations/*.csv          original annotations
  gs://<bucket>/processed/flickr30k/v1/{images,captions}.parquet
  BigQuery  <project>.flickr30k.images     one row per image   (md5, size, dims, split, gcs_uri)
            <project>.flickr30k.captions   one row per caption (5 per image)
            <project>.flickr30k.pairs      view: captions JOIN images -> training pairs
  data/flickr30k/pairs.parquet + snapshot.json               frozen training snapshot, tracked by DVC

Subcommands (run on the VM by scripts/flickr30k/ingest.sh):
  build        CSV + local image dir -> validated images.parquet / captions.parquet
  load-bq      parquet on GCS -> BigQuery tables + pairs view
  export       BigQuery pairs view -> data/flickr30k/pairs.parquet (then `dvc add`)
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pandas as pd

DATASET = "flickr30k"
VALID_SPLITS = {"train", "val", "test"}
CAPTIONS_PER_IMAGE = 5


# ---------------------------------------------------------------- parsing
def parse_annotations(csv_path, gcs_image_prefix: str, version: str = "v1"
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """nlphuji/flickr30k CSV -> (images_df, captions_df).

    CSV columns: raw (JSON list of captions), sentids (JSON list), split, filename, img_id.
    """
    raw = pd.read_csv(csv_path, dtype={"img_id": str, "filename": str, "split": str})
    missing = {"raw", "sentids", "split", "filename", "img_id"} - set(raw.columns)
    if missing:
        raise ValueError(f"annotations CSV is missing columns: {sorted(missing)}")

    prefix = gcs_image_prefix.rstrip("/") + "/"
    images = pd.DataFrame({
        "image_id": raw["img_id"].astype(str),
        "filename": raw["filename"],
        "split": raw["split"].str.strip().str.lower(),
        "gcs_uri": prefix + raw["filename"],
        "dataset_version": version,
    })

    rows = []
    for img_id, split, caps_json, ids_json in zip(raw["img_id"], images["split"], raw["raw"],
                                                  raw["sentids"], strict=True):
        caps, sent_ids = json.loads(caps_json), json.loads(ids_json)
        if len(sent_ids) != len(caps):
            sent_ids = [f"{img_id}_{i}" for i in range(len(caps))]
        for i, (cap, sid) in enumerate(zip(caps, sent_ids, strict=True)):
            cap = " ".join(str(cap).split())          # collapse whitespace / newlines
            rows.append({"sent_id": str(sid), "image_id": str(img_id), "caption_idx": i,
                         "caption": cap, "num_words": len(cap.split()), "split": split})
    captions = pd.DataFrame(rows, columns=["sent_id", "image_id", "caption_idx", "caption",
                                           "num_words", "split"])
    return images, captions


def image_stats(image_dir, filenames: List[str], verify: bool = False) -> pd.DataFrame:
    """md5 / bytes / width / height for each image. md5 pins exact content, so a training
    snapshot can later prove it used the same bytes (and DVC-style integrity checks work)."""
    from PIL import Image

    image_dir = Path(image_dir)
    out = []
    for name in filenames:
        path = image_dir / name
        rec = {"filename": name, "md5": None, "bytes": None, "width": None, "height": None,
               "readable": False}
        if path.exists():
            data = path.read_bytes()
            rec["md5"] = hashlib.md5(data).hexdigest()
            rec["bytes"] = len(data)
            try:
                with Image.open(path) as im:
                    rec["width"], rec["height"] = im.size
                    if verify:
                        im.verify()
                rec["readable"] = True
            except Exception:
                pass
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- validation
@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(images: pd.DataFrame, captions: pd.DataFrame, image_dir=None) -> Report:
    r = Report()
    r.stats["num_images"] = len(images)
    r.stats["num_captions"] = len(captions)
    r.stats["split_images"] = images["split"].value_counts().to_dict()

    bad_split = set(images["split"]) - VALID_SPLITS
    if bad_split:
        r.errors.append(f"unexpected split values: {sorted(bad_split)}")
    if images["filename"].duplicated().any():
        r.errors.append(f"{images['filename'].duplicated().sum()} duplicate filenames")
    if images["image_id"].duplicated().any():
        r.errors.append(f"{images['image_id'].duplicated().sum()} duplicate image_ids")

    empty = (captions["caption"].str.len() == 0).sum()
    if empty:
        r.errors.append(f"{empty} empty captions")
    per_image = captions.groupby("image_id").size().reindex(images["image_id"], fill_value=0)
    off = per_image[per_image != CAPTIONS_PER_IMAGE]
    if len(off):
        r.warnings.append(f"{len(off)} images do not have exactly {CAPTIONS_PER_IMAGE} captions")
    r.stats["caption_words_p50"] = float(captions["num_words"].median()) if len(captions) else 0
    r.stats["caption_words_max"] = int(captions["num_words"].max()) if len(captions) else 0

    if "md5" in images.columns:
        missing = images["md5"].isna().sum()
        unreadable = (~images["readable"].astype(bool)).sum() - missing
        if missing:
            r.errors.append(f"{missing} images listed in the CSV are missing on disk")
        if unreadable:
            r.errors.append(f"{unreadable} images could not be opened")
        dup_content = images["md5"].dropna().duplicated().sum()
        if dup_content:
            r.warnings.append(f"{dup_content} images are byte-identical duplicates of another image")
        r.stats["total_gb"] = round(float(images["bytes"].fillna(0).sum()) / 1e9, 3)
    if image_dir is not None:
        on_disk = {p.name for p in Path(image_dir).iterdir() if p.is_file()}
        extra = on_disk - set(images["filename"])
        if extra:
            r.warnings.append(f"{len(extra)} files on disk are not referenced by the CSV")
    return r


def bq_location(gcs_location: str) -> str:
    """GCS reports e.g. 'US-CENTRAL1' or 'US'; BigQuery wants 'us-central1' or 'US'.
    The dataset must be co-located with the bucket for load jobs to work."""
    loc = gcs_location.strip()
    return loc.lower() if "-" in loc else loc.upper()


# ---------------------------------------------------------------- commands
def cmd_build(args) -> int:
    images, captions = parse_annotations(args.csv, args.gcs_image_prefix, args.version)
    if args.image_dir:
        stats = image_stats(args.image_dir, images["filename"].tolist(), verify=args.verify)
        images = images.merge(stats, on="filename", how="left")
    images["ingested_at"] = pd.Timestamp.now(tz="UTC")

    report = validate(images, captions, args.image_dir)
    print(json.dumps(report.stats, indent=2, default=str))
    for w in report.warnings:
        print(f"WARNING: {w}")
    for e in report.errors:
        print(f"ERROR: {e}")
    if not report.ok:
        print("validation failed -- not writing outputs", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images.drop(columns=["readable"], errors="ignore").to_parquet(out / "images.parquet", index=False)
    captions.to_parquet(out / "captions.parquet", index=False)
    (out / "validation_report.json").write_text(json.dumps(
        {"stats": report.stats, "warnings": report.warnings}, indent=2, default=str))
    print(f"wrote {out}/images.parquet ({len(images)} rows), captions.parquet ({len(captions)} rows)")
    return 0


PAIRS_VIEW_SQL = """
SELECT c.sent_id, c.image_id, c.caption_idx, c.caption, c.num_words, i.split,
       i.filename, i.gcs_uri, i.md5, i.width, i.height, i.dataset_version
FROM `{project}.{dataset}.captions` c
JOIN `{project}.{dataset}.images` i USING (image_id)
"""


def cmd_load_bq(args) -> int:
    from google.cloud import bigquery

    client = bigquery.Client(project=args.project)
    ds = bigquery.Dataset(f"{args.project}.{args.dataset}")
    ds.location = bq_location(args.location)
    client.create_dataset(ds, exists_ok=True)

    for table in ("images", "captions"):
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # idempotent re-runs
            clustering_fields=["split"],
        )
        uri = f"{args.parquet_prefix.rstrip('/')}/{table}.parquet"
        job = client.load_table_from_uri(uri, f"{args.project}.{args.dataset}.{table}",
                                         job_config=job_config)
        job.result()
        t = client.get_table(f"{args.project}.{args.dataset}.{table}")
        print(f"loaded {uri} -> {t.full_table_id}: {t.num_rows} rows")

    view = bigquery.Table(f"{args.project}.{args.dataset}.pairs")
    view.view_query = PAIRS_VIEW_SQL.format(project=args.project, dataset=args.dataset)
    client.delete_table(view, not_found_ok=True)
    client.create_table(view)
    print(f"created view {args.project}.{args.dataset}.pairs")

    # Sanity check straight from BigQuery: rows per split.
    q = f"SELECT split, COUNT(DISTINCT image_id) images, COUNT(*) captions " \
        f"FROM `{args.project}.{args.dataset}.pairs` GROUP BY split ORDER BY split"
    for row in client.query(q).result():
        print(f"  {row.split:<6} images={row.images:>6} captions={row.captions:>7}")
    return 0


def cmd_export(args) -> int:
    """Freeze the current BigQuery pairs into a parquet snapshot that DVC versions.

    BigQuery tables are mutable (someone can re-load them); a training run must point at an
    immutable data version, so we materialize the exact rows used and `dvc add` them.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=args.project)
    query = (f"SELECT * FROM `{args.project}.{args.dataset}.pairs` "
             f"WHERE dataset_version = @version ORDER BY image_id, caption_idx")
    job = client.query(query, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("version", "STRING", args.version)]))
    df = job.to_dataframe()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "pairs.parquet", index=False)
    meta = {
        "source": f"{args.project}.{args.dataset}.pairs",
        "dataset_version": args.version,
        "query": query,
        "bq_job_id": job.job_id,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": len(df),
        "images": int(df["image_id"].nunique()),
        "rows_per_split": df["split"].value_counts().to_dict(),
    }
    (out / "snapshot.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--csv", required=True)
    b.add_argument("--image_dir", default=None, help="local extracted images (for md5/size/dims)")
    b.add_argument("--gcs_image_prefix", required=True, help="gs://<bucket>/raw/flickr30k/v1/images")
    b.add_argument("--version", default="v1")
    b.add_argument("--out_dir", default="build/flickr30k")
    b.add_argument("--verify", action="store_true", help="full PIL integrity check (slower)")
    b.set_defaults(fn=cmd_build)

    lq = sub.add_parser("load-bq")
    lq.add_argument("--project", required=True)
    lq.add_argument("--dataset", default=DATASET)
    lq.add_argument("--location", default="US", help="must match the bucket's location")
    lq.add_argument("--parquet_prefix", required=True, help="gs://<bucket>/processed/flickr30k/v1")
    lq.set_defaults(fn=cmd_load_bq)

    e = sub.add_parser("export")
    e.add_argument("--project", required=True)
    e.add_argument("--dataset", default=DATASET)
    e.add_argument("--version", default="v1")
    e.add_argument("--out_dir", default="data/flickr30k")
    e.set_defaults(fn=cmd_export)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
