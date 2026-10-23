"""Offline tests for the Flickr30k ingestion logic (no GCP needed)."""
import json

import pandas as pd
import pytest
from PIL import Image

from src.data import flickr30k as fk

PREFIX = "gs://bucket/raw/flickr30k/v1/images"


def _write_fixture(tmp_path, rows, make_images=True):
    """rows: (img_id, filename, split, captions)."""
    csv = tmp_path / "ann.csv"
    pd.DataFrame([{
        "raw": json.dumps(caps), "sentids": json.dumps([f"{i}{k}" for k in range(len(caps))]),
        "split": split, "filename": fn, "img_id": i,
    } for i, fn, split, caps in rows]).to_csv(csv, index=False)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    if make_images:
        for n, (_, fn, _, _) in enumerate(rows):
            Image.new("RGB", (32 + n, 24), (n * 40 % 255, 10, 10)).save(img_dir / fn)
    return csv, img_dir


FIVE = ["a dog runs", "a  brown dog\nruns fast", "dog outside", "an animal", "a pet"]
ROWS = [("0", "1.jpg", "train", FIVE), ("1", "2.jpg", "val", FIVE), ("2", "3.jpg", "test", FIVE)]


def test_parse_explodes_captions_and_builds_uris(tmp_path):
    csv, _ = _write_fixture(tmp_path, ROWS, make_images=False)
    images, captions = fk.parse_annotations(csv, PREFIX + "/")
    assert len(images) == 3 and len(captions) == 15
    assert images.loc[0, "gcs_uri"] == f"{PREFIX}/1.jpg"
    assert captions.loc[1, "caption"] == "a brown dog runs fast"     # whitespace normalized
    assert captions.loc[1, "num_words"] == 5
    assert set(captions["split"]) == {"train", "val", "test"}


def test_build_end_to_end_writes_valid_parquet(tmp_path):
    csv, img_dir = _write_fixture(tmp_path, ROWS)
    out = tmp_path / "out"
    rc = fk.main(["build", "--csv", str(csv), "--image_dir", str(img_dir),
                  "--gcs_image_prefix", PREFIX, "--out_dir", str(out), "--verify"])
    assert rc == 0
    images = pd.read_parquet(out / "images.parquet")
    assert {"md5", "bytes", "width", "height", "gcs_uri", "split", "ingested_at"} <= set(images.columns)
    assert images["width"].tolist() == [32, 33, 34]
    assert images["md5"].str.len().eq(32).all()
    assert len(pd.read_parquet(out / "captions.parquet")) == 15
    assert json.loads((out / "validation_report.json").read_text())["stats"]["num_images"] == 3


def test_missing_image_fails_build(tmp_path):
    csv, img_dir = _write_fixture(tmp_path, ROWS)
    (img_dir / "2.jpg").unlink()
    out = tmp_path / "out"
    rc = fk.main(["build", "--csv", str(csv), "--image_dir", str(img_dir),
                  "--gcs_image_prefix", PREFIX, "--out_dir", str(out)])
    assert rc == 1 and not (out / "images.parquet").exists()


def test_corrupt_image_is_an_error(tmp_path):
    csv, img_dir = _write_fixture(tmp_path, ROWS)
    (img_dir / "3.jpg").write_bytes(b"not a jpeg")
    images, captions = fk.parse_annotations(csv, PREFIX)
    images = images.merge(fk.image_stats(img_dir, images["filename"].tolist()), on="filename")
    report = fk.validate(images, captions, img_dir)
    assert not report.ok and any("could not be opened" in e for e in report.errors)


@pytest.mark.parametrize("rows, expect_error, expect_warning", [
    ([("0", "1.jpg", "train", FIVE), ("1", "1.jpg", "train", FIVE)], "duplicate filenames", None),
    ([("0", "1.jpg", "holdout", FIVE)], "unexpected split", None),
    ([("0", "1.jpg", "train", FIVE[:3])], None, "exactly 5 captions"),
])
def test_validation_rules(tmp_path, rows, expect_error, expect_warning):
    csv, _ = _write_fixture(tmp_path, rows, make_images=False)
    report = fk.validate(*fk.parse_annotations(csv, PREFIX))
    if expect_error:
        assert any(expect_error in e for e in report.errors)
    if expect_warning:
        assert report.ok and any(expect_warning in w for w in report.warnings)


def test_extra_files_on_disk_warn(tmp_path):
    csv, img_dir = _write_fixture(tmp_path, ROWS)
    Image.new("RGB", (8, 8)).save(img_dir / "stray.jpg")
    images, captions = fk.parse_annotations(csv, PREFIX)
    images = images.merge(fk.image_stats(img_dir, images["filename"].tolist()), on="filename")
    report = fk.validate(images, captions, img_dir)
    assert report.ok and any("not referenced" in w for w in report.warnings)


def test_pairs_view_sql_is_parameterized():
    sql = fk.PAIRS_VIEW_SQL.format(project="p", dataset="d")
    assert "`p.d.captions`" in sql and "`p.d.images`" in sql and "gcs_uri" in sql


@pytest.mark.parametrize("gcs, bq", [("US-CENTRAL1", "us-central1"), ("US", "US"), ("eu", "EU")])
def test_bq_location_matches_bucket(gcs, bq):
    assert fk.bq_location(gcs) == bq


# ---------------- BigQuery commands against a fake client (checks our call sequence) --------
class _FakeJob:
    def __init__(self, rows=None, df=None):
        self._rows, self._df, self.job_id = rows or [], df, "job_123"

    def result(self):
        return self._rows

    def to_dataframe(self):
        return self._df


class _FakeClient:
    calls = []

    def __init__(self, project=None):
        self.project = project

    def create_dataset(self, ds, exists_ok):
        self.calls.append(("create_dataset", ds.dataset_id, ds.location))

    def load_table_from_uri(self, uri, table_id, job_config):
        self.calls.append(("load", uri, table_id, job_config.write_disposition))
        return _FakeJob()

    def get_table(self, table_id):
        return type("T", (), {"full_table_id": table_id, "num_rows": 3})()

    def delete_table(self, table, not_found_ok):
        self.calls.append(("delete_view", table.table_id))

    def create_table(self, table):
        self.calls.append(("create_view", table.table_id, table.view_query))

    def query(self, sql, job_config=None):
        self.calls.append(("query", sql))
        df = pd.DataFrame({"image_id": ["0", "0", "1"], "split": ["train", "train", "test"],
                           "caption": ["a", "b", "c"]})
        row = type("R", (), {"split": "train", "images": 1, "captions": 2})()
        return _FakeJob(rows=[row], df=df)


@pytest.fixture
def fake_bq(monkeypatch):
    from google.cloud import bigquery
    _FakeClient.calls = []
    monkeypatch.setattr(bigquery, "Client", _FakeClient)
    return _FakeClient


def test_load_bq_sequence(fake_bq):
    assert fk.main(["load-bq", "--project", "p", "--location", "US-CENTRAL1",
                    "--parquet_prefix", "gs://b/processed/flickr30k/v1/"]) == 0
    kinds = [c[0] for c in fake_bq.calls]
    assert kinds[:3] == ["create_dataset", "load", "load"]
    assert fake_bq.calls[0][2] == "us-central1"
    assert fake_bq.calls[1][1] == "gs://b/processed/flickr30k/v1/images.parquet"
    assert all(c[3] == "WRITE_TRUNCATE" for c in fake_bq.calls if c[0] == "load")
    assert ("delete_view", "pairs") in fake_bq.calls
    assert any(c[0] == "create_view" and "`p.flickr30k.images`" in c[2] for c in fake_bq.calls)


def test_export_writes_snapshot(fake_bq, tmp_path):
    out = tmp_path / "snap"
    assert fk.main(["export", "--project", "p", "--out_dir", str(out)]) == 0
    assert len(pd.read_parquet(out / "pairs.parquet")) == 3
    meta = json.loads((out / "snapshot.json").read_text())
    assert meta["rows"] == 3 and meta["images"] == 2 and meta["bq_job_id"] == "job_123"
    assert meta["rows_per_split"] == {"train": 2, "test": 1}
