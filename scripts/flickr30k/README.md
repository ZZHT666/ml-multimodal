# Flickr30k → GCS → BigQuery → DVC

Real data never touches the laptop: download, validation and upload run on a VM
in the bucket's region; the laptop only runs git and a small (~15 MB) DVC snapshot.

```
Hugging Face (nlphuji/flickr30k)
      │ curl on the VM (cloud-to-cloud)
      ▼
gs://<bucket>/raw/flickr30k/v1/images/*.jpg          immutable raw data (--no-clobber; changes => v2)
gs://<bucket>/raw/flickr30k/v1/annotations/          original CSV + SHA256SUMS (provenance)
      │ src/data/flickr30k.py build  (md5/size/dims per image, 5 captions/image, split checks)
      ▼
gs://<bucket>/processed/flickr30k/v1/{images,captions}.parquet + validation_report.json
      │ load-bq (WRITE_TRUNCATE, clustered by split)
      ▼
BigQuery  flickr30k.images | flickr30k.captions | flickr30k.pairs (view)
      │ export (parameterized query, dataset_version = v1)
      ▼
data/flickr30k/pairs.parquet + snapshot.json  ──dvc add / dvc push──►  DVC remote (same bucket)
```

**Why both BigQuery and DVC?** BigQuery is the queryable source of truth (join,
filter, stats with SQL) but its tables are mutable. Training needs an immutable,
reproducible input, so the exact rows are frozen into a parquet snapshot and
versioned with DVC. The snapshot carries each image's `gcs_uri` + `md5`, so it
also pins the image bytes without copying 4.4 GB of images into DVC.

## 1. Create the VM (Cloud Shell)

```bash
export PROJECT_ID=<project>
bash scripts/flickr30k/create_vm.sh          # e2-standard-8, 100 GB disk, same region as the bucket
gcloud compute ssh ml-worker --zone=<zone printed above>
```

## 2. Get the code onto the VM and set it up

```bash
git clone https://github.com/<you>/ml-multimodal.git && cd ml-multimodal
bash scripts/flickr30k/setup_vm.sh
source ~/venv/bin/activate
```

## 3. Ingest (~15–25 min: 4.4 GB download, md5 + verify 31k images, upload)

```bash
bash scripts/flickr30k/ingest.sh             # BUCKET / VERSION / WORK env vars are optional
```

Check in the BigQuery console:

```sql
SELECT split, COUNT(DISTINCT image_id) AS images, COUNT(*) AS captions
FROM flickr30k.pairs GROUP BY split;
```

## 4. Freeze a training snapshot and version it with DVC

This step only touches ~15 MB of metadata, so it can run on the laptop
(`gcloud auth application-default login` first) or on the VM.

```bash
python -m src.data.flickr30k export --project $PROJECT_ID --version v1 --out_dir data/flickr30k
dvc add data/flickr30k
git add data/flickr30k.dvc data/.gitignore
git commit -m "data(flickr30k): v1 snapshot from BigQuery"
dvc push
git push
```

## 5. Stop paying for the VM when idle

```bash
bash scripts/flickr30k/create_vm.sh stop     # or `delete` when completely done
```
