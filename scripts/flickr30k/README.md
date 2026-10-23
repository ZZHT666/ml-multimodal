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

## 5. Feature extraction on the VM (~20–30 min on 8 vCPU)

Both backbones are frozen, so their outputs are computed once and cached;
training then only fits the projection heads on ~150 MB of vectors.

```bash
# on the VM, in a git clone of the repo (code comes from GitHub, not a zip)
git clone https://github.com/<you>/ml-multimodal.git ~/ml-multimodal && cd ~/ml-multimodal
bash scripts/flickr30k/setup_vm.sh && source ~/venv/bin/activate
dvc pull data/flickr30k.dvc                    # the exact snapshot version from step 4
tmux new -s features
python -m src.features --snapshot data/flickr30k/pairs.parquet \
    --image_root ~/flickr30k_work/extract/flickr30k-images --verify_md5 \
    --out_dir features/flickr30k-v1 \
    --upload gs://zht-mle-demo-2026-a1b7/features/flickr30k/v1/dinov2-small__minilm-l6
```

`--image_root` reads the local copy left by ingest (same bytes -- `--verify_md5`
checks every image against the snapshot). Without it, each image is read from its
`gcs_uri`, which is what a fresh machine would do.

## 6. Train, assemble, index -- on the same VM

Everything stays in the cloud: features come from GCS, MLflow artifacts go to GCS,
DVC outputs go to the GCS remote; only code + small pointer files go to GitHub.

```bash
cd ~/ml-multimodal && git pull && bash scripts/flickr30k/setup_vm.sh && source ~/venv/bin/activate
export MLFLOW_ARTIFACT_ROOT=gs://zht-mle-demo-2026-a1b7/mlflow-artifacts
rm -rf features/flickr30k-v1      # the local copy from step 5; re-import it *from GCS* with lineage
make features                     # dvc import-url gs://.../dinov2-small__minilm-l6 features/flickr30k-v1
make repro                        # dvc repro: train_flickr -> assemble -> build_index; dvc metrics show
make mlflow-backup                # mlflow.db -> GCS
dvc push                          # outputs (heads, fusion_model.pt, index) -> DVC remote in GCS
git add features/flickr30k-v1.dvc dvc.lock outputs/flickr30k/metrics.json
git commit -m "model(flickr30k): heads trained on cached features" && git push
```

MLflow UI: `make mlflow-ui` on the VM, then from Cloud Shell
`gcloud compute ssh ml-worker --zone=us-central1-a -- -L 8081:localhost:5000` and open
Cloud Shell's *Web Preview* on port 8081.

`dvc import-url` records *where* the features came from (the GCS URL) and their
md5, so `dvc update features/flickr30k-v1.dvc` re-imports if they are regenerated.
The heads are selected by val rsum; test Recall@1/5/10 is computed once at the end
(standard Flickr30k 1k-image test protocol).

## 7. Stop paying for the VM when idle

```bash
bash scripts/flickr30k/create_vm.sh stop     # or `delete` when completely done
```
