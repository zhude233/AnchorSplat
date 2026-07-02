# Hugging Face Release

Recommended public surfaces:

- Model: `zhude233/AnchorSplat-20x`
- Processed third-party datasets: `zhude233/anchorsplat-processed-third-party`
- 3DGS-SR dataset: `zhude233/3dgs-sr`
- Collection: `zhude233/anchorsplat`

## Login

```bash
pip install -U huggingface_hub hf_xet
hf auth login
```

## Create Repositories

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
owner = "zhude233"

api.create_repo(f"{owner}/AnchorSplat-20x", repo_type="model", exist_ok=True)
api.create_repo(f"{owner}/anchorsplat-processed-third-party", repo_type="dataset", exist_ok=True)
api.create_repo(f"{owner}/3dgs-sr", repo_type="dataset", exist_ok=True)
PY
```

## Upload Checkpoint

```bash
hf upload zhude233/AnchorSplat-20x \
  checkpoints/anchorsplat_20x.pth \
  anchorsplat_20x.pth \
  --repo-type model
```

## Upload Datasets

```bash
HF_XET_HIGH_PERFORMANCE=1 hf upload zhude233/anchorsplat-processed-third-party \
  /path/to/processed-third-party-data \
  . \
  --repo-type dataset

HF_XET_HIGH_PERFORMANCE=1 hf upload zhude233/3dgs-sr \
  /path/to/3dgs-sr \
  . \
  --repo-type dataset
```

## Create A Collection

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
collection = api.create_collection(
    title="AnchorSplat",
    namespace="zhude233",
    description="Code, model, and datasets for AnchorSplat.",
    exists_ok=True,
)

api.add_collection_item(collection.slug, "zhude233/AnchorSplat-20x", "model", exists_ok=True)
api.add_collection_item(collection.slug, "zhude233/anchorsplat-processed-third-party", "dataset", exists_ok=True)
api.add_collection_item(collection.slug, "zhude233/3dgs-sr", "dataset", exists_ok=True)
print(collection.url)
PY
```

After upload, replace the README and release-note artifact links with the public Hugging Face URLs.
