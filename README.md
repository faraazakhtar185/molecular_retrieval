# Polymer Similarity Explorer

Interactive molecule similarity search for polymer and SMILES-based workflows.
The app serves a FastAPI backend with a small frontend and can run either:

- locally from files on disk
- on a host like Render without a database

## Local Run

Create a virtual environment and install the demo dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r demo/requirements.txt
```

Start the app from the repo root:

```bash
.venv/bin/uvicorn demo.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Dataset Selection

By default the app uses:

- `chembl_22_clean_1576904_sorted_std_final.smi` if present
- otherwise `SMILES_Big_Data_Set.csv`

You can override that behavior with environment variables:

- `DATASET_PATH`: path to the dataset file to load
- `DATASET_URL`: optional URL to download the dataset from if `DATASET_PATH` does not exist yet
- `DATA_DIR`: base directory used for downloaded datasets
- `CACHE_DIR`: where the search cache should live
- `MODEL_PATH`: path to `full_contrastive_model.pth`

## Deploy Without A Database

This project does not need Postgres, Redis, or any other database just to serve
similarity search. The cheapest deploy pattern is:

- keep the model in the repo or on disk
- store the large molecule file on a persistent disk or object storage
- keep the generated cache on persistent disk

### Render Option

The included [`render.yaml`](/Users/faraazakhtar/molecular_retrieval/render.yaml)
is configured for a Docker web service with a mounted disk at `/data`.

What you need to do after creating the service:

1. Deploy the repo on Render.
2. Upload or copy your large `.smi` file onto the mounted disk at `/data/chembl_22_clean_1576904_sorted_std_final.smi`.
3. Redeploy or restart the service.

The service is already configured to use:

- `DATASET_PATH=/data/chembl_22_clean_1576904_sorted_std_final.smi`
- `CACHE_DIR=/data/cache`

That means the expensive cache build can persist across restarts.

### Object Storage Option

If you do not want to manually place the dataset on a disk, you can host the
`.smi` file in object storage and set:

```bash
DATASET_URL=https://your-storage.example.com/chembl_22_clean_1576904_sorted_std_final.smi
DATASET_PATH=/data/chembl_22_clean_1576904_sorted_std_final.smi
CACHE_DIR=/data/cache
```

On first startup, the app will download the file to `DATASET_PATH` and reuse it
on later boots as long as the disk persists.

## Notes

- The large `.smi` file is intentionally git-ignored because it exceeds normal GitHub limits.
- `demo/cache/` is also git-ignored because it is generated at runtime.
- For production, persistent disk or object storage is usually enough. A database is only needed if you later add accounts, saved searches, analytics, or richer metadata queries.
