# Polymer Similarity Explorer

This app serves a FastAPI backend and small frontend for nearest-neighbor
search over the bundled `SMILES_Big_Data_Set.csv` dataset.

## Local Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r demo/requirements.txt
.venv/bin/uvicorn demo.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Deploy

The deploy is configured to use the bundled CSV dataset and store runtime cache
in `/tmp/cache`, which is a better fit for lightweight hosting.

For Render:

- `DATASET_PATH=SMILES_Big_Data_Set.csv`
- `CACHE_DIR=/tmp/cache`

Remove any old `DATASET_URL` environment variable before redeploying.
