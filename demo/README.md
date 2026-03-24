# Polymer Similarity Explorer

This demo turns the project into an interactive nearest-neighbor search tool.
Visitors can paste a SMILES string and get a ranked list of similar molecules
from the library in `SMILES_Big_Data_Set.csv`.

## How it works

- Preferred mode: use the trained contrastive graph encoder from
  `full_contrastive_model.pth`, embed the query and the molecule library, then
  rank neighbors by cosine similarity.
- Fallback mode: if the graph-learning stack is unavailable, use RDKit Morgan
  fingerprints and rank molecules by Tanimoto similarity.

## Run locally

1. Install the dependencies:

   ```bash
   pip install -r demo/requirements.txt
   ```

2. Start the app from the repo root:

   ```bash
   uvicorn demo.app:app --reload
   ```

3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

The API will build a cached search index in `demo/cache/` the first time it
starts. Subsequent launches reuse that cache.

## Deploy

This project is easiest to deploy as a Docker-based web service because the
chemistry and graph packages can be finicky on plain Python builds.

### Render

1. Push this repo to GitHub.
2. Sign in to Render and create a new Blueprint or Web Service.
3. Point Render at the repository.
4. Render will detect `render.yaml` and `Dockerfile`.
5. Deploy and wait for the first image build to finish.

The app starts with:

```bash
uvicorn demo.app:app --host 0.0.0.0 --port $PORT
```

### Railway

1. Push this repo to GitHub.
2. Create a new Railway project from the repo.
3. Railway should detect the `Dockerfile` automatically.
4. Generate a public domain in the service networking settings.

If you want a sharper production setup later, the next step would be to move
the frontend to Next.js and keep this similarity service as the API.
