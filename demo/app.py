import os
import threading
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.similarity import SimilarityEngine


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "demo" / "static"
DEFAULT_DATASET_PATH = (
    ROOT / "chembl_22_clean_1576904_sorted_std_final.smi"
    if (ROOT / "chembl_22_clean_1576904_sorted_std_final.smi").exists()
    else ROOT / "SMILES_Big_Data_Set.csv"
)
DEFAULT_MODEL_PATH = ROOT / "full_contrastive_model.pth"


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _maybe_download_dataset(dataset_path: Path) -> Path:
    dataset_url = os.getenv("DATASET_URL")
    if dataset_path.exists() or not dataset_url:
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(dataset_url, dataset_path)
    return dataset_path


def _default_download_path() -> Path:
    dataset_url = os.getenv("DATASET_URL")
    if not dataset_url:
        return DEFAULT_DATASET_PATH

    parsed = urlparse(dataset_url)
    filename = Path(parsed.path).name or "dataset.smi"
    data_dir = _resolve_path(os.getenv("DATA_DIR"), ROOT / "data")
    return data_dir / filename


DATASET_PATH = _resolve_path(os.getenv("DATASET_PATH"), _default_download_path())
MODEL_PATH = _resolve_path(os.getenv("MODEL_PATH"), DEFAULT_MODEL_PATH)
CACHE_DIR = _resolve_path(os.getenv("CACHE_DIR"), ROOT / "demo" / "cache")

app = FastAPI(title="Polymer Similarity Explorer")
engine = SimilarityEngine(DATASET_PATH, MODEL_PATH, cache_dir=CACHE_DIR)
engine_lock = threading.Lock()
engine_state = {"loading": False, "error": None}


def _initialize_engine() -> None:
    with engine_lock:
        if engine.ready or engine_state["loading"]:
            return
        engine_state["loading"] = True
        engine_state["error"] = None

    try:
        engine.dataset_path = _maybe_download_dataset(engine.dataset_path)
        engine.initialize()
    except Exception as exc:  # pragma: no cover
        engine_state["error"] = str(exc)
    finally:
        engine_state["loading"] = False


def _start_engine_thread() -> None:
    worker = threading.Thread(target=_initialize_engine, daemon=True)
    worker.start()


class SearchRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=25)


@app.on_event("startup")
def load_engine() -> None:
    _start_engine_thread()


@app.get("/api/health")
def health() -> dict[str, object]:
    if not engine.ready and not engine_state["loading"] and engine_state["error"] is None:
        _start_engine_thread()
    return {
        "ok": True,
        "mode": engine.mode,
        "ready": engine.ready,
        "loading": engine_state["loading"],
        "error": engine_state["error"],
        "library_size": len(engine.records),
        "invalid_smiles": engine.stats["invalid_smiles"],
        "dataset": engine.dataset_path.name,
    }


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, object]:
    return engine.search(request.smiles, request.top_k)


@app.get("/api/render")
def render(smiles: str, width: int = 320, height: int = 180) -> Response:
    svg = engine.render_svg(smiles, width=width, height=height)
    if not svg:
        raise HTTPException(status_code=400, detail="Could not render molecule")
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
