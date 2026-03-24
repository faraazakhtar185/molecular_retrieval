const searchButton = document.querySelector("#search-button");
const smilesInput = document.querySelector("#smiles-input");
const topKInput = document.querySelector("#top-k");
const topKValue = document.querySelector("#top-k-value");
const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");
const metricMode = document.querySelector("#metric-mode");
const metricLibrary = document.querySelector("#metric-library");
const queryTitle = document.querySelector("#query-title");
const querySubtitle = document.querySelector("#query-subtitle");
const queryCanonical = document.querySelector("#query-canonical");
const querySmiles = document.querySelector("#query-smiles");
const queryCanvas = document.querySelector("#query-canvas");
const resultsMeta = document.querySelector("#results-meta");
const resultsGrid = document.querySelector("#results-grid");
const resultTemplate = document.querySelector("#result-template");

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    metricMode.textContent = payload.mode;
    metricLibrary.textContent = payload.library_size.toLocaleString();
    if (payload.ready) {
      statusDot.classList.add("ready");
      statusText.textContent = `Similarity engine is ready from ${payload.dataset}`;
    } else {
      statusDot.classList.add("warn");
      statusText.textContent = "Engine is installed but not ready yet";
    }
    if (payload.invalid_smiles) {
      resultsMeta.textContent = `${payload.invalid_smiles.toLocaleString()} invalid SMILES were skipped during indexing.`;
    }
  } catch (error) {
    statusDot.classList.add("warn");
    statusText.textContent = "Could not reach the API";
    metricMode.textContent = "offline";
    metricLibrary.textContent = "0";
  }
}

function drawSmiles(canvas, smiles) {
  if (!canvas || !smiles) {
    return;
  }
  const width = canvas.classList.contains("result-canvas") ? 320 : 420;
  const height = canvas.classList.contains("result-canvas") ? 180 : 220;
  canvas.src = `/api/render?smiles=${encodeURIComponent(smiles)}&width=${width}&height=${height}`;
  canvas.onerror = () => {
    canvas.removeAttribute("src");
    canvas.alt = "Preview unavailable";
  };
}

function renderResults(payload) {
  resultsGrid.innerHTML = "";
  if (payload.error) {
    resultsMeta.textContent = payload.error;
    return;
  }

  queryTitle.textContent = payload.query_smiles || payload.canonical_smiles || "Invalid query";
  querySubtitle.textContent = `Running in ${payload.mode} mode against ${payload.library_size.toLocaleString()} indexed molecules.`;
  queryCanonical.textContent =
    payload.canonical_smiles && payload.canonical_smiles !== payload.query_smiles
      ? `Canonicalized as ${payload.canonical_smiles}`
      : "Canonical form matches your input.";
  querySmiles.textContent = payload.query_smiles;
  drawSmiles(queryCanvas, payload.canonical_smiles || payload.query_smiles);
  resultsMeta.textContent = `${payload.results.length} matches ranked by similarity`;

  payload.results.forEach((result, index) => {
    const node = resultTemplate.content.firstElementChild.cloneNode(true);
    node.style.animationDelay = `${index * 70}ms`;
    node.querySelector(".rank-pill").textContent = `#${result.rank}`;
    node.querySelector(".score-pill").textContent = result.score.toFixed(3);
    node.querySelector(".result-smiles").textContent = result.smiles;
    drawSmiles(node.querySelector(".result-canvas"), result.smiles);

    const metaList = node.querySelector(".meta-list");
    const entries = Object.entries(result.metadata || {})
      .filter(([key]) => !["mol", "logP"].includes(key))
      .slice(0, 4);
    if (entries.length === 0) {
      const dt = document.createElement("dt");
      dt.textContent = "info";
      const dd = document.createElement("dd");
      dd.textContent = "No extra metadata";
      metaList.append(dt, dd);
    } else {
      entries.forEach(([key, value]) => {
        const dt = document.createElement("dt");
        dt.textContent = key;
        const dd = document.createElement("dd");
        dd.textContent = value;
        metaList.append(dt, dd);
      });
    }

    resultsGrid.appendChild(node);
  });
}

async function runSearch() {
  const smiles = smilesInput.value.trim();
  if (!smiles) {
    queryTitle.textContent = "Enter a SMILES string";
    querySubtitle.textContent = "The explorer needs a molecule query before it can rank neighbors.";
    queryCanonical.textContent = "";
    querySmiles.textContent = "";
    resultsMeta.textContent = "No search submitted yet.";
    resultsGrid.innerHTML = "";
    return;
  }

  searchButton.disabled = true;
  searchButton.textContent = "Searching...";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        smiles,
        top_k: Number(topKInput.value),
      }),
    });
    const payload = await response.json();
    renderResults(payload);
  } catch (error) {
    resultsMeta.textContent = "The search request failed.";
    resultsGrid.innerHTML = "";
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "Explore neighbors";
  }
}

searchButton.addEventListener("click", runSearch);
smilesInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    runSearch();
  }
});

loadHealth();
