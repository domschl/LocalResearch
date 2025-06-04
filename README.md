## [WIP] implementation of a vector search tool for local notes, libraries, and Calibre books

**Note:** This is work-in-progress!

## Preparations

This project uses `uv` for library dependencies. Use your package manager to install `uv` first. (`brew install uv`, `pacman -S uv`, etc.).

```bash
uv sync
uv run iq.py
```
Upon first start of `iq.py`, a configuration file is created at `~/IcoTqStore/config/icotq.json`, and a default list of embeddings model is stores in the same directory at `model_list.json`. Review the content and especially adapt paths to the content you want to index and search in `icotq.json`. When adding models (from Huggingface) to `model_list.json`, make sure that the model you add is trained for text-embedding! Search will not work when using arbitrary chat models!

The indexer stores (potentially large) amounts of data at `~/IcoTqStore` 

## First use

First step is to `sync` the document sources you defined in `icotq.json`. Supported are text formats (like `.txt`, `.md`, `.py`) and PDFs. Since extracting PDFs is an involved process, text content of PDFs is cached in `PDFTextCache`. Text is extracted via pymudpdf, and if that fails, LLM based OCR is used with pymupdf4llm.

Start `iq.py` with `uv run iq.py`, and once the console prompt appears, use `sync` to extract textual information as base for the embeddings index.

```
sync
```

Once `sync` finished do `list sources`, it should show the number of documents available for indexing for each source defined in `icotq.json`.

Then do `list models` and `select <number>` to select a model from the list for generating the embeddings.

Then start the indexing process with:

```
index
```

Note: Indexing requires considerable compute. Internally we use pytorch which will leverage the accelerator hardware available (Metal for macOS, graphics cards on Linux). Calculation of embeddings requires compute power more than VRam: Apple silicon macs offer lots of VRam, but comparably less compute power than dedicated graphics cards, hence for best results, use a dedicated graphics card (8GB VRam is fine).

Start small! Start with a small collection of texts for your first tests, to get an impression of resources required to index, embed, and search.

Once the index is completed, search:

```
search this is the search-topic I am interested in
```


## Web stuff (preliminary)

### Update dependencies manually

```bash
curl -o static/js/three.module.js https://unpkg.com/three@0.150.0/build/three.module.js
curl -o static/js/OrbitControls.js https://unpkg.com/three@0.150.0/examples/jsm/controls/OrbitControls.js
curl -o static/js/three.core.js https://unpkg.com/three@0.177.0/build/three.core.js
```

Update manually header of OrbitControls.js (import path):

```javascript
import {
	Controls,
	MOUSE,
	Quaternion,
	Spherical,
	TOUCH,
	Vector2,
	Vector3,
	Plane,
	Ray,
	MathUtils
} from '/js/three.module.js';
```