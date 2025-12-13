# LocalResearch

WARNING: This is incomplete work in progress. It is not ready for use.

LocalResearch is a command-line tool designed for managing, indexing, and semantically searching a local library of documents. It provides a powerful interface to organize your personal knowledge base, supporting various formats like Markdown and PDF.

## Key Features

*   **Document Management**: Syncs with local data sources, tracks changes, and manages document metadata (authors, tags, dates).
*   **Semantic Search**: Generates vector embeddings for your documents to enable semantic search capabilities. Find relevant information based on meaning, not just keywords.
*   **Multi-Model Support**: Supports multiple embedding models. You can list, enable, disable, and select the active model for indexing and searching.
*   **Timeline Generation**: Extracts events from your documents to compile interactive timelines, filterable by date, domain, and keywords.
*   **Data Integrity**: Includes tools to check and clean caches (PDF, SHA256) and verify index consistency.
*   **Interactive CLI**: A robust REPL (Read-Eval-Print Loop) interface with command history and auto-completion.

## Infrastructure

LocalResearch is designed to run on a distributed network of workstations (macOS and Linux) within a local LAN.
*   **Heterogeneous Hardware**: High-performance nodes (CUDA/MPS/XPU) handle resource-intensive vector indexing, while low-power, always-on machines serve queries and small updates.
*   **Data Synchronization**: Nodes share data stored in `~/.local/share/local_research/` via a synchronized file system path `~/LocalResearch`.
    *   `publish`: Copies local data to the shared `~/LocalResearch` folder.
    *   `import`: Updates the local data from the shared `~/LocalResearch` folder.

## Usage

Run the main console application:


```bash
python research_console.py
```

Or start the web server:

```bash
python web_server.py
```

### Common Commands

*   `help`: Show an overview of available commands.
*   `sync`: Sync data sources and check for new or changed documents.
*   `index`: Generate vector indices for new content.
*   `search <query>`: Perform a semantic search.
*   `text`: Display the full text of the last search result.
*   `timeline`: Generate a timeline of events (e.g., `timeline time=2023 domains=work`).
*   `list models`: List available embedding models.
*   `check`: Verify data structures and caches.

## Supported Formats

*   **Markdown**: Native support for markdown files.
*   **PDF**: Extracts text from PDF documents (via Calibre/other handlers).
*   **OrgMode**: Support for Emacs Org-mode files.
*   **Web Interface**: A 3D interactive web client for visual exploration and search (see [web_client/README.md](web_client/README.md)).

## Configuration

Configuration and history are stored in `~/.config/local_research`. The tool allows runtime configuration of variables and processing devices (CPU/CUDA/MPS/XPU).
