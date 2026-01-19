# LocalResearch Web Client

The Web Client provides a rich, interactive interface for exploring your document library. It features a 3D visualization of document embeddings, enabling you to see clusters of related information and navigate your knowledge base spatially.

## Features

-   **3D Point Cloud Visualization**: Interact with a 3D scatter plot of your document embeddings. Clusters represent semantic similarity.
-   **Interactive Search**: Perform semantic searches directly from the browser. Results are highlighted in the 3D view.
-   **Document Inspection**: Click on any point in the 3D cloud to view document chunks, metadata, and full text.
-   **Real-time Logs**: Monitor server activity and search progress via the built-in log pane.
-   **Responsive Layout**: A pane-based layout allows you to arrange the Model view, Search Results, and Property inspector to your liking.

## Architecture

The web client is a single-page application (SPA) built with:
-   **Frontend**: Vanilla JavaScript (ES Next) and CSS.
-   **3D Engine**: [Three.js](https://threejs.org/) for high-performance WebGL rendering.
-   **Communication**: WebSockets for low-latency, bidirectional communication with the `web_server.py`.

## Getting Started

### Prerequisites

Ensure the `LocalResearch` backend is installed and dependencies are met (see main [README](../README.md)).

### Running the Server

Start the web server from the project root:

```bash
python web_server.py
```

By default, the server listens on `0.0.0.0:8080`.

### Accessing the Client

Open your web browser and navigate to:

```
http://localhost:8080
```

## Configuration

Server configuration can be customized via `~/.config/local_research/web_server.json`:

```json
{
    "port": 8080,
    "host": "0.0.0.0",
    "tls": false,
    "cert_file": null,
    "key_file": null
}
```

## Usage

1.  **Select Model**: Upon loading, use the "Models" pane to select an active embedding model.
2.  **Explore**: Drag to rotate the 3D view, scroll to zoom.
3.  **Search**: Context-click or use the search pane to enter natural language queries.
4.  **Inspect**: Click individual points (documents) to reveal their content in the property pane.
