import os
import sys
import json
import logging
from typing import TypedDict, cast, Any

import numpy as np
from aiohttp import web
import aiohttp_cors

# Assume icotq_store.py contains the updated class
try:
    from icotq_store import (
        IcoTqStore, TqSource, SearchResult as StoreSearchResult,
        IcotqError,
        ModelInfo
    )
except ImportError:
    print("ERROR: Could not import IcoTqStore. Make sure 'icotq_store.py' is accessible.")
    sys.exit(1)

# --- Configuration ---
HOST = "0.0.0.0" # Or "localhost"
PORT = 8000
# Determine store path (e.g., relative to this script or from environment)
# For simplicity, let's assume a 'store_data' directory exists sibling to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORE_PATH = os.path.expanduser("~/IcoTqStore")
# Allow overriding via environment variable
STORE_DATA_PATH = DEFAULT_STORE_PATH # os.environ.get("ICOTQ_STORE_PATH", DEFAULT_STORE_PATH)
config_path = os.path.join(STORE_DATA_PATH, "config")
CONFIG_FILE_PATH = os.path.expanduser(os.path.join(config_path, "icotq.json"))  # join(STORE_DATA_PATH, "icotq_config_aiohttp.json") # Use a dedicated config

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
log = logging.getLogger("IcoTqBackend")
logging.getLogger("IcoTqStore").setLevel(logging.INFO) # Adjust as needed

# --- Data Structures for API (using TypedDict) ---

# Reusing TypedDicts from icotq_store where possible.
# Define specific API response structures if they differ significantly.

class StatusResponse(TypedDict):
    library_size: int
    current_model_name: str | None
    available_models: list[str]
    sources: list[TqSource]
    # Add other relevant status info

# For adding/updating sources via API (example)
class SourceApiData(TypedDict):
    name: str
    tqtype: str # 'folder' or 'calibre_library'
    path: str
    file_types: list[str]

# API response structure for search results
class SearchApiResponseItem(TypedDict):
    cosine: float
    index: int
    offset: int
    desc: str
    chunk: str
    # text: str # Omit full text by default in API response for brevity
    yellow_liner: list[float] | None

class ModelInfo(TypedDict): # For listing models
     model_hf_name: str
     model_name: str
     # Add other fields if needed by UI

# --- Global Store Instance ---
# This holds the state. For multi-process, this needs rethinking (e.g., shared memory, separate service).
# For local single-process server, this is okay.
store: IcoTqStore | None = None

async def initialize_store(app: web.Application):
    """Initialize the IcoTqStore instance."""
    global store
    log.info(f"Using IcoTqStore data path: {STORE_DATA_PATH}")
    log.info(f"Using config file: {CONFIG_FILE_PATH}")

    # Ensure store path exists for config file
    if not os.path.exists(STORE_DATA_PATH):
        try:
            os.makedirs(STORE_DATA_PATH)
            log.info(f"Created store data directory: {STORE_DATA_PATH}")
        except OSError as e:
             log.critical(f"Failed to create store directory {STORE_DATA_PATH}: {e}. Exiting.")
             sys.exit(1) # Exit if we can't create essential path

    try:
        # Pass the specific config file path to the store
        store = IcoTqStore(config_file_override=CONFIG_FILE_PATH)
        log.info("IcoTqStore initialized successfully.")
        app['store'] = store # Make store accessible in handlers via app instance
    except IcotqError as e:
        log.critical(f"Failed to initialize IcoTqStore: {e}", exc_info=True)
        # Allow app to start but log critical failure
        app['store'] = None
    except Exception as e:
         log.critical(f"Unexpected error initializing IcoTqStore: {e}", exc_info=True)
         app['store'] = None


async def cleanup_store(_app: web.Application):
    """Placeholder for any cleanup needed on shutdown."""
    log.info("Cleaning up backend resources...")
    # store instance might have resources to release if added later (e.g., file handles)
    pass


# --- Request Handler Helpers ---

def get_bool_query_param(request: web.Request, param_name: str, default: bool = False) -> bool:
    """Safely parses a boolean query parameter."""
    val = request.query.get(param_name)
    if val is None:
        return default
    return val.lower() in ['true', '1', 'yes', 'on']

def get_int_query_param(request: web.Request, param_name: str, default: int) -> int:
     """Safely parses an integer query parameter."""
     val_str = request.query.get(param_name)
     if val_str is None:
         return default
     try:
         return int(val_str)
     except (ValueError, TypeError):
         # Consider raising HTTPBadRequest here or just return default
         log.warning(f"Invalid integer value '{val_str}' for query param '{param_name}'. Using default {default}.")
         return default

# --- API Handlers ---

async def get_status_handler(request: web.Request) -> web.Response:
    """Returns current status of the store."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None:
         return web.json_response({"error": "Store not initialized"}, status=503) # Service Unavailable

    try:
        lib_size, model_name, available_models, sources = store_instance.get_status_info()

        status = StatusResponse(
            library_size=lib_size,
            current_model_name=model_name,
            available_models=available_models,
            sources=sources
        )
        return web.json_response(cast(dict[str, Any], status))  # pyright:ignore[reportInvalidCast, reportExplicitAny]
    except Exception as e:
        log.error(f"Error getting status: {e}", exc_info=True)
        return web.json_response({"error": "Internal server error getting status"}, status=500)


async def trigger_sync_handler(request: web.Request) -> web.Response:
    """Triggers the sync_texts operation."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None: return web.json_response({"error": "Store not initialized"}, status=503)

    log.info("API: Received request to trigger sync...")
    # Note: This runs synchronously in the handler. For long operations,
    # consider background tasks (e.g., asyncio.create_task, requires more complex state management)
    try:
        store_instance.sync_texts()
        log.info("API: sync_texts completed.")
        return web.json_response({"message": "Sync completed successfully."})
    except IcotqError as e:
        log.error(f"API: Sync error: {e}", exc_info=True)
        return web.json_response({"error": f"Sync failed: {str(e)}"}, status=500)
    except Exception as e:
         log.error(f"API: Unexpected sync error: {e}", exc_info=True)
         return web.json_response({"error": "Unexpected server error during sync."}, status=500)


async def trigger_index_handler(request: web.Request) -> web.Response:
    """Triggers the generate_embeddings operation."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None: return web.json_response({"error": "Store not initialized"}, status=503)

    purge = get_bool_query_param(request, 'purge', False)
    log.info(f"API: Received request to trigger index (purge={purge})...")

    try:
        store_instance.generate_embeddings(purge=purge)
        log.info(f"API: generate_embeddings (purge={purge}) completed.")
        return web.json_response({"message": f"Index generation (purge={purge}) completed."})
    except IcotqError as e:
        log.error(f"API: Index error: {e}", exc_info=True)
        return web.json_response({"error": f"Index generation failed: {str(e)}"}, status=500)
    except Exception as e:
         log.error(f"API: Unexpected index error: {e}", exc_info=True)
         return web.json_response({"error": "Unexpected server error during index generation."}, status=500)


async def trigger_check_handler(request: web.Request) -> web.Response:
    """Triggers the check_clean operation."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None: return web.json_response({"error": "Store not initialized"}, status=503)

    dry_run = get_bool_query_param(request, 'dry_run', True)
    log.info(f"API: Received request to trigger check/clean (dry_run={dry_run})...")

    try:
        # check_clean logs its findings internally
        store_instance.check_clean(dry_run=dry_run)
        log.info(f"API: check_clean (dry_run={dry_run}) completed.")
        return web.json_response({"message": f"Check/Clean operation (dry_run={dry_run}) completed. Check server logs for details."})
    except IcotqError as e:
        log.error(f"API: Check/clean error: {e}", exc_info=True)
        return web.json_response({"error": f"Check/clean failed: {str(e)}"}, status=500)
    except Exception as e:
         log.error(f"API: Unexpected check/clean error: {e}", exc_info=True)
         return web.json_response({"error": "Unexpected server error during check/clean."}, status=500)


async def get_models_handler(request: web.Request) -> web.Response:
    """Lists available models known to the store."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None: return web.json_response({"error": "Store not initialized"}, status=503)

    try:
        models_list: list[ModelInfo] = store_instance.get_available_model_info()
        return web.json_response(cast(list[dict[str, Any]], models_list))  # pyright:ignore[reportExplicitAny]
    except Exception as e:
        log.error(f"API: Error getting models list: {e}", exc_info=True)
        return web.json_response({"error": "Failed to retrieve model list."}, status=500)


async def load_model_handler(request: web.Request) -> web.Response:
    """Loads a specified model."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None: return web.json_response({"error": "Store not initialized"}, status=503)

    try:
        data = await request.json()  # pyright:ignore[reportAny]
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e: # Handle cases where request body isn't JSON
        log.warning(f"Failed to read request body as JSON: {e}")
        return web.json_response({"error": "Could not parse request body"}, status=400)


    # Manual validation
    model_name = data.get('model_name')  # pyright:ignore[reportAny]
    if not model_name or not isinstance(model_name, str):
        return web.json_response({"error": "Missing or invalid 'model_name' field"}, status=400)

    device = data.get('device', 'auto')  # pyright:ignore[reportAny]
    if not isinstance(device, str):
         return web.json_response({"error": "Invalid 'device' field type"}, status=400)

    trust_code = data.get('trust_remote_code', False)  # pyright:ignore[reportAny]
    if not isinstance(trust_code, bool):
         return web.json_response({"error": "Invalid 'trust_remote_code' field type"}, status=400)

    log.info(f"API: Received request to load model '{model_name}'")
    try:
        success = store_instance.load_model(
            name=model_name,
            device=device,
            trust_remote_code=trust_code
        )
        if success:
            return web.json_response({"message": f"Model '{model_name}' loaded successfully."})
        else:
            # Check logs for specific failure reason
            return web.json_response({"error": f"Failed to load model '{model_name}'. See server logs."}, status=500) # Use 500 as it's a server-side load fail
    except IcotqError as e:
         log.error(f"API: Load model error: {e}", exc_info=True)
         return web.json_response({"error": str(e)}, status=500)
    except Exception as e:
          log.error(f"API: Unexpected load model error: {e}", exc_info=True)
          return web.json_response({"error": "Unexpected server error loading model."}, status=500)


async def search_api_handler(request: web.Request) -> web.Response:
    """Handles search requests."""
    store_instance: IcoTqStore | None = request.app.get('store')
    if store_instance is None: return web.json_response({"error": "Store not initialized"}, status=503)

    try:
        req_data = await request.json()  # pyright:ignore[reportAny]
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        log.warning(f"Failed to read search request body as JSON: {e}")
        return web.json_response({"error": "Could not parse request body"}, status=400)

    # Manual validation and default setting
    search_text = req_data.get('search_text')  # pyright:ignore[reportAny]
    if not search_text or not isinstance(search_text, str):
        return web.json_response({"error": "Missing or invalid 'search_text'"}, status=400)

    try:
        max_results = int(req_data.get('max_results', 10))  # pyright:ignore[reportAny]
        if max_results <= 0: max_results = 10
    except (ValueError, TypeError):
        return web.json_response({"error": "Invalid 'max_results' value"}, status=400)

    yellow_liner = req_data.get('yellow_liner', False)  # pyright:ignore[reportAny]
    if not isinstance(yellow_liner, bool):
         return web.json_response({"error": "Invalid 'yellow_liner' value"}, status=400)

    try:
        context_length = int(req_data.get('context_length', 16))  # pyright:ignore[reportAny]
        if context_length <= 0: context_length = 16
    except (ValueError, TypeError):
        return web.json_response({"error": "Invalid 'context_length' value"}, status=400)

    try:
        context_steps = int(req_data.get('context_steps', 4))  # pyright:ignore[reportAny]
        if context_steps <= 0: context_steps = 4
    except (ValueError, TypeError):
        return web.json_response({"error": "Invalid 'context_steps' value"}, status=400)

    compression_mode = req_data.get('compression_mode', 'none')  # pyright:ignore[reportAny]
    if compression_mode not in ['none', 'light', 'full']:
         return web.json_response({"error": "Invalid 'compression_mode' value"}, status=400)

    log.info(f"API: Received search request: '{search_text[:50]}...'")
    try:
        results: list[StoreSearchResult] = store_instance.search(
            search_text=search_text,
            max_results=max_results,
            yellow_liner=yellow_liner,
            context_length=context_length,
            context_steps=context_steps,
            compression_mode=compression_mode  # pyright:ignore[reportAny]
        )

        # Convert results to API response format (JSON serializable)
        api_results: list[SearchApiResponseItem] = []
        for res in results:
            yellow_list: list[float] | None = None
            if res.get('yellow_liner') is not None:
                # Ensure it's serializable (numpy arrays aren't by default)
                if isinstance(res['yellow_liner'], np.ndarray):
                    yellow_list = cast(list[float], res['yellow_liner'].tolist())
                else:
                    yellow_list = None

            api_item: SearchApiResponseItem = {
                "cosine": res["cosine"],
                "index": res["index"],
                "offset": res["offset"],
                "desc": res["desc"],
                "chunk": res["chunk"],
                # "text": res["text"], # Omit full text
                "yellow_liner": yellow_list
            }
            api_results.append(api_item)

        return web.json_response(cast(list[dict[str, Any]], api_results))  # pyright:ignore[reportExplicitAny]

    except IcotqError as e:
         log.error(f"API: Search error: {e}", exc_info=True)
         return web.json_response({"error": f"Search failed: {str(e)}"}, status=500)
    except Exception as e:
          log.error(f"API: Unexpected search error: {e}", exc_info=True)
          return web.json_response({"error": "Unexpected server error during search."}, status=500)

# --- Setup and Run ---
async def root_handler(_request: web.Request) -> web.Response:
    """Serves the main index.html file."""
    index_path = os.path.join(SCRIPT_DIR, 'static', 'index.html')
    try:
        return cast(web.Response, web.FileResponse(index_path))  # pyright:ignore[reportInvalidCast]
    except FileNotFoundError:
        log.error(f"index.html not found at {index_path}")
        return web.Response(text="Frontend not found.", status=404)
    except Exception as e:
         log.error(f"Error serving index.html: {e}", exc_info=True)
         return web.Response(text="Error serving frontend.", status=500)

def setup_routes(app: web.Application):
    """Add routes to the application."""
    _ = app.router.add_get("/api/status", get_status_handler)
    _ = app.router.add_post("/api/actions/sync", trigger_sync_handler)
    _ = app.router.add_post("/api/actions/index", trigger_index_handler) # Takes ?purge=true
    _ = app.router.add_post("/api/actions/check", trigger_check_handler) # Takes ?dry_run=false
    _ = app.router.add_get("/api/models", get_models_handler)
    _ = app.router.add_post("/api/models/load", load_model_handler)
    _ = app.router.add_post("/api/search", search_api_handler)
    # TODO: Add routes for managing sources (GET /api/sources, POST /api/sources, DELETE /api/sources/{id})
    # Static File Serving
    static_dir = os.path.join(SCRIPT_DIR, 'static')
    if not os.path.isdir(static_dir):
        log.warning(f"Static directory not found at {static_dir}. Frontend will not be served.")
    else:
        # Serve files under /static/ URL path from the ./static/ directory
        _ = app.router.add_static('/static/', path=static_dir, name='static')
        log.info(f"Serving static files from: {static_dir}")

        # Serve index.html at the root path '/'
        _ = app.router.add_get('/', root_handler)

def main():
    """Main entry point to set up and run the server."""
    app = web.Application()

    # Setup store initialization and cleanup
    app.on_startup.append(initialize_store)
    app.on_cleanup.append(cleanup_store)

    # Setup routes
    setup_routes(app)

    # --- Configure CORS ---
    # Allow requests from the origin where the frontend is served (localhost:8000)
    # Use "*" for development convenience, but restrict in production.
    cors = aiohttp_cors.setup(app, defaults={
        # Use "*" for local dev, replace with specific frontend origin in prod
        "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*", # Allow common methods
            )
    })

    # Apply CORS to all routes.
    for route in list(app.router.routes()):
        cors.add(route)  # pyright:ignore[reportUnknownMemberType]
    # ----------------------

    log.info(f"Starting IcoTq backend server on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT, access_log=log) # Use built-in runner

if __name__ == "__main__":
    main()