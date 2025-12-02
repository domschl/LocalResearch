import logging
import os
import json
import asyncio
import threading
import queue
import time
from pathlib import Path
import aiohttp
from aiohttp import web
from document_store import DocumentStore
from vector_store import VectorStore
from research_defs import get_files_of_extensions, ProgressState

# --- Configuration ---
CONFIG_DIR = Path.home() / ".config" / "local_research"
CONFIG_FILE = CONFIG_DIR / "web_server.json"

def load_config():
    defaults = {
        "port": 8080,
        "host": "0.0.0.0",
        "tls": False,
        "cert_file": None,
        "key_file": None
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                defaults.update(config)
        except Exception as e:
            print(f"Error loading config: {e}")
    return defaults

# --- Queues ---
request_queue = queue.Queue()
response_queue = queue.Queue()

# --- Session Management & Logging ---
class ConnectionManager:
    def __init__(self):
        self.active_connections = set()
        self.request_map = {} # uuid -> websocket

    async def connect(self, ws):
        self.active_connections.add(ws)

    def disconnect(self, ws):
        self.active_connections.remove(ws)
        # Clean up request map? 
        # Ideally we'd remove any pending requests for this WS, 
        # but for now we'll just handle closed WS in dispatcher.

    def register_request(self, uuid, ws):
        self.request_map[uuid] = ws

    def unregister_request(self, uuid):
        if uuid in self.request_map:
            del self.request_map[uuid]
            # logger.info(f"Unregistered request {uuid}") # logger not available in class scope easily without passing it or getting it

    def get_ws(self, uuid):
        return self.request_map.get(uuid)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        json_msg = json.dumps(message)
        tasks = []
        for ws in self.active_connections:
            if not ws.closed:
                 tasks.append(ws.send_str(json_msg))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_all(self):
        for ws in list(self.active_connections):
            if not ws.closed:
                await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b'Server shutdown')

manager = ConnectionManager()

# --- Logging Handler ---
class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.loop = None  # Will be set to the event loop
    
    def emit(self, record):
        log_entry = self.format(record)
        # print(f"Handler log entry: {log_entry}")
        msg = {
            "token": "",
            "uuid": "",
            "cmd": "log",
            "payload": log_entry
        }
        # Schedule the broadcast on the event loop from any thread
        if self.loop and manager:
            asyncio.run_coroutine_threadsafe(manager.broadcast(msg), self.loop)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ResearchServer")
ws_handler = WebSocketLogHandler()
ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(ws_handler)  # Add to root logger to capture all modules

# --- Worker Thread ---
def worker_proc():
    logger.debug("Worker thread started")
    
    # Initialize backend
    try:
        ds = DocumentStore()
        vs = VectorStore(ds.storage_path, ds.config_path)
        logger.info("ResearchServer backend initialized")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        return

    while True:
        try:
            req = request_queue.get()
            if req is None: # Poison pill
                break
            
            uuid = req.get('uuid')
            cmd = req.get('cmd')
            payload = req.get('payload')
            
            logger.debug(f"Worker processing: {cmd} ({uuid})")
            
            response_payload = None
            
            if cmd == 'test':
                # Simulate work
                time.sleep(0.5) 
                response_payload = f"Echo: {payload} (Processed)"
                
            elif cmd == 'list_models':
                models = []
                for ind, model in enumerate(vs.model_list):
                    path = vs.model_embedding_path(model['model_name'])
                    cnt = len(get_files_of_extensions(path, ['pt']))
                    models.append({
                        "id": ind + 1,
                        "name": model['model_name'],
                        "docs": cnt,
                        "enabled": model['enabled'],
                        "active": model['model_name'] == vs.config['embeddings_model_name']
                    })
                response_payload = models
            
            elif cmd == 'search':
                search_string = payload
                
                def progress_callback(ps:ProgressState):
                    progress_msg = {
                        "token": req.get('token'),
                        "uuid": uuid,
                        "cmd": "progress",
                        "payload": json.dumps(ps)
                    }
                    response_queue.put(progress_msg)

                # Perform search
                try:
                    results = vs.search(search_string, ds.text_library, progress_callback=progress_callback, highlight=True)
                    
                    # Format results for client
                    formatted_results = []
                    for res in results:
                        descriptor = res['entry']['descriptor']
                        metadata = ds.get_metadata(descriptor)
                        
                        formatted_results.append({
                            "cosine": res['cosine'],
                            "descriptor": descriptor,
                            "hash": res['hash'],
                            "text": res['text'],
                            "chunk_index": res['chunk_index'],
                            "significance": res['significance'],
                            "metadata": metadata
                        })
                    response_payload = formatted_results
                except Exception as e:
                    logger.error(f"Search failed: {e}")
                    response_payload = [] # Or error message
            
            elif cmd == 'select':
                try:
                    logger.debug(f"Processing select command with payload: {payload} (type: {type(payload)})")
                    model_id = int(payload)
                    result = vs.select(model_id)
                    logger.debug(f"vs.select result: {result}")
                    if result is None:
                         # Check if it was a valid selection that just didn't change anything or failed
                         # vs.select logs errors, but doesn't return success/fail clearly other than None vs Name
                         # For now, assume if no exception, it's 'ok' or 'already active'
                         pass
                    response_payload = {"status": "ok", "selected_id": model_id}
                except Exception as e:
                    logger.error(f"Select failed: {e}")
                    response_payload = {"error": str(e)}

            elif cmd == 'get_3d_viz_data':
                model_name = payload
                # Sanitize model name to prevent directory traversal
                if not model_name or '..' in model_name or '/' in model_name:
                     response_payload = {"error": "Invalid model name"}
                else:
                    try:
                        viz_file = os.path.join(vs.visualization_3d, model_name + '.json')
                        if os.path.exists(viz_file):
                            with open(viz_file, 'r') as f:
                                data = json.load(f)
                            response_payload = data
                        else:
                            # If file doesn't exist, maybe try to generate it?
                            # For now, just return error or empty
                            logger.warning(f"3D visualization file not found: {viz_file}")
                            response_payload = {"error": "Visualization data not found", "code": "not_found"}
                    except Exception as e:
                        logger.error(f"Failed to load 3D data: {e}")
                        response_payload = {"error": str(e)}

            elif cmd == 'get_metadata':
                hash_val = payload
                if hash_val in ds.text_library:
                    descriptor = ds.text_library[hash_val]['descriptor']
                    logger.info(f"get_metadata: hash={hash_val}, descriptor={descriptor}")
                    metadata = ds.get_metadata(descriptor)
                    response_payload = metadata
                else:
                    response_payload = {"error": "Document not found"}

            elif cmd == 'get_text':
                hash_val = payload
                if hash_val in ds.text_library:
                    text = ds.text_library[hash_val]['text']
                    response_payload = text
                else:
                    response_payload = {"error": "Document not found"}

            elif cmd == 'get_text_chunk':
                try:
                    hash_val = payload.get('hash')
                    chunk_index = payload.get('chunk_index')
                    logger.info(f"get_text_chunk: hash={hash_val}, chunk_index={chunk_index}")
                    if hash_val in ds.text_library:
                        text = ds.text_library[hash_val]['text']
                        chunk_text = VectorStore.get_chunk_context_aware(
                            text, 
                            chunk_index, 
                            vs.config['chunk_size'], 
                            vs.config['chunk_overlap']
                        )
                        response_payload = chunk_text
                    else:
                        response_payload = {"error": "Document not found"}
                except Exception as e:
                    logger.error(f"Failed to get text chunk: {e}")
                    response_payload = {"error": str(e)}

            elif cmd == 'get_chunk_details':
                try:
                    hash_val = payload.get('hash')
                    chunk_index = payload.get('chunk_index')
                    if hash_val in ds.text_library:
                        entry = ds.text_library[hash_val]
                        text = entry['text']
                        descriptor = entry['descriptor']
                        
                        chunk_text = VectorStore.get_chunk_context_aware(
                            text, 
                            chunk_index, 
                            vs.config['chunk_size'], 
                            vs.config['chunk_overlap']
                        )
                        chunk_text = VectorStore.clean_text(chunk_text)
                        
                        metadata = ds.get_metadata(descriptor)
                        
                        response_payload = {
                            "hash": hash_val,
                            "chunk_index": chunk_index,
                            "descriptor": descriptor,
                            "text": chunk_text,
                            "metadata": metadata,
                            "significance": None # No significance for direct selection
                        }
                    else:
                        response_payload = {"error": "Document not found"}
                except Exception as e:
                    logger.error(f"Failed to get chunk details: {e}")
                    response_payload = {"error": str(e)}

            else:
                response_payload = f"Unknown command: {cmd}"

            response = {
                "token": req.get('token'),
                "uuid": uuid,
                "cmd": "response",
                "request_cmd": cmd, # Echo back the command so client knows what this is for
                "payload": response_payload,
                "final": True
            }
            response_queue.put(response)
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
            # Send error response?
    logger.info("Worker thread stopped")

# --- Response Dispatcher ---
async def response_dispatcher(app):
    logger.debug("Response dispatcher started")
    loop = asyncio.get_running_loop()
    try:
        while True:
            # Run blocking get in executor
            res = await loop.run_in_executor(None, response_queue.get)
            if res is None:
                break
                
            uuid = res.get('uuid')
            ws = manager.get_ws(uuid)
            
            if ws and not ws.closed:
                try:
                    await ws.send_str(json.dumps(res))
                    # logger.info(f"Dispatched response for {uuid}")
                    
                    if res.get('final', False):
                        manager.unregister_request(uuid)
                        logger.debug(f"Request {uuid} completed and unregistered")
                        
                except Exception as e:
                    logger.error(f"Error sending response: {e}")
            else:
                logger.warning(f"WebSocket not found or closed for {uuid}")
                # Cleanup anyway if it was final, to avoid leak
                if res.get('final', False):
                    manager.unregister_request(uuid)
                
    except asyncio.CancelledError:
        pass
    logger.info("Response dispatcher stopped")

# --- Handlers ---

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    await manager.connect(ws)
    logger.info("Websocket connection opened")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                if msg.data == 'close':
                    await ws.close()
                else:
                    try:
                        data = json.loads(msg.data)
                        token = data.get("token", "")
                        uuid = data.get("uuid", "")
                        cmd = data.get("cmd", "")
                        
                        logger.debug(f"Received cmd: {cmd}, uuid: {uuid}")
                        
                        # Register request mapping
                        manager.register_request(uuid, ws)
                        
                        # Queue request
                        request_queue.put(data)
                        logger.debug(f"Queued request {uuid}")

                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {msg.data}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"Websocket connection closed with exception {ws.exception()}")
    finally:
        manager.disconnect(ws)
        logger.debug("Websocket connection closed")

    return ws

async def index_handler(request):
    return web.FileResponse('./web_client/index.html')

async def client_js_handler(request):
    return web.FileResponse('./web_client/client.js')

# --- App Lifecycle ---
async def on_startup(app):
    # Set the event loop for the WebSocket log handler so it can broadcast from any thread
    ws_handler.loop = asyncio.get_event_loop()
    
    # Start worker thread
    app['worker_thread'] = threading.Thread(target=worker_proc, daemon=True)
    app['worker_thread'].start()
    
    # Start dispatcher task
    app['dispatcher_task'] = asyncio.create_task(response_dispatcher(app))

async def on_shutdown(app):
    logger.info("Shutting down...")
    # Close all websockets
    await manager.close_all()
    
    # Stop dispatcher
    if 'dispatcher_task' in app:
        app['dispatcher_task'].cancel()
        try:
            await app['dispatcher_task']
        except asyncio.CancelledError:
            pass

async def on_cleanup(app):
    # Stop worker
    logger.info("Stopping worker thread...")
    request_queue.put(None)
    if 'worker_thread' in app:
        app['worker_thread'].join()
    
    # Unblock response dispatcher executor thread
    response_queue.put(None)
    
    logger.info("Cleanup complete")

def create_app():
    app = web.Application()
    app.add_routes([
        web.get('/', index_handler),
        web.get('/client.js', client_js_handler),
        web.get('/ws', websocket_handler),
        web.static('/third_party', './web_client/third_party'),
    ])
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)
    return app

if __name__ == '__main__':
    config = load_config()
    logger.info(f"Starting server with config: {config}")
    
    app = create_app()
    ssl_context = None
    
    try:
        web.run_app(app, host=config['host'], port=config['port'], ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server stopped with error: {e}")
