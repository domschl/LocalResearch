import logging
import os
import sys
import argparse
from argparse import ArgumentParser
from rich.console import Console
from icotq_store import IcoTqStore, SearchResult
from typing import cast, TypedDict, Any
import numpy as np
import http.server
import socketserver
import threading
import json
import webbrowser
from urllib.parse import parse_qs, urlparse

def iq_info(its: IcoTqStore, _logger:logging.Logger) -> None:
    its.list_sources()
    print()
    print("Use 'list models' for an overview of available models and their index status")
    print("Use 'sync' to load new/updated documents from all sources")
    print("Use 'index' to index using the current model (Use 'sync' first to load new/updated documents)")
    print("Use 'search <search phrase>' to search using the current model's index")

def iq_index(its: IcoTqStore, _logger:logging.Logger, param:str):
    if param == 'purge':
        purge = True
    else:
        purge = False
    its.generate_embeddings(purge=purge)

def iq_search(its: IcoTqStore, logger:logging.Logger, search_spec: str):
    max_results = 8
    context_length = 32
    context_steps = 4
    yellow = True
    cols, _ = os.get_terminal_size()
    results: list[SearchResult] = its.search(search_text=search_spec, yellow_liner=yellow, context_length=context_length, context_steps=context_steps, max_results=max_results, compression_mode="full")
    print()
    print()
    console = Console()
    if len(results) > 0:
        for i in range(len(results)):
            result = results[i]
            y_min: float | None = None
            y_max: float | None = None
            ryel = result['yellow_liner']
            # print(ryel)
            yels: list[float] = []
            if ryel is not None:
                if len(ryel.shape) == 1:
                    yels = cast(list[float], ryel.tolist())
                    for y in yels:
                            if y_min is None or y<y_min:
                                y_min = y
                            if y_max is None or y>y_max:
                                y_max = y
                else:
                    logger.error(f"Yellow-liner result has wrong shape: {ryel.shape}")
                    continue
            else:
                print(results[i]['chunk'])
            if y_min == None:
                    y_min = 0
            if y_max == None:
                    y_max = 1
            ind = result['desc'].rfind('/')
            if ind!=-1:
                short_desc = result['desc'][ind+1:]
            else:
                short_desc = result['desc']
            title_text = f"Document: {short_desc}, {result['cosine'] * 100.0:2.1f} %"
            if len(title_text) < cols: 
                title_text += " " * (cols - len(title_text))
            else:
                title_text = title_text[:cols]
            console.print("[#FFFFFF on #D0D0D0]"+"-"*cols+"[/]")
            console.print("[black on #E0E0E0]"+title_text+"[/]")
            console.print("[#FFFFFF on #E0E0E0]"+"-"*cols+"[/]")
            # sys.stdout.flush()
            # print(best_chunk)
            # print(y_min, y_max)
            if y_min == y_max:
                    print(f"Search gave no meaningful result: y_min: {y_min}, y_max: {y_max}, search-embedding vector is trivial (language not supported?)")
                    print(result['chunk'])
                    continue
            if yels != []:
                line = ""
                char_ind = 0
                for i, c in enumerate(result['chunk']):
                    y_ind = i//context_steps
                    if y_ind >= len(yels):
                        print(f"Index out of range: {y_ind}, len(yels): {len(yels)}")
                    else:
                        yel:float = (yels[y_ind]-y_min)/(y_max - y_min)
                        if yel < 0.5:
                            yel = 0.0
                        col = hex(255 - int(yel*127.0))[2:]
                        if c == "\n":
                            rest = "[black on #FFFFFF]" + " " * (cols - char_ind%cols) + "[/]"
                            line += rest
                            char_ind = 0
                        else:
                            line += f"[black on #FFFF{col}]"+c+"[/]"
                            if ord(c)>31:
                                char_ind += 1
                if char_ind > 0:
                    rest = "[black on #FFFFFF]" + "-" * (cols - char_ind%cols) + "[/]"
                    line += rest
                    char_ind = 0
                console.print(line) # , soft_wrap=True)
        console.print("[#FFFFFF on #E0E0E0]"+"-"*cols+"[/]")
    else:
        print("No search result available!")
    
def iq_sync(its: IcoTqStore, _logger:logging.Logger, max_imports_str:str|None=None):
    if max_imports_str is not None:
        try:
            max_imports = int(max_imports_str)
        except ValueError:
            max_imports = None
    else:
        max_imports = None
    its.sync_texts(max_imports=max_imports)
    print()
    print("Use 'index' to update the current model's index, use 'list models' for an overview of models and indices available")

def iq_select(its: IcoTqStore, _logger:logging.Logger, model_id:str):
    try:
        ind = int(model_id)
        if ind > 0 and ind <= len(its.model_list):
            model_id = its.model_list[ind-1]['model_name']
    except ValueError:
        pass
    _ = its.load_model(model_id, its.config["embeddings_device"], its.config["embeddings_model_trust_code"])
    print()
    print("Use 'list models' for list of available models (from config/model_list.json), update current model's index with 'index'")

def serialize_gr_command(**cmd: Any):  # pyright: ignore[reportExplicitAny, reportAny]
    payload: bytes = cast(bytes, cmd.pop('payload', None))
    cmd_str: str = ','.join(f'{k}={v}' for k, v in cmd.items())  # pyright: ignore[reportAny]
    ans: list[bytes] = []
    w = ans.append
    w(b'\033_G')
    w(cmd_str.encode('ascii'))
    if payload:
        w(b';')
        w(payload)
    w(b'\033\\')
    return b''.join(ans)

def write_chunked(**cmd: Any):  # pyright: ignore[reportExplicitAny, reportAny]
    data: bytes = cast(bytes, cmd.pop('data'))
    while data:
        chunk, data = data[:4096], data[4096:]
        m = 1 if data else 0
        _ = sys.stdout.buffer.write(serialize_gr_command(payload=chunk, m=m, **cmd))
        _ = sys.stdout.flush()
        cmd.clear()

def iq_list(its: IcoTqStore, _logger:logging.Logger, param:str):
    sub_params = param.split(' ')
    if param == 'models':
        class ModelInfo(TypedDict):
            selected: bool
            indexed: int

        models: dict[str, ModelInfo] = {}
        doc_cnt: int = len(its.lib)
        part = False
        for ind, model in enumerate(its.model_list):
            if model['model_name'] == its.config["embeddings_model_name"]:
                models[model['model_name']] = ModelInfo({'selected': True, 'indexed': 0})
                sel:str = '[*]'
            else:
                models[model['model_name']] = ModelInfo({'selected': False, 'indexed': 0})
                sel = '   '
            for entry in its.lib:
                if model['model_name'] in entry["emb_ptrs"]:
                    models[model['model_name']]["indexed"] += 1
            if models[model['model_name']]["indexed"] == 0:
                index_state:str = "not indexed"
                part = True
            elif models[model['model_name']]["indexed"] == doc_cnt:
                index_state = "fully indexed"
            else:
                index_state = "partially indexed"
                part = True            
            print(f"{ind+1}: {sel} {model['model_name']}, Index: {models[model['model_name']]["indexed"]}/{doc_cnt}, {index_state} ")
        print()
        if part is True:
            print("To update the index, use 'select <model-number>' to select a model, then use 'index' to update that model's index")
        print("Use 'sync' to update new or changed documents from document sources ('list sources'), after syncing, the index needs to be updated with 'index'.")

    elif param == 'sources':
        for ind, source in enumerate(its.config["tq_sources"]):
            cnt = 0
            for entry in its.lib:
                if entry['source_name'] == source['name']:
                    cnt += 1
            print(f"{ind+1}: {source['name']} at {source['path']} ({source['tqtype']}), {cnt} docs")
        print("\nUse 'list models' for the current index status, use 'sync' to synchronized new or changed documents.")
    elif sub_params[0] == 'docs':            
        for ind, entry in enumerate(its.lib):
            found:bool = False
            if len(sub_params) == 1:
                found = True
            else:
                found = True
                for sp in sub_params[1:]:
                    if sp.lower() not in entry["desc_filename"].lower():
                        found = False
                        break
            if found is True:
                if entry['icon'] != '':
                    if os.environ.get('TERM', '').startswith('xterm-kitty'):
                        write_chunked(a='T', f=100, data=entry['icon'].encode('utf-8'))
                    else:
                        print("Terminal has no graphics support")
                else:
                    print("<no icon>")
                print(f"{ind+1} {entry['desc_filename']}")
    else:
        print("Usage either 'list models', 'list sources', or 'list docs'.")

def iq_check(its: IcoTqStore, _logger:logging.Logger, param:str=""):
    if param == "" or "pdf" in param:
        its.check_clean(dry_run=True)

def iq_clean(its: IcoTqStore, _logger:logging.Logger, param:str=""):
    if param == "" or "pdf" in param:
        its.check_clean(dry_run=False)

def iq_plot(its: IcoTqStore, _logger: logging.Logger, param: str = ""):
    pari = param.split(' ')
    
    max_points = None
    model_name = None
    
    if len(pari) > 0:
        try:
            max_points = int(pari[0])
            print(f"Limiting visualization to {max_points} points")
        except ValueError:
            # Not a number, might be a model name
            model_name = pari[0]
            
            if len(pari) > 1:
                try:
                    max_points = int(pari[1])
                    print(f"Limiting visualization to {max_points} points")
                except ValueError:
                    print(f"Warning: Invalid number '{pari[1]}' for max_points, using default")
    
    # Check if embeddings are available
    if its.embeddings_matrix is None:
        print("No embeddings available. Load a model and generate embeddings first.")
        return
    
    # Find the visualization page URL
    url = "http://localhost:8080/visualization.html"
    
    # Add query parameters if needed
    params = []
    if model_name:
        params.append(f"model={model_name}")
    if max_points:
        params.append(f"max={max_points}")
    
    if params:
        url += "?" + "&".join(params)
    
    # Open in browser
    print(f"Opening embedding visualization in browser: {url}")
    webbrowser.open(url)
    
    return url

def iq_help(parser:argparse.ArgumentParser, valid_actions:list[tuple[str, str]]):
    parser.print_help()
    print()
    print("Command can either be provided as command-line arguments or at the '> ' prompt.")
    print("Valid commands are: \n" + '\n    '.join([f"{command}: {help}" for command, help in valid_actions]))
    print()
    print("Start with 'sync' to synchronize the documents from your sources, select a model ('list models', then 'select <model-number>'), and index your documents with the selected model using 'index'.")
    print()
    print("To exit, use ^D, 'exit', or 'quit'")

def parse_cmd(its: IcoTqStore, logger: logging.Logger, args):
    """Parse and execute a command with arguments"""
    if not args:
        return
        
    cmd = args[0].lower()
    param = " ".join(args[1:]) if len(args) > 1 else ""
    
    # Command mapping
    cmd_map = {
        "info": lambda: iq_info(its, logger),
        "index": lambda: iq_index(its, logger, param),
        "search": lambda: iq_search(its, logger, param),
        "sync": lambda: iq_sync(its, logger, param if param else None),
        "select": lambda: iq_select(its, logger, param),
        "plot": lambda: iq_plot(its, logger, param),
        "list": lambda: iq_list(its, logger, param),
        # Add other commands here
    }
    
    if cmd in cmd_map:
        try:
            result = cmd_map[cmd]()
            return result
        except Exception as e:
            logger.error(f"Error executing command '{cmd}': {str(e)}")
            print(f"Error: {str(e)}")
    else:
        print(f"Unknown command: {cmd}")
        print("Available commands: " + ", ".join(cmd_map.keys()))

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--config-path", help="Path to config directory")
    parser.add_argument("--no-server", action="store_true", help="Don't start the web visualization server")
    parser.add_argument("--server-port", type=int, default=8080, help="Port for the web visualization server")
    
    args, remaining_args = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("iq")
    
    try:
        its = IcoTqStore(config_file_override=args.config, config_path_override=args.config_path)
        
        # Start visualization server unless disabled
        vis_server = None
        if not args.no_server:
            vis_server = EmbeddingVisServer(its, port=args.server_port)
            vis_server.start()
        
        if len(remaining_args) > 0:
            # Command-line mode
            parse_cmd(its, logger, remaining_args)
        else:
            # Interactive mode
            while True:
                try:
                    user_input = input("iq> ")
                    if user_input.lower() in ["exit", "quit", "q"]:
                        break
                    parse_cmd(its, logger, user_input.split())
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit")
        
        # Stop server when done
        if vis_server and vis_server.running:
            vis_server.stop()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

class EmbeddingVisServer:
    """Web server for embedding visualizations"""
    
    def __init__(self, icotq_store, host="localhost", port=8080):
        self.icotq_store = icotq_store
        self.host = host
        self.port = self._find_available_port(port)
        self.server = None
        self.server_thread = None
        self.running = False
        self.logger = logging.getLogger("EmbeddingVisServer")
        
        # Check that static directory exists
        self.static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        if not os.path.exists(self.static_dir):
            os.makedirs(self.static_dir)
            self.logger.warning(f"Created empty static directory at {self.static_dir}")
            print(f"Warning: Static directory created at {self.static_dir} but no files exist.")
            print("Please add the necessary HTML/JS/CSS files for visualization.")
    
    def _find_available_port(self, start_port):
        """Find an available port starting from start_port"""
        port = start_port
        max_port = start_port + 100
        
        while port < max_port:
            try:
                with socketserver.TCPServer(("", port), None) as s:
                    pass
                return port
            except OSError:
                port += 1
        
        raise RuntimeError(f"Could not find an available port between {start_port} and {max_port}")
    
    def start(self):
        """Start the web server in a background thread"""
        if self.running:
            self.logger.info(f"Server already running at http://{self.host}:{self.port}")
            return
        
        # Create request handler class with access to icotq_store
        icotq_store_ref = self.icotq_store
        static_dir_ref = self.static_dir
        
        class APIHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.icotq_store = icotq_store_ref
                super().__init__(*args, directory=static_dir_ref, **kwargs)
            
            def do_GET(self):
                """Handle GET requests"""
                # Parse URL
                parsed_url = urlparse(self.path)
                path = parsed_url.path
                
                # API endpoints
                if path.startswith('/api/'):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    # Handle different API endpoints
                    if path == '/api/embeddings':
                        query = parse_qs(parsed_url.query)
                        model_name = query.get('model', [None])[0]
                        max_points = query.get('max', [None])[0]
                        
                        if max_points:
                            try:
                                max_points = int(max_points)
                            except ValueError:
                                max_points = None
                        
                        # Generate embedding data
                        data = self._generate_embedding_data(model_name, max_points)
                        self.wfile.write(json.dumps(data).encode())
                    
                    elif path == '/api/models':
                        # Return list of available models
                        models = self.icotq_store.list_models(return_result=True)
                        self.wfile.write(json.dumps(models).encode())
                    
                    else:
                        # Unknown API endpoint
                        self.send_error(404, "API endpoint not found")
                
                # Serve static files
                else:
                    return super().do_GET()
            
            def _generate_embedding_data(self, model_name=None, max_points=None):
                """Generate embedding data for visualization"""
                import numpy as np
                import colorsys
                import warnings
                
                # Suppress deprecation warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")
                    
                    # Ensure model is loaded
                    if model_name and model_name != self.icotq_store.current_model.get('model_name'):
                        try:
                            self.icotq_store.load_model(model_name)
                        except Exception as e:
                            return {"error": f"Failed to load model: {str(e)}"}
                    
                    if self.icotq_store.embeddings_matrix is None:
                        return {"error": "No embeddings available. Load a model and generate embeddings first."}
                    
                    # Extract embeddings
                    embeddings = self.icotq_store.embeddings_matrix.cpu().numpy()
                    
                    # Limit points if needed
                    if max_points and embeddings.shape[0] > max_points:
                        indices = np.random.choice(embeddings.shape[0], max_points, replace=False)
                        indices.sort()  # Keep order for better document correlation
                        embeddings = embeddings[indices]
                    
                    # Build document mapping
                    doc_mapping = []
                    doc_ids = []
                    chunk_texts = []
                    unique_docs = set()
                    
                    for entry in self.icotq_store.lib:
                        if self.icotq_store.current_model['model_name'] in entry.get('emb_ptrs', {}):
                            start, length = entry['emb_ptrs'][self.icotq_store.current_model['model_name']]
                            doc_id = entry['desc_filename']
                            unique_docs.add(doc_id)
                            
                            # Only process chunks within our embedding limits
                            actual_length = min(length, embeddings.shape[0] - start)
                            for i in range(actual_length):
                                idx = start + i
                                if idx >= embeddings.shape[0]:
                                    break
                                
                                doc_mapping.append((idx, doc_id))
                                doc_ids.append(doc_id)
                                chunk_text = self.icotq_store.get_chunk(
                                    entry['text'], i, 
                                    self.icotq_store.current_model['chunk_size'], 
                                    self.icotq_store.current_model['chunk_overlap']
                                )
                                chunk_texts.append(chunk_text[:200])  # Limit text size
                    
                    # Dimensionality reduction
                    from sklearn.decomposition import PCA
                    import umap
                    
                    # PCA first for efficiency
                    if embeddings.shape[1] > 50:
                        pca = PCA(n_components=50)
                        embeddings_reduced = pca.fit_transform(embeddings)
                    else:
                        embeddings_reduced = embeddings
                    
                    # UMAP with parallel processing
                    reducer = umap.UMAP(
                        n_components=3, 
                        metric='cosine', 
                        n_neighbors=15, 
                        min_dist=0.1,
                        n_jobs=-1,
                        low_memory=False
                    )
                    embeddings_3d = reducer.fit_transform(embeddings_reduced)
                    
                    # Assign colors
                    unique_doc_list = list(unique_docs)
                    doc_color_map = {}
                    
                    for i, doc in enumerate(unique_doc_list):
                        # Generate a color from HSV for better distribution
                        hue = i / len(unique_doc_list)
                        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                        doc_color_map[doc] = [float(c) for c in rgb]
                    
                    colors = [doc_color_map[doc_id] for doc_id in doc_ids]
                    
                    # Prepare output data
                    output_data = {
                        "points": embeddings_3d.tolist(),
                        "colors": colors,
                        "docs": doc_ids,
                        "texts": chunk_texts,
                        "doc_map": {doc: i for i, doc in enumerate(unique_doc_list)},
                        "model_name": self.icotq_store.current_model['model_name'],
                        "total_points": len(embeddings_3d)
                    }
                    
                    return output_data
        
        # Create and start the server
        self.server = socketserver.ThreadingTCPServer((self.host, self.port), APIHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.running = True
        self.logger.info(f"Visualization server started at http://{self.host}:{self.port}")
        print(f"Visualization server running at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the web server"""
        if self.server and self.running:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            self.logger.info("Visualization server stopped")
            print("Visualization server stopped")

if __name__ == "__main__":
    main()