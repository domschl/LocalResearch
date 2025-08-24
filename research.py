import logging
import os
import sys
import argparse
from argparse import ArgumentParser
from rich.console import Console
from vector_store import VectorStore, SearchResult
from typing import cast, TypedDict, Any
import json
import shutil
import pathlib
import argparse # Ensure argparse is imported

def vec_info(its: VectorStore, _logger:logging.Logger) -> None:
    its.list_sources()
    print()
    print("Use 'list models' for an overview of available models and their index status")
    print("Use 'sync' to load new/updated documents from all sources")
    print("Use 'index' to index using the current model (Use 'sync' first to load new/updated documents)")
    print("Use 'search <search phrase>' to search using the current model's index")

def vec_index(its: VectorStore, _logger:logging.Logger, param:str):
    if param == 'purge':
        purge = True
    else:
        purge = False
    its.generate_embeddings(purge=purge)

def vec_search(its: VectorStore, logger:logging.Logger, search_spec: str):
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
    
def vec_sync(its: VectorStore, _logger:logging.Logger, max_imports_str:str|None=None):
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

def vec_select(its: VectorStore, _logger:logging.Logger, model_id:str):
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

def vec_list(its: VectorStore, _logger:logging.Logger, param:str):
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
        for ind, source in enumerate(its.config["vec_sources"]):
            cnt = 0
            for entry in its.lib:
                if entry['source_name'] == source['name']:
                    cnt += 1
            print(f"{ind+1}: {source['name']} at {source['path']} ({source['vectype']}), {cnt} docs")
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

def vec_check(its: VectorStore, _logger:logging.Logger, param:str=""):
    if param == "" or "pdf" in param:
        its.check_clean(dry_run=True)

def vec_clean(its: VectorStore, _logger:logging.Logger, param:str=""):
    if param == "" or "pdf" in param:
        its.check_clean(dry_run=False)

def vec_export(its: VectorStore, logger: logging.Logger, params_str: str):
    parser = argparse.ArgumentParser(description="Export data for web server.")
    _ = parser.add_argument("--output_dir", default="web_server/data", help="Base directory to export the data to.")
    _ = parser.add_argument("--max_points", type=int, help="Maximum number of points for visualization.", default=None)
    # No --model_name argument, will use current model from VectorStore

    try:
        args = parser.parse_args(params_str.split())
    except SystemExit:
        logger.error("Invalid export parameters.")
        # print("Invalid export parameters. Use 'export <output_dir> [--max_points N]'")
        return

    current_model_name = its.get_current_model_name()
    if not current_model_name:
        logger.error("No model is currently active. Please select a model using 'select <model_id>' first.")
        # print("No model is currently active. Please select a model using 'select <model_id>' first.")
        return

    output_base_dir = pathlib.Path(cast(str, args.output_dir))
    try:
        output_base_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating base directory {output_base_dir}: {e}")
        # print(f"Error creating base directory {output_base_dir}: {e}")
        return

    logger.info(f"Exporting data for current model: {current_model_name} to {output_base_dir}")

    # 1. Copy shared vector_library.json to the root of output_dir
    try:
        library_source_path_str = its.get_library_path()
        if library_source_path_str:
            library_source_path = pathlib.Path(library_source_path_str)
            library_dest_path = output_base_dir / "vector_library.json"
            _ = shutil.copy2(library_source_path, library_dest_path)
            logger.info(f"Exported shared library to {library_dest_path}")
            # print(f"Exported shared library to {library_dest_path}")
        else:
            logger.warning("Could not determine library path. Skipping library export.")
            # print("Could not determine library path. Skipping library export.")
    except Exception as e:
        logger.error(f"Error exporting shared library: {e}")
        # print(f"Error exporting shared library: {e}")

    # Create model-specific subdirectory
    model_export_dir = output_base_dir / current_model_name
    try:
        model_export_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating model directory {model_export_dir}: {e}")
        # print(f"Error creating model directory {model_export_dir}: {e}")
        return

    # 2. Prepare and save model-specific visualization_data.json
    try:
        logger.info(f"Preparing visualization data for {current_model_name} with max_points={cast(int, args.max_points)}...")
        vis_data = its.prepare_visualization_data(
            max_points=cast(int, args.max_points)
        )
        if vis_data:
            vis_data_path = model_export_dir / "visualization_data.json"
            with open(vis_data_path, 'w') as f:
                json.dump(vis_data, f)
            logger.info(f"Exported visualization data to {vis_data_path}")
            # print(f"Exported visualization data to {vis_data_path}")
        else:
            logger.error("Failed to generate visualization data.")
            # print("Failed to generate visualization data.")
    except Exception as e:
        logger.error(f"Error preparing or saving visualization data: {e}")
        # print(f"Error preparing or saving visualization data: {e}")

    # 3. Copy model-specific embeddings tensor
    try:
        tensor_source_path_str = its._get_tensor_path(current_model_name)
        if tensor_source_path_str:
            tensor_source_path = pathlib.Path(tensor_source_path_str)
            # Ensure the source tensor file exists
            if not tensor_source_path.is_file():
                logger.error(f"Embeddings tensor file not found at {tensor_source_path}")
                # print(f"Embeddings tensor file not found at {tensor_source_path}")
            else:
                tensor_dest_path = model_export_dir / "embeddings.pt" # Generic name within model folder
                shutil.copy2(tensor_source_path, tensor_dest_path)
                logger.info(f"Exported embeddings tensor to {tensor_dest_path}")
                # print(f"Exported embeddings tensor to {tensor_dest_path}")
        else:
            logger.warning(f"Could not determine tensor path for model {current_model_name}. Skipping tensor export.")
            # print(f"Could not determine tensor path for model {current_model_name}. Skipping tensor export.")
    except Exception as e:
        logger.error(f"Error exporting embeddings tensor: {e}")
        # print(f"Error exporting embeddings tensor: {e}")

    logger.info(f"Export completed for model {current_model_name}. Files are in {output_base_dir} and {model_export_dir}")
    # print(f"Export completed for model {current_model_name}. Files are in {output_base_dir} and {model_export_dir.resolve()}")

def vec_help(parser:argparse.ArgumentParser, valid_actions:list[tuple[str, str]]):
    parser.print_help()
    print()
    print("Command can either be provided as command-line arguments or at the '> ' prompt.")
    print("Valid commands are: \n" + '\n    '.join([f"{command}: {help}" for command, help in valid_actions]))
    print()
    print("Start with 'sync' to synchronize the documents from your sources, select a model ('list models', then 'select <model-number>'), and index your documents with the selected model using 'index'.")
    print()
    print("To exit, use ^D, 'exit', or 'quit'")

def parse_cmd(its: VectorStore, logger: logging.Logger, args):
    """Parse and execute a command with arguments"""
    if not args:
        return
        
    cmd = args[0].lower()
    param = " ".join(args[1:]) if len(args) > 1 else ""
    
    # Command mapping
    cmd_map = {
        "info": lambda: vec_info(its, logger),
        "index": lambda: vec_index(its, logger, param),
        "search": lambda: vec_search(its, logger, param),
        "sync": lambda: vec_sync(its, logger, param if param else None),
        "select": lambda: vec_select(its, logger, param),
        "export": lambda: vec_export(its, logger, param),
        "list": lambda: vec_list(its, logger, param),
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
    # Removed --no-server and --server-port arguments
    
    args, remaining_args = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("vec")
    
    try:
        its = VectorStore(config_file_override=args.config, config_path_override=args.config_path)
        
        # Removed EmbeddingVisServer instantiation and start/stop logic
        
        if len(remaining_args) > 0:
            # Command-line mode
            parse_cmd(its, logger, remaining_args)
        else:
            # Interactive mode
            while True:
                try:
                    user_input = input("vec> ")
                    if user_input.lower() in ["exit", "quit", "q"]:
                        break
                    parse_cmd(its, logger, user_input.split())
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
