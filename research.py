
import logging
import readline
import os
import atexit
from typing import cast

print("\rStarting...\r", end="", flush=True)

from vector_store import DocumentStore, VectorStore

def repl(ds: DocumentStore, vs: VectorStore, log: logging.Logger):
    history_file = os.path.join(os.path.expanduser("~/.config/local_research"), "repl_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Read history: {e}")
        
    readline.set_history_length(1000)
    _ = atexit.register(readline.write_history_file, history_file)
    while True:
        try:
            # Get user input with a prompt
            line: str = input(">>> ")
            ind:int = line.find(' ')
            if ind != -1:
                command = line[:ind].lower()
                arguments = line[ind+1:]
            else:
                command = line.lower()
                arguments = ""
            
            if command == 'sync':
                log.info("Starting sync...")
                ds.sync_texts(arguments)
            elif command == 'check':
                ds.check(arguments)
                doc_hashes: list[str] = list(ds.library.keys())
                vs.check(arguments, doc_hashes)
            elif command == 'list':
                ds.list_info(arguments)
                vs.list_info(arguments)
            elif command == 'set':
                comps = arguments.split(' ')
                if len(comps) != 2:
                    log.error('Usage: set <name> <value>')
                    log.info("Use list of list of known variable names and types")
                else:
                    _ = ds.set_var(comps[0], comps[1])
            elif command == 'select':
                ind = -1
                try:
                    ind = int(arguments)
                except:
                    pass
                if ind != -1:
                    new_model = vs.select(ind)
                    if new_model is None:
                        log.info("Model unchanged.")
                    else:
                        log.info(f"New active model is {new_model}")
                else:
                    log.error(f"Invalid ID {arguments}, integer required, use 'list models' for valid range")
            elif command == 'enable':
                ind = -1
                try:
                    ind = int(arguments)
                except:
                    pass
                if ind != -1:
                    new_model = vs.enable(ind)
                else:
                    log.error(f"Invalid ID {arguments}, integer required, use 'list models' for valid range")
            elif command == 'disable':
                ind = -1
                try:
                    ind = int(arguments)
                except:
                    pass
                if ind != -1:
                    new_model = vs.disable(ind)
                else:
                    log.error(f"Invalid ID {arguments}, integer required, use 'list models' for valid range")
            elif command == 'index':
                if ds.local_update_required() is True:
                    if 'force' not in arguments.split(' '):
                        log.warning("Remote data is newer than local data! Please use import first! (or override with 'force'")
                        continue
                    else:
                        log.warning("Version override, starting indexing")
                if 'all' in arguments.split(' '):
                    vs.index_all(ds.library)
                else:
                    vs.index(ds.library)
            elif command == 'search':
                search_results: int = cast(int, ds.get_var('search_results'))
                highlight: bool = cast(bool, ds.get_var('highlight'))
                cutoff = cast(float, ds.get_var('highlight_cutoff'))
                damp:float = cast(float, ds.get_var('highlight_dampening'))
                vs.search(arguments, ds.library, search_results, highlight, cutoff, damp)
            elif command == 'publish':
                _ = ds.publish(arguments)
            elif command == 'import':
                if ds.import_local(arguments) is True:
                    log.info("Import successful, reloading data...")
                    del ds
                    del vs
                    ds = DocumentStore()
                    vs = VectorStore(ds.storage_path, ds.config_path)
                else:
                    log.error("Import failed")
            elif command == 'help':
                print("Use 'list [models|sources]', 'sync', 'check [index|pdf] [clean]', 'select <model-ID>', 'index [force] [all]', 'search <search-string>', 'publish', 'import', set <var-name> <value>")
            elif command == 'exit' or command == 'quit':
                break

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D to exit gracefully
            break
    

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("ResearchCLI")
    log.info("Local Research v1.0")

    ds = DocumentStore()
    vs = VectorStore(ds.storage_path, ds.config_path)
    repl(ds, vs, log)
    
if __name__ == "__main__":
    main()
