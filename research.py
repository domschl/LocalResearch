import logging
import readline
import os
import atexit
from vector_store import DocumentStore, VectorStore

def repl(ds: DocumentStore, vs: VectorStore, log: logging.Logger):
    history_file = os.path.join(os.path.expanduser("~/.config/vector_store"), "repl_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
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
            elif command == 'list':
                ds.list(arguments)
                vs.list(arguments)
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
                    log.error(f"Invalid index {arguments}, integer required, use 'list models' for valid range")
            elif command == 'enable':
                ind = -1
                try:
                    ind = int(arguments)
                except:
                    pass
                if ind != -1:
                    new_model = vs.enable(ind)
                else:
                    log.error(f"Invalid index {arguments}, integer required, use 'list models' for valid range")
            elif command == 'disable':
                ind = -1
                try:
                    ind = int(arguments)
                except:
                    pass
                if ind != -1:
                    new_model = vs.disable(ind)
                else:
                    log.error(f"Invalid index {arguments}, integer required, use 'list models' for valid range")
            elif command == 'index':
                if ds.local_update_required() is True:
                    if 'force' not in arguments.split(' '):
                        log.warning("Remote data is newer than local data! Please use import first! (or override with 'force'")
                        continue
                    else:
                        log.warning("Version override, starting indexing")
                vs.index(ds.library)
            elif command == 'search':
                vs.search(arguments, ds.library)
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
                print("Use 'list [models|sources]', 'sync', 'check [pdf]', 'select [model-index]', 'index', 'search <search-string>', 'publish', 'import'")
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
