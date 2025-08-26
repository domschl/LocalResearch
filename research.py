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
            ind = line.find(' ')
            if ind != -1:
                command = line[:ind].lower()
                arguments = line[ind+1:]
            else:
                command = line.lower()
                arguments = ""
            
            if command == 'sync':
                log.info("Starting sync...")
                ds.sync_texts()
            elif command == 'check':
                ds.check(arguments)
            elif command == 'list':
                ds.list(arguments)
                vs.list(arguments)
            elif command == 'select':
                ind: int = -1
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
            elif command == 'index':
                vs.index(ds.library)
            elif command == 'search':
                vs.search(arguments, ds.library)
            elif command == 'help':
                print("Use 'list [models|sources]', 'sync', 'check [pdf]', 'select [model-index]'")
            elif command == 'exit' or command == 'quit':
                break

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D to exit gracefully
            break
    

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("ResearchCLI")

    ds = DocumentStore()
    vs = VectorStore(ds.storage_path, ds.config_path)
    repl(ds, vs, log)
    
if __name__ == "__main__":
    main()
