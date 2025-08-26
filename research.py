import logging
import readline
import os
import atexit
from vector_store import VectorStore

def repl(its: VectorStore, log: logging.Logger):
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
                its.sync_texts()
            elif command == 'check':
                its.check(arguments)
            elif command == 'list':
                its.list(arguments)
            elif command == 'select':
                ind: int = -1
                try:
                    ind = int(arguments)
                except:
                    pass
                if ind != -1:
                    its.select(ind)
                else:
                    log.error(f"Invalid index {arguments}, integer required, use 'list models' for valid range")
            elif command == 'exit' or command == 'quit':
                break

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D to exit gracefully
            break
    

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("VectorCLI")

    its = VectorStore()
    repl(its, log)
    
if __name__ == "__main__":
    main()
