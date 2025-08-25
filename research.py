import logging
import readline
import os
import atexit


# Set the maximum number of lines to be saved in the history file

print("Welcome to the simple Python REPL!")
print("Enter 'exit' to quit.")


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
            line = input(">>> ")
            if line.lower() == 'sync':
                log.info("Starting sync...")
                its.sync_texts()
                
            if line.lower() == 'exit':
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
