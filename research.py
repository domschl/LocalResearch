import logging

from vector_store import VectorStore

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("VectorCLI")

    its = VectorStore()
    
if __name__ == "__main__":
    main()
