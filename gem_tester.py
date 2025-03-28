import os
import sys
import random
import time
import tempfile
import shutil
import logging
import uuid
import json # Make sure json is imported

# Assume icotq_store.py is in the same directory or PYTHONPATH
try:
    # Import necessary components (typing import likely not needed now)
    from icotq_store import (
        IcoTqStore,
        IcotqConfig,
        TqSource,
        IcotqError,
        IcotqCriticalError,
        IcotqConsistencyError,
        IcotqConfigurationError
    )
except ImportError:
    print("ERROR: Could not import IcoTqStore. Make sure 'icotq_store.py' is accessible.")
    sys.exit(1)

# --- Configuration ---
NUM_ITERATIONS = 50
MAX_FILES = 15
MIN_FILES = 5
TEST_DIR_PREFIX = "icotq_test_"
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Logging Setup ---
# ... (logging setup remains the same) ...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger("TestApp")
logging.getLogger("IcoTqStore").setLevel(logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# --- Helper Functions ---

def generate_unique_token(filename: str) -> str:
    """Generates a unique, searchable token based on the filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return f"unique_token_{base.replace(' ', '_')}"

def generate_content(filename: str, version: int = 1) -> str:
    """Generates simple, searchable text content for a file."""
    unique_token = generate_unique_token(filename)
    return f"""
    This is text content for file '{os.path.basename(filename)}'. Version {version}.
    Unique phrase: {unique_token}.
    Random words: apple banana cherry date. Timestamp: {time.time()}
    """

# Return type changed to str | None
def create_file(directory: str, existing_files: list[str]) -> str | None:
    """Creates a new .txt file with generated content."""
    if len(existing_files) >= MAX_FILES:
        log.debug("Max files reached, skipping add.")
        return None
    base_name = f"file_{uuid.uuid4().hex[:8]}.txt"
    filepath = os.path.join(directory, base_name)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(generate_content(filepath, version=1))
        log.info(f"Created file: {base_name}")
        return filepath
    except IOError as e:
        log.error(f"Failed to create file {filepath}: {e}")
        return None

# Return type changed to str | None
def modify_file(directory: str, existing_files: list[str]) -> str | None:
    """Modifies the content of a randomly chosen existing file."""
    if not existing_files:
        log.debug("No files exist to modify, skipping modify.")
        return None
    filepath_to_modify = random.choice(existing_files)
    try:
        current_version = 1
        try:
            with open(filepath_to_modify, "r", encoding="utf-8") as f:
                content = f.read()
                if "Version " in content: # Match exact case from generate_content
                    num_str = content.split("Version ")[1].split(".")[0]
                    current_version = int(num_str)
        except (IOError, ValueError, IndexError): pass

        new_version = current_version + 1
        with open(filepath_to_modify, "w", encoding="utf-8") as f:
            f.write(generate_content(filepath_to_modify, version=new_version))
        log.info(f"Modified file: {os.path.basename(filepath_to_modify)} (v{new_version})")
        return filepath_to_modify
    except IOError as e:
        log.error(f"Failed to modify file {filepath_to_modify}: {e}")
        return None

# Return type changed to str | None
def delete_file(directory: str, existing_files: list[str]) -> str | None:
    """Deletes a randomly chosen existing file."""
    if len(existing_files) <= MIN_FILES:
        log.debug(f"Min files ({MIN_FILES}) reached, skipping delete.")
        return None
    if not existing_files:
        log.debug("No files exist to delete, skipping delete.")
        return None

    filepath_to_delete = random.choice(existing_files)
    try:
        os.remove(filepath_to_delete)
        log.info(f"Deleted file: {os.path.basename(filepath_to_delete)}")
        return filepath_to_delete
    except OSError as e:
        log.error(f"Failed to delete file {filepath_to_delete}: {e}")
        return None

# --- Main Test Function ---

def run_test_loop():
    """Sets up IcoTqStore and runs the test loop."""
    sync_errors = 0; index_errors = 0; search_errors = 0; validation_failures = 0
    base_dir: str | None = None # Initialize base_dir
    store: IcoTqStore | None = None # Changed Optional

    try:
        # 1. Setup Temporary Directories
        base_dir = tempfile.mkdtemp(prefix="icotq_base_")
        source_dir = os.path.join(base_dir, "source_files")
        store_dir = os.path.join(base_dir, "store_data")
        os.makedirs(source_dir); os.makedirs(store_dir)
        log.info(f"Created base test directory: {base_dir}")
        log.info(f"Source files directory: {source_dir}")
        log.info(f"IcoTqStore data directory: {store_dir}")

        # 2. Create Initial Files
        current_files: list[str] = [] # Changed from List[str]
        for _ in range(MIN_FILES + 2):
             new_file = create_file(source_dir, current_files)
             if new_file: current_files.append(new_file)

        # 3. Configure and Initialize IcoTqStore
        log.info("Initializing IcoTqStore...")
        temp_config = IcotqConfig({
            'icotq_path': store_dir,
            'tq_sources': [TqSource({'name': 'TestSource', 'tqtype': 'folder', 'path': source_dir, 'file_types': ['txt']})],
            'embeddings_model_name': MODEL_NAME, 'embeddings_device': 'auto',
            'embeddings_model_trust_code': True, 'auto_fix_inconsistency': True
        })
        temp_config_path = os.path.join(base_dir, "temp_test_config.json")
        with open(temp_config_path, 'w') as f: json.dump(temp_config, f)
        log.info(f"Created temporary config file: {temp_config_path}")

        try:
            store = IcoTqStore(config_file_override=temp_config_path)
        except IcotqError as init_e:
            log.critical(f"IcoTqStore initialization failed: {init_e}", exc_info=True); raise

        # --- Verification after __init__ ---
        if store.current_model and store.engine:
             log.info(f"IcoTqStore initialized SUCCESSFULLY. Model: {store.current_model['model_name']}")
        else:
             log.critical(f"IcoTqStore init FAILED TO LOAD MODEL.")
             log.info("Attempting explicit load_model post-init...")
             load_success = store.load_model(MODEL_NAME, store.config['embeddings_device'], store.config['embeddings_model_trust_code'])
             if load_success and store.current_model and store.engine: log.info(f"Explicit load_model SUCCEEDED.")
             else: log.critical("Explicit load_model FAILED. Aborting."); raise IcotqError("Failed to load embedding model.")

        # 4. Initial Sync and Index
        log.info("Performing initial sync...")
        try: store.sync_texts()
        except IcotqError as e: log.critical(f"Initial sync failed: {e}"); raise
        log.info("Performing initial index...")
        try: store.generate_embeddings(save_every_sec=0)
        except IcotqError as e: log.critical(f"Initial index generation failed: {e}"); raise

        # --- 5. Main Test Loop ---
        for i in range(NUM_ITERATIONS):
            log.info(f"--- Iteration {i+1}/{NUM_ITERATIONS} ---")
            action_taken: str | None = None # Changed Optional
            affected_file_path: str | None = None # Changed Optional
            current_files_basenames: list[str] = [os.path.basename(f) for f in current_files] # Changed List

            action = random.choice(["add", "modify", "delete", "none"])
            if action == "add" and len(current_files) < MAX_FILES:
                affected_file_path = create_file(source_dir, current_files)
                if affected_file_path: current_files.append(affected_file_path); action_taken = "add"
            elif action == "modify" and current_files:
                affected_file_path = modify_file(source_dir, current_files)
                if affected_file_path: action_taken = "modify"
            elif action == "delete" and len(current_files) > MIN_FILES:
                # Need to re-assign result of delete_file to use it later
                filepath_to_delete = delete_file(source_dir, current_files)
                if filepath_to_delete:
                    affected_file_path = filepath_to_delete # Store the path of the deleted file
                    current_files.remove(filepath_to_delete)
                    action_taken = "delete"
            else: log.info("No file system change."); action_taken = "none"

            log.info("Running sync_texts...")
            try: store.sync_texts()
            except (IcotqCriticalError, IcotqConsistencyError) as e: log.error(f"Sync failed IcotqError: {e}", exc_info=True); sync_errors += 1
            except Exception as e: log.error(f"Sync unexpected error: {e}", exc_info=True); sync_errors += 1

            log.info("Running generate_embeddings...")
            try: store.generate_embeddings(save_every_sec=0)
            except (IcotqCriticalError, IcotqConsistencyError) as e: log.error(f"Index failed IcotqError: {e}", exc_info=True); index_errors += 1
            except Exception as e: log.error(f"Index unexpected error: {e}", exc_info=True); index_errors += 1

            if (i + 1) % 5 == 0:
                 log.info("Running check_clean(dry_run=False)...")
                 try: store.check_clean(dry_run=False)
                 except (IcotqCriticalError, IcotqConsistencyError) as e: log.error(f"check_clean failed IcotqError: {e}", exc_info=True)
                 except Exception as e: log.error(f"check_clean unexpected error: {e}", exc_info=True)

            if current_files:
                file_to_search_for = random.choice(current_files)
                base_name_to_search = os.path.basename(file_to_search_for)
                expected_desc = "{TestSource}" + base_name_to_search
                search_term = generate_unique_token(file_to_search_for)
                log.info(f"Searching for '{base_name_to_search}' using term: '{search_term}'")

                try:
                    search_results = store.search(search_term, max_results=5)
                    log.info(f"Search returned {len(search_results)} results.")
                    found_expected = False
                    if search_results:
                        log.info(f"Top result: desc='{search_results[0]['desc']}', score={search_results[0]['cosine']:.4f}")
                        if any(res['desc'] == expected_desc for res in search_results):
                             found_expected = True
                             log.info(f"Validation PASSED: Found '{expected_desc}'.")
                    if not found_expected:
                        log.warning(f"Validation FAILED: Did not find '{expected_desc}' for term '{search_term}'.")
                        validation_failures += 1
                except IcotqError as e: log.error(f"Search failed IcotqError: {e}", exc_info=True); search_errors += 1
                except Exception as e: log.error(f"Search unexpected error: {e}", exc_info=True); search_errors += 1
            else: log.info("Skipping search validation (no files).")
            time.sleep(0.1)

    except (IcotqCriticalError, IcotqConsistencyError, IcotqConfigurationError) as e:
         log.critical(f"Test loop aborted critical IcoTqStore error: {e}", exc_info=True)
    except Exception as e:
        log.critical(f"Test loop aborted unexpected error: {e}", exc_info=True)
    finally:
        if base_dir and os.path.exists(base_dir):
            try: shutil.rmtree(base_dir); log.info(f"Cleaned up test directory: {base_dir}")
            except Exception as e: log.error(f"Failed to clean up {base_dir}: {e}")

        log.info("--- Test Summary ---")
        log.info(f"Sync Errors: {sync_errors}"); log.info(f"Index Errors: {index_errors}"); log.info(f"Search Errors: {search_errors}"); log.info(f"Validation Failures: {validation_failures}")
        if sync_errors == 0 and index_errors == 0 and search_errors == 0 and validation_failures == 0: log.info("Overall Result: PASSED")
        else: log.error("Overall Result: FAILED")

# --- Run the Test ---
if __name__ == "__main__":
    run_test_loop()