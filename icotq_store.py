```python
# --- START OF FILE icotq_store_refactored.py ---

import logging
import os
import json
import uuid
import time
import numpy as np
import aiohttp
import aiohttp.web
import asyncio
import threading
import tempfile
import traceback
from contextlib import contextmanager

import torch
from sentence_transformers import SentenceTransformer

from typing import TypedDict, cast, NotRequired, Generator, Any
import pymupdf  # pyright: ignore[reportMissingTypeStubs]


class TqSource(TypedDict):
    name: str
    tqtype: str
    path: str
    file_types: list[str]

class IcotqConfig(TypedDict):
    icotq_path: str
    tq_sources: list[TqSource]
    embeddings_model_name: str
    embeddings_device: str
    embeddings_model_trust_code: bool
    auto_fix_inconsistency: bool # New flag

class LibEntry(TypedDict):
    source_name: str
    filename: str
    desc_filename: str
    text: str
    emb_ptrs: dict[str, tuple[int, int]]   # model_name -> (emb_ptr, emb_len)

class PDFIndex(TypedDict):
    previous_failure: bool
    filename: str
    file_size: int

class SearchRequest(TypedDict):
    search_text: str
    max_results: NotRequired[int]
    yellow_liner: NotRequired[bool]
    context_length: NotRequired[int]
    context_steps: NotRequired[int]
    compression_mode: NotRequired[str]

class SearchResult(TypedDict):
    cosine: float
    index: int
    offset: int
    desc: str
    text: str
    chunk: str
    yellow_liner: np.typing.NDArray[np.float32] | None

# See: https://huggingface.co/spaces/mteb/leaderboard
# nomic-ai/nomic-embed-text-v2-moe
class EmbeddingsModel(TypedDict):
    model_hf_name: str
    model_name: str
    emb_dim: int
    max_input_token: int
    chunk_size: int
    chunk_overlap: int

# --- Custom Exceptions ---
class IcotqError(Exception):
    """Base exception for IcoTqStore errors."""
    pass

class IcotqConsistencyError(IcotqError):
    """Indicates a recoverable inconsistency between library and tensors."""
    pass

class IcotqCriticalError(IcotqError):
    """Indicates a critical or potentially unrecoverable error."""
    pass

class IcotqConfigurationError(IcotqError):
    """Indicates an error in the configuration."""
    pass


class IcoTqStore:
    def __init__(self) -> None:
        self.log:logging.Logger = logging.getLogger("IcoTqStore")
        # Disable log spam
        tmp = logging.getLogger("transformers_modules")
        tmp.setLevel(logging.ERROR)
        tmp_st = logging.getLogger("sentence_transformers")
        tmp_st.setLevel(logging.WARNING)

        config_path = os.path.expanduser("~/.config/icotq")  # Turquoise icosaeder
        if not os.path.isdir(config_path):
            os.makedirs(config_path)

        # --- Core State ---
        self.lib: list[LibEntry] = []
        self.pdf_index:dict[str, PDFIndex] = {}
        self.config_file:str = os.path.join(config_path, "icoqt.json")
        self.config:IcotqConfig
        self.current_model: EmbeddingsModel | None = None
        self.engine: SentenceTransformer | None = None
        self.device: str | None = None
        self.embeddings_matrix: torch.Tensor | None = None # In-memory tensor for the current_model
        self.root_path:str = ""
        self.embeddings_path: str = ""
        self.model_list: list[EmbeddingsModel] = []

        # --- Concurrency Control ---
        # Lock protecting self.lib, self.pdf_index, self.embeddings_matrix (loading/saving/modification),
        # file system operations (sync, index, clean), and model loading.
        self._lock = threading.Lock()

        # --- Server State ---
        self.server_running:bool = False
        self.loop:asyncio.AbstractEventLoop | None = None
        self.server_thread:threading.Thread | None = None

        self._load_or_init_config()
        self._validate_config_paths()
        self._load_or_init_model_list()
        self._ensure_storage_dirs()

        # --- Load Initial State (Inside Lock for safety if config loading triggers model load) ---
        with self._lock:
            if self.config['embeddings_model_name']:
                try:
                    # Use internal method that assumes lock is held
                    self._load_model_internal(self.config['embeddings_model_name'],
                                            self.config['embeddings_device'],
                                            self.config['embeddings_model_trust_code'])
                except IcotqError as e:
                    self.log.error(f"Failed to load initial model specified in config: {e}")
                    # Proceed without a model loaded

            self._read_library_internal() # Load library and PDF index
            if self.current_model:
                try:
                    # Load tensor for the current model
                    self._load_tensor_internal(model_name=self.current_model['model_name'])
                except IcotqConsistencyError as e:
                    self.log.error(f"Consistency Error loading tensor for initial model '{self.current_model['model_name']}': {e}")
                    if self.config.get('auto_fix_inconsistency', False):
                        self.log.warning(f"Attempting automatic fix (re-index) for model '{self.current_model['model_name']}' due to inconsistency.")
                        try:
                            # Need to release lock temporarily if generate_embeddings acquires it
                            # Or make generate_embeddings take a flag to know lock is held
                            # For now, let's assume _generate_embeddings_internal exists
                            self._generate_embeddings_internal(purge=True, model_name_override=self.current_model['model_name'])
                            self.log.info(f"Automatic re-index for '{self.current_model['model_name']}' completed.")
                            self._load_tensor_internal(model_name=self.current_model['model_name']) # Reload after fix
                        except IcotqError as fix_e:
                            self.log.error(f"Automatic fix failed for model '{self.current_model['model_name']}': {fix_e}")
                            self.embeddings_matrix = None # Ensure matrix is None if fix failed
                            # Raise a critical error to signal that the store might be unusable for this model
                            raise IcotqCriticalError(f"Failed initial load and automatic fix for model '{self.current_model['model_name']}'. Manual intervention likely required ('index purge').") from fix_e
                    else:
                        self.log.warning("Automatic fixing is disabled. Manual 'index purge' may be required.")
                        # Set matrix to None to prevent operations on inconsistent data
                        self.embeddings_matrix = None
                except IcotqError as e:
                     self.log.error(f"Error loading tensor for initial model: {e}")
                     self.embeddings_matrix = None

        self.log.info("IcoTqStore initialized.")
        # Perform a quick check on startup (dry run)
        try:
            self.check_clean(dry_run=True)
        except IcotqError as e:
            self.log.warning(f"Initial consistency check found issues: {e}")

    # === Configuration and Initialization Methods ===

    def _load_or_init_config(self):
        """Loads configuration from file or creates a default one."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    iqc: IcotqConfig = json.load(f)
                    # Ensure new keys have defaults if loading old config
                    iqc.setdefault('auto_fix_inconsistency', False)
                    self.config = iqc
                self.log.info(f"Loaded configuration from {self.config_file}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.log.error(f"Failed to load or parse config file {self.config_file}: {e}. Using default config.")
                self._create_default_config()
                # Attempt to save the valid default config
                try:
                    self._save_config_internal()
                except IcotqError as save_e:
                     self.log.error(f"Failed to save default config: {save_e}") # Non-fatal
        else:
            self._create_default_config()
            self.log.warning(f"Created default configuration at {self.config_file}, please review!")
            try:
                self._save_config_internal() # Use internal save as lock might not be held yet
            except IcotqError as save_e:
                self.log.error(f"Failed to save initial default config: {save_e}") # Non-fatal

    def _create_default_config(self):
        """Sets the default configuration."""
        self.config = IcotqConfig({
            'icotq_path': '~/IcoTqStore',
            'tq_sources': [
                TqSource({
                    'name': 'Calibre', 'tqtype': 'calibre_library',
                    'path': '~/ReferenceLibrary/Calibre Library', 'file_types': ['txt', 'pdf']
                }),
                TqSource({
                    'name': 'Notes', 'tqtype': 'folder',
                    'path': '~/Notes', 'file_types': ['md']
                })],
            'embeddings_model_name': 'ibm-granite/granite-embedding-107m-multilingual', # Updated default
            'embeddings_device': 'auto',
            'embeddings_model_trust_code': True,
            'auto_fix_inconsistency': False # Default to False for safety
        })

    def _validate_config_paths(self):
        """Validates paths and sources in the configuration."""
        self.root_path = os.path.expanduser(self.config['icotq_path'])
        if not self.root_path:
             raise IcotqConfigurationError("`icotq_path` cannot be empty in configuration.")

        valid_sources = []
        known_types: list[str] = ['txt', 'md', 'pdf']
        known_tqtypes = ['calibre_library', 'folder']

        for source in self.config['tq_sources']:
            valid = True
            source_path_expanded = os.path.expanduser(source.get('path', ''))
            if not source.get('name'):
                self.log.error(f"Source missing 'name': {source}. Ignoring.")
                valid = False
            if not source.get('tqtype') or source['tqtype'] not in known_tqtypes:
                self.log.error(f"Source '{source.get('name')}' has invalid or missing tqtype '{source.get('tqtype')}'. Valid types: {known_tqtypes}. Ignoring source.")
                valid = False
            if not source_path_expanded:
                 self.log.error(f"Source '{source.get('name')}' has empty or missing 'path'. Ignoring source.")
                 valid = False
            elif not os.path.exists(source_path_expanded):
                self.log.error(f"Source '{source.get('name')}' path does not exist: '{source_path_expanded}'. Ignoring source.")
                valid = False
            if not source.get('file_types') or not isinstance(source['file_types'], list):
                self.log.error(f"Source '{source.get('name')}' has invalid or missing 'file_types'. Must be a list. Ignoring source.")
                valid = False
            else:
                for tp in source['file_types']:
                    if tp not in known_types:
                        self.log.error(f"Source '{source.get('name')}' has invalid file type '{tp}'. Allowed types: {known_types}. Ignoring source.")
                        valid = False
                        break # No need to check other types for this source

            if valid:
                # Ensure path is stored expanded
                source['path'] = source_path_expanded
                valid_sources.append(source)
            else:
                self.log.warning(f"Please fix configuration file: {self.config_file}")

        if len(valid_sources) < len(self.config['tq_sources']):
            self.config['tq_sources'] = valid_sources
            # Save the cleaned config (best effort)
            try:
                self._save_config_internal()
            except IcotqError as e:
                self.log.error(f"Failed to save config after source validation: {e}")

    def _load_or_init_model_list(self):
        """Loads the list of known embedding models or creates a default."""
        model_list_path = os.path.join(self.root_path, "model_list.json")
        if os.path.exists(model_list_path):
            try:
                with open(model_list_path, 'r') as f:
                    self.model_list = json.load(f)
                self.log.info(f"Loaded model list from {model_list_path}")
            except (json.JSONDecodeError, TypeError) as e:
                self.log.error(f"Failed to load model list {model_list_path}: {e}. Using default.")
                self._create_default_model_list()
                try:
                    self._atomic_save_json(self.model_list, model_list_path)
                except IcotqError as save_e:
                    self.log.error(f"Failed to save default model list: {save_e}")
        else:
            self._create_default_model_list()
            self.log.warning(f"Initialized {model_list_path} with default embeddings model list. Please verify.")
            try:
                self._atomic_save_json(self.model_list, model_list_path)
            except IcotqError as save_e:
                self.log.error(f"Failed to save initial default model list: {save_e}")

    def _create_default_model_list(self):
        """Sets the default list of embedding models."""
        self.model_list = [
             # Add models here as needed
            {
                'model_hf_name': 'ibm-granite/granite-embedding-107m-multilingual',
                'model_name': 'granite-embedding-107m-multilingual',
                'emb_dim': 384, 'max_input_token': 512,
                'chunk_size': 2048, 'chunk_overlap': 2048 // 3
            },
            {
                'model_hf_name': 'nomic-ai/nomic-embed-text-v1.5', # Example alternative
                'model_name': 'nomic-embed-text-v1.5',
                'emb_dim': 768, 'max_input_token': 2048, # Note different token limit
                'chunk_size': 4096, 'chunk_overlap': 4096 // 3
            },
             # Add more models based on leaderboards/requirements
        ]

    def _ensure_storage_dirs(self):
        """Creates necessary storage directories if they don't exist."""
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
            self.log.warning(f"Creating IcoTq storage path at {self.root_path}.")

        config_subdirs = ['Embeddings', 'PDFTextCache']
        for cdir in config_subdirs:
            full_path = os.path.join(self.root_path, cdir)
            if not os.path.isdir(full_path):
                try:
                    os.makedirs(full_path)
                except OSError as e:
                    raise IcotqConfigurationError(f"Failed to create required directory {full_path}: {e}") from e

        self.embeddings_path = os.path.join(self.root_path, "Embeddings")
        self.pdf_cache_path = os.path.join(self.root_path, "PDFTextCache") # Store path


    # === Atomic Save Helpers ===

    def _atomic_save_json(self, data: Any, final_path: str):
        """Atomically saves data as JSON using a temporary file and rename."""
        temp_fd, temp_path = None, None
        try:
            # Create temp file in the same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix=os.path.basename(final_path) + '.tmp')
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f, indent=2) # Keep indent consistent
            os.replace(temp_path, final_path) # Atomic rename/replace
            # self.log.debug(f"Atomically saved JSON to {final_path}")
        except (IOError, OSError, json.JSONDecodeError, TypeError) as e:
            # Clean up temp file if rename failed
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as rm_e:
                    self.log.error(f"Failed to remove temporary file {temp_path} after save error: {rm_e}")
            raise IcotqCriticalError(f"Failed to atomically save JSON to {final_path}: {e}\n{traceback.format_exc()}") from e
        finally:
            # Ensure fd is closed if os.fdopen failed before 'with'
            if temp_fd is not None and not os.fdopen(temp_fd, 'w').closed:
                 try:
                     os.close(temp_fd)
                 except OSError:
                     pass # Ignore close error if already closed by 'with' or failed

    def _atomic_save_tensor(self, tensor_data: torch.Tensor | None, final_path: str):
        """Atomically saves a PyTorch tensor using a temporary file and rename."""
        if tensor_data is None:
            # If None, ensure the final file is removed if it exists
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                    self.log.info(f"Removed obsolete tensor file {final_path}")
                except OSError as e:
                     raise IcotqCriticalError(f"Failed to remove obsolete tensor file {final_path}: {e}") from e
            return # Nothing to save

        temp_fd, temp_path = None, None
        try:
            temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix=os.path.basename(final_path) + '.tmp')
            # Close the file descriptor returned by mkstemp before torch.save uses the path
            os.close(temp_fd)
            temp_fd = None # Indicate fd is closed
            torch.save(tensor_data, temp_path)
            os.replace(temp_path, final_path)
            # self.log.debug(f"Atomically saved tensor to {final_path}")
        except (IOError, OSError, RuntimeError) as e: # Catch torch save errors too
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as rm_e:
                    self.log.error(f"Failed to remove temporary tensor file {temp_path} after save error: {rm_e}")
            raise IcotqCriticalError(f"Failed to atomically save tensor to {final_path}: {e}\n{traceback.format_exc()}") from e
        finally:
             if temp_fd is not None: # If os.close failed or wasn't reached
                 try:
                     os.close(temp_fd)
                 except OSError:
                     pass


    # === Core State Management (Internal, Assumes Lock Held) ===

    def _get_library_path(self) -> str:
        return os.path.join(self.root_path, "icotq_library.json")

    def _get_pdf_index_path(self) -> str:
        return os.path.join(self.pdf_cache_path, "pdf_index.json")

    def _get_tensor_path(self, model_name: str) -> str:
        """Gets the expected path for a model's embeddings tensor."""
        return os.path.join(self.embeddings_path, f"embeddings_{model_name}.pt")

    def _read_library_internal(self):
        """Loads library and PDF index from disk. Assumes lock is held."""
        print("\rLoading library...", end="", flush=True)
        lib_path = self._get_library_path()
        if os.path.exists(lib_path):
            try:
                with open(lib_path, 'r') as f:
                    self.lib = json.load(f)
                print("\r", end="", flush=True)
                self.log.info(f"Library loaded, {len(self.lib)} entries")
            except (json.JSONDecodeError, TypeError) as e:
                 print("\r", end="", flush=True)
                 self.log.error(f"Failed to load library file {lib_path}: {e}. Initializing empty library.")
                 self.lib = []
                 # Consider raising a warning or error depending on desired robustness
        else:
            print("\r", end="", flush=True)
            self.log.info(f"No library state file found at {lib_path}. Initializing empty library.")
            self.lib = []

        pdf_cache_index = self._get_pdf_index_path()
        print("\rLoading PDF cache index...", end="", flush=True)
        if os.path.exists(pdf_cache_index):
            try:
                with open(pdf_cache_index, 'r') as f:
                    self.pdf_index = json.load(f)
                print("\r", end="", flush=True)
                self.log.info(f"PDF text cache index loaded, {len(self.pdf_index.keys())} entries")
            except (json.JSONDecodeError, TypeError) as e:
                 print("\r", end="", flush=True)
                 self.log.error(f"Failed to load PDF index file {pdf_cache_index}: {e}. Initializing empty index.")
                 self.pdf_index = {}
        else:
            self.pdf_index = {}
            print("\r", end="", flush=True)
            self.log.info(f"No PDF index file found at {pdf_cache_index}. Initializing empty index.")


    def _write_library_internal(self):
        """Saves library and PDF index atomically. Assumes lock is held."""
        lib_path = self._get_library_path()
        pdf_index_path = self._get_pdf_index_path()
        try:
            self._atomic_save_json(self.lib, lib_path)
            self._atomic_save_json(self.pdf_index, pdf_index_path)
            # self.log.debug("Library and PDF index saved atomically.")
        except IcotqCriticalError as e:
             # Error already logged by atomic_save
             # Re-raise to signal failure up the call chain
             raise IcotqCriticalError(f"Failed to save library or PDF index state: {e}") from e

    def _save_tensor_internal(self, model_name: str | None = None) -> bool:
        """
        Saves the in-memory embeddings_matrix for the specified model (or current model) atomically.
        Assumes lock is held.
        Returns True on success, raises IcotqCriticalError on failure.
        """
        model_to_save: EmbeddingsModel | None = None
        tensor_to_save: torch.Tensor | None = None

        if model_name:
            if self.current_model and self.current_model['model_name'] == model_name:
                model_to_save = self.current_model
                tensor_to_save = self.embeddings_matrix
            else:
                # This case should ideally not happen if only saving the current matrix
                # If saving arbitrary tensors is needed, loading would be required first.
                 self.log.warning(f"_save_tensor_internal called for non-current model '{model_name}' without providing tensor data. Cannot save.")
                 # Or raise error? Let's return False for now as it was the old behavior signature somewhat
                 # Better: Raise an error as this indicates incorrect usage.
                 raise IcotqError(f"Cannot save tensor for non-current model '{model_name}' without explicit tensor data.")

        elif self.current_model:
            model_to_save = self.current_model
            tensor_to_save = self.embeddings_matrix
        else:
            self.log.error("Can't save embeddings tensor: no current model loaded!")
            # Raise error instead of returning False for clarity
            raise IcotqError("Cannot save tensor: No model is currently loaded.")

        if model_to_save:
            embeddings_tensor_file = self._get_tensor_path(model_to_save['model_name'])
            try:
                self._atomic_save_tensor(tensor_to_save, embeddings_tensor_file)
                self.log.info(f"Embeddings tensor for '{model_to_save['model_name']}' saved to {embeddings_tensor_file}")
                return True
            except IcotqCriticalError as e:
                 self.log.error(f"Failed to save embeddings tensor for {model_to_save['model_name']} to {embeddings_tensor_file}: {e}")
                 raise # Re-raise the critical error
        else:
             # This case should be unreachable if logic above is correct
             raise IcotqError("Cannot save tensor: Model context lost unexpectedly.")


    def _load_tensor_internal(self, model_name: str, device_override: str | None = None, check_consistency: bool = True) -> bool:
        """
        Loads tensor for a given model name. Assumes lock is held.
        Handles consistency checks and potential automatic fixing.
        Returns True if loaded (even if inconsistent but auto-fix off), raises errors otherwise.
        """
        embeddings_tensor_file = self._get_tensor_path(model_name)
        target_device = self.resolve_device(device_override if device_override else self.config['embeddings_device'])
        map_location = torch.device(target_device)
        loaded_tensor: torch.Tensor | None = None

        if os.path.exists(embeddings_tensor_file):
            try:
                # Load tensor
                loaded_tensor = torch.load(embeddings_tensor_file, map_location=map_location)
                self.log.info(f"Loaded tensor for model '{model_name}' onto device '{target_device}'. Shape: {loaded_tensor.shape if loaded_tensor is not None else 'N/A'}")

                # --- Consistency Check ---
                if check_consistency and loaded_tensor is not None:
                    expected_rows = 0
                    for entry in self.lib:
                        if model_name in entry.get('emb_ptrs', {}):
                             expected_rows += entry['emb_ptrs'][model_name][1]

                    actual_rows = loaded_tensor.shape[0]
                    self.log.info(f"Consistency check for '{model_name}': Tensor rows={actual_rows}, Library expected rows={expected_rows}")

                    if actual_rows != expected_rows:
                        consistency_msg = f"Embeddings tensor '{model_name}' is INCONSISTENT with library! Tensor has {actual_rows} rows, library expects {expected_rows}."
                        # Raise IcotqConsistencyError instead of just warning
                        raise IcotqConsistencyError(consistency_msg)

            except FileNotFoundError:
                 # Should be caught by os.path.exists, but handle defensively
                 self.log.warning(f"Tensor file {embeddings_tensor_file} vanished before loading.")
                 loaded_tensor = None
            except (RuntimeError, EOFError, Exception) as e: # Catch torch load errors and others
                 self.log.error(f"Failed to load or process tensor file {embeddings_tensor_file}: {e}", exc_info=True)
                 # Raise a critical error as the file might be corrupt
                 raise IcotqCriticalError(f"Corrupted or unreadable tensor file: {embeddings_tensor_file}") from e
        else:
            self.log.warning(f"No embeddings tensor file found for model '{model_name}' at {embeddings_tensor_file}. Use 'index' to generate.")
            loaded_tensor = None

        # Update the in-memory matrix *only if* this is the current model
        if self.current_model and self.current_model['model_name'] == model_name:
            self.embeddings_matrix = loaded_tensor
            # Update the device info if it changed
            if self.device != target_device:
                self.log.info(f"Updating active device context to '{target_device}'")
                self.device = target_device
                if self.engine:
                    self.engine = self.engine.to(map_location) # Move model if device changed

        return loaded_tensor is not None # Return true if *something* was loaded, even if inconsistent (error raised separately)


    def _save_config_internal(self):
        """Saves config atomically. Assumes lock is held."""
        try:
            self._atomic_save_json(self.config, self.config_file)
            self.log.info(f"Configuration changes saved to {self.config_file}")
        except IcotqCriticalError as e:
             # Error already logged by atomic_save
             raise IcotqCriticalError(f"Failed to save configuration: {e}") from e


    # === PDF Handling (Internal, Assumes Lock Held) ===

    def _get_pdf_text_internal(self, desc:str, full_path:str) -> tuple[str | None, bool]:
        """Gets PDF text, using cache if possible. Assumes lock is held."""
        text: str | None = None
        changed: bool = False # Indicates if pdf_index was modified

        # Check cache validity
        if desc in self.pdf_index:
            cached_info = self.pdf_index[desc]
            try:
                cur_file_size = os.path.getsize(full_path)
                # If size matches AND not a known previous failure AND cache file exists
                if (cur_file_size == cached_info['file_size'] and
                        not cached_info['previous_failure'] and
                        cached_info.get('filename')): # Check filename exists

                    basename = os.path.basename(cached_info['filename'])
                    local_path = os.path.join(self.pdf_cache_path, basename)

                    if os.path.exists(local_path): # Double-check cache file exists
                        try:
                            with open(local_path, 'r', encoding='utf-8') as f: # Specify encoding
                                text = f.read()
                            # self.log.debug(f"Read PDF cache for {desc}")
                            return text, False # Return cached text, index not changed
                        except Exception as e:
                            self.log.warning(f"Failed to read PDF cache file {local_path} for {desc}: {e}. Re-extracting.")
                            # Cache file corrupted/unreadable, proceed to re-extract
                            text = None
                            # No need to delete from index yet, will be overwritten
                    else:
                         self.log.warning(f"PDF cache index points to non-existent file {local_path} for {desc}. Re-extracting.")
                         text = None
                         # No need to delete from index yet, will be overwritten

                elif cur_file_size != cached_info['file_size']:
                    self.log.info(f"PDF file size changed for {desc}, re-importing text.")
                    text = None # Force re-extraction
                elif cached_info['previous_failure']:
                    # Known failure, don't retry unless file changed (handled above)
                    # self.log.debug(f"Skipping PDF {desc} due to previous extraction failure.")
                    return None, False # Return None, index not changed
                else:
                    # Size matches, but previous_failure is false and filename is missing/empty? -> Inconsistent state
                    self.log.warning(f"Inconsistent PDF cache state for {desc} (size matches, not failed, but no cache file?). Re-extracting.")
                    text = None # Force re-extraction

            except FileNotFoundError:
                self.log.warning(f"Original PDF file {full_path} not found while checking cache for {desc}. Cannot get text.")
                # Remove the potentially invalid index entry? Or leave it? Let's remove it.
                if desc in self.pdf_index:
                    del self.pdf_index[desc]
                    changed = True
                return None, changed
            except Exception as e:
                self.log.error(f"Error accessing file {full_path} or its cache info for {desc}: {e}. Cannot get text.")
                return None, False # Don't change index on access error

        # If text is still None, attempt extraction
        if text is None:
            extracted_text: str | None = None
            failure = True
            cache_filename = ""
            try:
                doc = pymupdf.open(full_path)
                extracted_pages = []
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text() # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                        if isinstance(page_text, str):
                            extracted_pages.append(page_text)
                        else:
                            self.log.warning(f"Non-string text found on page {page_num+1} of {full_path}. Skipping page.")
                    except Exception as page_e:
                        self.log.warning(f"Failed to extract text from page {page_num+1} of {full_path}: {page_e}")
                doc.close()

                if extracted_pages:
                    extracted_text = "\n".join(extracted_pages) # Join pages with newline
                    if extracted_text.strip(): # Check if not just whitespace
                        failure = False
                        cache_filename = str(uuid.uuid4()) + ".txt" # Add extension
                        self.log.info(f"Successfully extracted text from: {desc}")
                    else:
                        self.log.info(f"Extracted only whitespace from: {desc}. Treating as failure.")
                        extracted_text = None # Treat as failure
                else:
                    self.log.info(f"No text could be extracted from: {desc}")
                    extracted_text = None

            except FileNotFoundError:
                 self.log.error(f"PDF file {full_path} not found during extraction for {desc}.")
                 # If we reached here, the index entry might have been deleted already or didn't exist
                 # Ensure it's gone if present
                 if desc in self.pdf_index:
                     del self.pdf_index[desc]
                     changed = True
                 return None, changed
            except Exception as e:
                self.log.error(f"Failed to open or process PDF {full_path} for {desc}: {e}", exc_info=True)
                extracted_text = None # Ensure failure state

            # --- Update Cache Index and File ---
            new_pdf_ind: PDFIndex = {
                'filename': cache_filename,
                'file_size': os.path.getsize(full_path) if os.path.exists(full_path) else -1, # Store size even on failure if possible
                'previous_failure': failure
            }

            # Write new cache file if extraction succeeded
            if not failure and extracted_text is not None:
                cache_file_path = os.path.join(self.pdf_cache_path, cache_filename)
                try:
                    # Use atomic save for the text cache file itself
                    temp_fd, temp_path = tempfile.mkstemp(dir=self.pdf_cache_path, prefix=cache_filename + '.tmp')
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    os.replace(temp_path, cache_file_path)
                    self.log.info(f"Added/Updated {desc} in PDF cache ({cache_filename}), size: {len(self.pdf_index.keys())}")
                    text = extracted_text # Set return text
                except (IOError, OSError) as e:
                     self.log.error(f"Failed to write PDF cache file {cache_file_path} for {desc}: {e}. Extraction result lost.")
                     # Revert state to failure
                     new_pdf_ind['previous_failure'] = True
                     new_pdf_ind['filename'] = ""
                     text = None
                     # Clean up temp file
                     if temp_path and os.path.exists(temp_path):
                         try: os.remove(temp_path)
                         except OSError: pass
                except Exception as e: # Catch any other unexpected errors during write
                    self.log.error(f"Unexpected error writing PDF cache file {cache_file_path} for {desc}: {e}", exc_info=True)
                    new_pdf_ind['previous_failure'] = True
                    new_pdf_ind['filename'] = ""
                    text = None
                    if temp_path and os.path.exists(temp_path):
                         try: os.remove(temp_path)
                         except OSError: pass


            # Remove old cache file if it exists and filename changed or extraction failed
            old_filename = self.pdf_index.get(desc, {}).get('filename')
            if old_filename and old_filename != cache_filename:
                old_cache_path = os.path.join(self.pdf_cache_path, os.path.basename(old_filename))
                if os.path.exists(old_cache_path):
                    try:
                        os.remove(old_cache_path)
                        self.log.debug(f"Removed old PDF cache file {old_cache_path}")
                    except OSError as e:
                        self.log.warning(f"Failed to remove old PDF cache file {old_cache_path}: {e}")

            # Update the index entry
            self.pdf_index[desc] = new_pdf_ind
            changed = True # Index was modified

        return text, changed


    # === Public Interface Methods (Acquire Lock) ===

    @contextmanager
    def _modify(self) -> Generator[None, None, None]:
        """Context manager to acquire the lock for modification."""
        with self._lock:
            yield

    def save_config(self):
        """Public method to save configuration."""
        with self._modify():
            self._save_config_internal()

    def list_sources(self) -> None:
        """Prints the list of configured sources."""
        # Read-only access to config, lock might be optional but safer
        with self._lock:
            if not self.config['tq_sources']:
                print("No valid sources configured.")
                return
            print("Configured Sources:")
            for i, source in enumerate(self.config['tq_sources']):
                print(f"  {i:02d}: Name='{source['name']}', Type='{source['tqtype']}', Path='{source['path']}', Files={source['file_types']}")

    def read_library(self):
        """Public method to reload library and PDF index from disk."""
        with self._modify():
            self._read_library_internal()
            # Optionally reload tensor for current model if needed after library reload
            if self.current_model:
                try:
                    self._load_tensor_internal(self.current_model['model_name'])
                except IcotqError as e: # Catch consistency or critical errors
                    self.log.error(f"Error reloading tensor for current model after library reload: {e}")
                    # Decide if self.embeddings_matrix should be cleared
                    self.embeddings_matrix = None


    def load_model(self, name: str, device:str="auto", trust_remote_code:bool=False) -> bool:
        """Loads an embeddings model and its corresponding tensor."""
        with self._modify():
            try:
                # Load model definition
                loaded_model_def = self._load_model_internal(name, device, trust_remote_code)
                if loaded_model_def:
                     # Load associated tensor, handle potential inconsistencies
                     try:
                         self._load_tensor_internal(loaded_model_def['model_name'])
                     except IcotqConsistencyError as e:
                         self.log.error(f"Consistency Error loading tensor for new model '{name}': {e}")
                         if self.config.get('auto_fix_inconsistency', False):
                             self.log.warning(f"Attempting automatic fix (re-index) for model '{name}'.")
                             try:
                                 self._generate_embeddings_internal(purge=True, model_name_override=name)
                                 self.log.info(f"Automatic re-index for '{name}' completed.")
                                 self._load_tensor_internal(name) # Reload after fix
                             except IcotqError as fix_e:
                                 self.log.error(f"Automatic fix failed for model '{name}': {fix_e}")
                                 self.embeddings_matrix = None # Ensure matrix is None if fix failed
                                 # Don't necessarily fail the whole model load, but warn heavily
                                 self.log.critical(f"Model '{name}' loaded, but embeddings are inconsistent and could not be fixed automatically. Manual 'index purge' required.")

                         else:
                             self.log.warning("Automatic fixing is disabled. Embeddings for this model are inconsistent. Manual 'index purge' may be required.")
                             self.embeddings_matrix = None # Prevent usage

                     except IcotqCriticalError as e:
                          # Corrupt tensor file etc. - treat as major failure for this model
                          self.log.error(f"Critical Error loading tensor for model '{name}': {e}")
                          self.embeddings_matrix = None
                          # Consider if current_model should be unset or if loading just the definition is ok
                          # Let's keep current_model set, but matrix is None
                          self.log.critical(f"Model '{name}' loaded, but its embeddings tensor is corrupt or unreadable. Indexing needed.")
                     except IcotqError as e:
                          # Other errors during tensor load
                          self.log.error(f"Error loading tensor for model '{name}': {e}")
                          self.embeddings_matrix = None

                     # Update config only if model load itself succeeded
                     self.config['embeddings_model_name'] = name
                     self.config['embeddings_device'] = device # Store requested device (auto resolved internally)
                     self.config['embeddings_model_trust_code'] = trust_remote_code
                     self._save_config_internal() # Save changes
                     return True
                else:
                    # _load_model_internal failed, error already logged
                    return False

            except IcotqError as e:
                self.log.error(f"Failed to load model '{name}': {e}")
                # Ensure state is clean if model load fails
                self.engine = None
                self.current_model = None
                self.embeddings_matrix = None
                self.device = None
                return False
            except Exception as e:
                 # Catch unexpected errors during model loading
                 self.log.critical(f"Unexpected critical error loading model '{name}': {e}", exc_info=True)
                 self.engine = None
                 self.current_model = None
                 self.embeddings_matrix = None
                 self.device = None
                 # Raise a critical error to signal something went very wrong
                 raise IcotqCriticalError(f"Unexpected failure loading model {name}") from e


    def _load_model_internal(self, name: str, device_str:str="auto", trust_remote_code:bool=False) -> EmbeddingsModel | None:
        """Internal model loading logic. Assumes lock is held."""
        self.log.info(f"Attempting to load model '{name}'...")
        selected_model: EmbeddingsModel | None = None
        for model in self.model_list:
            if model.get('model_hf_name') == name or model.get('model_name') == name:
                selected_model = model
                break

        if not selected_model:
            raise IcotqConfigurationError(f"Model '{name}' is unknown, not found in model_list.json")

        hf_name = selected_model['model_hf_name']
        try:
            # Release lock temporarily ONLY for the potentially long model download/load
            # This is a trade-off: allows other operations but model state is briefly undefined
            # Re-acquiring the lock ensures state updates are safe.
            # Alternative: Keep lock, blocking everything. Let's keep lock for simplicity/safety first.

            # --- Load Model ---
            engine = SentenceTransformer(hf_name, trust_remote_code=trust_remote_code)
            resolved_device = self.resolve_device(device_str)
            engine = engine.to(torch.device(resolved_device))

            # --- Update State ---
            self.engine = engine
            self.device = resolved_device
            self.current_model = selected_model
            # Clear the matrix, it will be loaded by the caller (_load_tensor_internal)
            self.embeddings_matrix = None

            self.log.info(f"Model '{name}' ({hf_name}) loaded successfully onto device '{resolved_device}'.")
            return selected_model

        except Exception as e:
            # Catch errors from SentenceTransformer or .to()
            self.log.error(f"Failed to load or initialize model '{name}' ({hf_name}): {e}", exc_info=True)
            # Raise specific error type
            raise IcotqError(f"SentenceTransformer failed for model '{name}'") from e


    def sync_texts(self, max_imports: int | None = None):
        """
        Synchronizes the library with source files, handling additions, updates, and deletions.
        Performs cleanup atomically.
        """
        self.log.info("Starting text synchronization...")
        with self._modify():
            if not self.config['tq_sources']:
                self.log.error("No valid sources defined in config, cannot sync.")
                return

            initial_lib_size = len(self.lib)
            lib_changed = False
            pdf_index_changed = False
            processed_desc_paths = set() # Track files found on disk

            # --- Pass 1: Scan sources and update/add to in-memory library ---
            self.log.debug("Sync Pass 1: Scanning sources...")
            lib_map = {entry['desc_filename']: entry for entry in self.lib}
            new_lib: list[LibEntry] = [] # Build a new library list

            abort_scan = False
            current_imports = 0

            for source in self.config['tq_sources']:
                if abort_scan: break
                source_path = source['path'] # Already expanded in validation
                self.log.info(f"Scanning source '{source['name']}' at '{source_path}'...")

                for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: self.log.warning(f"Cannot access directory {e.filename}: {e.strerror}")):
                    if abort_scan: break
                    # Filter dirs? Maybe later if needed for performance or exclusion rules

                    for filename in files:
                        if abort_scan: break

                        # Check max imports limit
                        current_imports +=1 # Count scanned eligible files
                        # if max_imports is not None and current_imports > max_imports:
                        #     self.log.warning(f"Reached maximum import scan limit ({max_imports}). Stopping scan.")
                        #     abort_scan = True
                        #     break # Stop processing files in this directory

                        base, ext_with_dot = os.path.splitext(filename)
                        ext = ext_with_dot[1:].lower() if ext_with_dot else ""

                        if ext not in source['file_types']:
                            continue

                        # Skip alternative formats if a preferred one exists (e.g., skip PDF if TXT exists)
                        # Note: This logic might need refinement based on desired precedence.
                        # Example: Prefer TXT > MD > PDF
                        preferred_ext_exists = False
                        preferred_order = ['txt', 'md'] # PDF is implicitly last
                        if ext == 'pdf': # Only check for preferred alternatives if current is PDF
                             for pref_ext in preferred_order:
                                 alt_file_path = os.path.join(root, base + '.' + pref_ext)
                                 if os.path.exists(alt_file_path):
                                     # self.log.debug(f"Skipping '{filename}' because preferred format '.{pref_ext}' exists.")
                                     preferred_ext_exists = True
                                     break
                        if preferred_ext_exists:
                            continue

                        full_path = os.path.join(root, filename)
                        # Create unique descriptor path relative to source root
                        relative_path = os.path.relpath(full_path, source_path)
                        desc_path = "{" + source['name'] + "}" + relative_path

                        processed_desc_paths.add(desc_path)

                        # Get current text
                        current_text: str | None = None
                        pdf_changed_during_get = False
                        try:
                            if ext in ['md', 'txt']: # Add other plain text types if needed
                                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    current_text = f.read()
                            elif ext == 'pdf':
                                current_text, pdf_changed_during_get = self._get_pdf_text_internal(desc_path, full_path)
                                if pdf_changed_during_get:
                                    pdf_index_changed = True
                            # Add elif for other types if needed
                        except FileNotFoundError:
                            self.log.warning(f"File vanished during sync: {full_path}. Skipping.")
                            continue
                        except Exception as e:
                             self.log.error(f"Error reading file {full_path} for sync: {e}. Skipping.")
                             continue

                        # Check against existing library entry
                        existing_entry = lib_map.get(desc_path)

                        if existing_entry:
                            # File exists in library, check for changes
                            if current_text is not None and existing_entry.get('text') != current_text:
                                self.log.info(f"Updating text for {desc_path}")
                                existing_entry['text'] = current_text
                                # Mark for re-embedding by clearing pointers
                                if existing_entry.get('emb_ptrs'):
                                    # Add old pointers to debris before clearing
                                    # This happens in Pass 2 now
                                    pass
                                existing_entry['emb_ptrs'] = {} # Reset pointers
                                lib_changed = True
                            elif current_text is None and existing_entry.get('text') is not None:
                                # Text became unreadable, treat as update (clear text and pointers)
                                self.log.warning(f"Text for {desc_path} became unreadable. Clearing entry text.")
                                existing_entry['text'] = "" # Or None? Empty string is safer for JSON/typing
                                if existing_entry.get('emb_ptrs'):
                                     pass # Debris collected in Pass 2
                                existing_entry['emb_ptrs'] = {}
                                lib_changed = True
                            # Update filename in case path casing changed etc.
                            if existing_entry['filename'] != full_path:
                                 existing_entry['filename'] = full_path
                                 lib_changed = True
                            # Keep existing entry (potentially updated)
                            new_lib.append(existing_entry)

                        elif current_text is not None:
                            # New file found with readable text
                            self.log.info(f"Adding new entry for {desc_path}")
                            entry: LibEntry = LibEntry({
                                'source_name': source['name'],
                                'desc_filename': desc_path,
                                'filename': full_path,
                                'text': current_text,
                                'emb_ptrs': {} # New entries have no embeddings yet
                            })
                            new_lib.append(entry)
                            lib_changed = True
                        else:
                            # New file found but text is not readable (e.g., PDF failed extraction)
                            # Don't add it to the library.
                            self.log.debug(f"Skipping add for new file {desc_path} as text is not available.")

            # Apply max_imports limit *after* scanning all sources if specified
            # This ensures we don't arbitrarily cut off sources
            if max_imports is not None and len(new_lib) > max_imports:
                 self.log.warning(f"Library size ({len(new_lib)}) exceeds max_imports ({max_imports}). Pruning...")
                 # Keep the first 'max_imports' entries (implicitly assumes some stable order, maybe sort first?)
                 # For now, simple prune based on scan order.
                 entries_to_prune = new_lib[max_imports:]
                 new_lib = new_lib[:max_imports]
                 # Add pruned entries to processed_desc_paths so they are considered for debris
                 for entry in entries_to_prune:
                     processed_desc_paths.discard(entry['desc_filename']) # Remove from found set
                 lib_changed = True


            # --- Pass 2: Identify Debris and Collect Old Pointers ---
            self.log.debug("Sync Pass 2: Identifying debris and collecting old pointers...")
            debris_lib_entries: list[LibEntry] = []
            debris_pdf_indices: list[str] = []
            # Pointers from updated entries (cleared in Pass 1) or deleted entries
            tensor_debris: dict[str, list[tuple[int, int]]] = {} # model_name -> list[(start, len)]

            original_lib_descs = set(lib_map.keys())
            current_lib_descs = {entry['desc_filename'] for entry in new_lib}

            # Find entries present in original lib but NOT in the new lib (deleted or pruned)
            deleted_or_pruned_descs = original_lib_descs - current_lib_descs
            for desc_path in deleted_or_pruned_descs:
                entry = lib_map[desc_path]
                self.log.info(f"Detected deleted/pruned library entry: {desc_path}")
                debris_lib_entries.append(entry)
                # Collect its embedding pointers for cleanup
                for model_name, ptr_info in entry.get('emb_ptrs', {}).items():
                     if model_name not in tensor_debris: tensor_debris[model_name] = []
                     tensor_debris[model_name].append(ptr_info)
                lib_changed = True # Library structure changed

            # Find entries updated in Pass 1 (still in new_lib, but pointers were cleared)
            for entry in new_lib:
                 desc_path = entry['desc_filename']
                 original_entry = lib_map.get(desc_path)
                 # If it existed before AND its pointers are now empty AND it used to have pointers
                 if original_entry and not entry.get('emb_ptrs') and original_entry.get('emb_ptrs'):
                      self.log.info(f"Detected updated library entry (needs re-index): {desc_path}")
                      # Collect its old embedding pointers for cleanup
                      for model_name, ptr_info in original_entry['emb_ptrs'].items():
                          if model_name not in tensor_debris: tensor_debris[model_name] = []
                          tensor_debris[model_name].append(ptr_info)
                      # lib_changed is already True if text was updated


            # Find PDF index entries whose corresponding lib entry is gone
            all_current_lib_descs = {entry['desc_filename'] for entry in new_lib}
            for desc_path in list(self.pdf_index.keys()): # Iterate over copy of keys
                 if desc_path not in all_current_lib_descs:
                     self.log.info(f"Detected orphaned PDF cache index entry: {desc_path}")
                     debris_pdf_indices.append(desc_path)
                     pdf_index_changed = True


            # --- Pass 3: Execute Cleanup (Modify Tensors, Update Pointers, Save State) ---
            self.log.debug("Sync Pass 3: Executing cleanup...")
            if tensor_debris or debris_lib_entries or debris_pdf_indices:
                self.log.info("Performing cleanup of embeddings and library state.")

                modified_tensors: dict[str, torch.Tensor] = {} # Store modified tensors in memory first
                new_pointer_maps: dict[str, dict[int, tuple[int, int]]] = {} # model -> {old_start: new_ptr_tuple}
                all_models_processed_ok = True

                # Step 3a: Process Tensor Debris (In Memory)
                for model_name, removals in tensor_debris.items():
                    if not removals: continue
                    self.log.info(f"Calculating cleanup for tensor '{model_name}' ({len(removals)} chunks to remove).")
                    try:
                        # Load the original tensor for this model
                        # Use a temporary load, don't affect self.embeddings_matrix yet
                        # No consistency check needed here, just load raw data
                        temp_tensor_path = self._get_tensor_path(model_name)
                        if not os.path.exists(temp_tensor_path):
                            self.log.warning(f"Tensor file for model '{model_name}' not found. Cannot clean up its pointers. Corresponding library pointers will be removed.")
                            # Mark existing pointers for this model in new_lib for removal
                            for i, entry in enumerate(new_lib):
                                if model_name in entry.get('emb_ptrs', {}):
                                    del new_lib[i]['emb_ptrs'][model_name]
                                    lib_changed = True
                            continue # Skip to next model

                        temp_device = self.resolve_device(self.config['embeddings_device']) # Load to default device
                        original_tensor = torch.load(temp_tensor_path, map_location=torch.device(temp_device))

                        # --- Calculate Row Removals ---
                        # Sort removals by start index in REVERSE order for safe deletion
                        sorted_removals = sorted(removals, key=lambda x: x[0], reverse=True)

                        # Create list of indices to keep
                        num_rows = original_tensor.shape[0]
                        keep_indices = list(range(num_rows))
                        removed_count = 0
                        # This reverse iteration removal simulation is complex, maybe build keep_indices directly?

                        # Alternative: Build a mask or list of indices to keep
                        keep_mask = torch.ones(num_rows, dtype=torch.bool, device=original_tensor.device)
                        offsets = torch.zeros(num_rows, dtype=torch.long, device=original_tensor.device)
                        total_removed_len = 0

                        # Mark rows to remove
                        for start, length in removals:
                             if start < 0 or start + length > num_rows:
                                 self.log.error(f"Invalid removal range [{start}:{start+length}] for tensor '{model_name}' size {num_rows}. Skipping this removal.")
                                 continue
                             if torch.any(keep_mask[start : start + length] == False):
                                  self.log.warning(f"Overlapping removal detected near index {start} for tensor '{model_name}'. Check logic.")
                                  # Continue removal, but be aware of potential issues if overlap is unexpected
                             keep_mask[start : start + length] = False
                             total_removed_len += length

                        # Create the new tensor by indexing with the keep_mask
                        modified_tensor = original_tensor[keep_mask]
                        modified_tensors[model_name] = modified_tensor # Store modified tensor

                        # --- Calculate Pointer Adjustments ---
                        # Calculate cumulative removed length *before* each original index
                        removed_lengths_cumsum = torch.cumsum(~keep_mask, dim=0)

                        current_model_pointers: dict[int, tuple[int,int]] = {} # old_start -> new_ptr
                        for entry in new_lib:
                            if model_name in entry.get('emb_ptrs', {}):
                                old_start, old_length = entry['emb_ptrs'][model_name]

                                if old_start < 0 or old_start >= num_rows:
                                    self.log.error(f"Entry '{entry['desc_filename']}' has invalid old start pointer {old_start} for tensor '{model_name}' size {num_rows}. Removing pointer.")
                                    del entry['emb_ptrs'][model_name]
                                    lib_changed = True
                                    continue

                                # Offset is the number of removed rows *before* the old_start index
                                offset = removed_lengths_cumsum[old_start - 1].item() if old_start > 0 else 0
                                new_start = old_start - offset

                                if new_start < 0:
                                     # This indicates a major logic error
                                     self.log.critical(f"CRITICAL ERROR during pointer adjustment for '{entry['desc_filename']}' model '{model_name}': New start index became negative ({new_start}). Old: {old_start}, Offset: {offset}.")
                                     # Raise a critical exception - state is potentially corrupt
                                     raise IcotqCriticalError(f"Pointer adjustment calculation failed critically for model {model_name}. State inconsistent.")

                                # Check if the chunk itself was removed (shouldn't happen if logic is right)
                                if not torch.all(keep_mask[old_start : old_start + old_length]):
                                     self.log.error(f"Logic Error: Entry '{entry['desc_filename']}' (model '{model_name}') points to rows [{old_start}:{old_start+old_length}] which were marked for removal. Removing pointer.")
                                     del entry['emb_ptrs'][model_name]
                                     lib_changed = True
                                else:
                                     # Store the mapping: old_start -> (new_start, old_length)
                                     current_model_pointers[old_start] = (new_start, old_length)

                        new_pointer_maps[model_name] = current_model_pointers

                    except IcotqCriticalError:
                         # Propagate critical errors immediately
                         all_models_processed_ok = False
                         raise
                    except Exception as e:
                         # Catch other errors during tensor processing for a model
                         self.log.error(f"Failed to process tensor cleanup for model '{model_name}': {e}", exc_info=True)
                         all_models_processed_ok = False
                         # Decide how to handle: Stop sync, or try other models?
                         # Let's stop sync to be safe.
                         raise IcotqCriticalError(f"Failure during tensor processing for {model_name}. Aborting sync.") from e


                # Step 3b: Commit Changes (If all models processed OK)
                if all_models_processed_ok:
                    # --- Save Modified Tensors ---
                    for model_name, tensor_data in modified_tensors.items():
                        try:
                            tensor_path = self._get_tensor_path(model_name)
                            self._atomic_save_tensor(tensor_data, tensor_path)
                            self.log.info(f"Atomically saved cleaned tensor for model '{model_name}'. New shape: {tensor_data.shape}")
                        except IcotqCriticalError as e:
                            # If saving fails even after calculation, state is bad.
                            self.log.critical(f"CRITICAL: Failed to save cleaned tensor for model '{model_name}' after calculation: {e}")
                            # Raise critical error, as calculated state doesn't match disk
                            raise IcotqCriticalError(f"Failed to commit cleaned tensor for {model_name}. State may be corrupt.") from e

                    # --- Apply Pointer Updates to new_lib ---
                    for model_name, ptr_map in new_pointer_maps.items():
                        for i, entry in enumerate(new_lib):
                             if model_name in entry.get('emb_ptrs', {}):
                                 old_start = entry['emb_ptrs'][model_name][0]
                                 if old_start in ptr_map:
                                     new_ptr_tuple = ptr_map[old_start]
                                     if new_lib[i]['emb_ptrs'][model_name] != new_ptr_tuple:
                                          # self.log.debug(f"Updating pointer for {entry['desc_filename']} model {model_name} from {entry['emb_ptrs'][model_name]} to {new_ptr_tuple}")
                                          new_lib[i]['emb_ptrs'][model_name] = new_ptr_tuple
                                          lib_changed = True
                                 else:
                                     # This entry's pointer wasn't in the map - should have been handled earlier?
                                     # Log error and remove pointer for safety
                                     self.log.error(f"Pointer consistency error: Old start {old_start} for {entry['desc_filename']} model {model_name} not found in adjustment map. Removing pointer.")
                                     del new_lib[i]['emb_ptrs'][model_name]
                                     lib_changed = True

                    # --- Update In-Memory Library ---
                    self.lib = new_lib

                    # --- Clean PDF Index ---
                    for desc_path in debris_pdf_indices:
                         if desc_path in self.pdf_index:
                             pdf_info = self.pdf_index[desc_path]
                             cache_filename = pdf_info.get('filename')
                             if cache_filename:
                                 cache_file_path = os.path.join(self.pdf_cache_path, os.path.basename(cache_filename))
                                 if os.path.exists(cache_file_path):
                                     try:
                                         os.remove(cache_file_path)
                                         self.log.info(f"Removed orphaned PDF cache file: {cache_file_path}")
                                     except OSError as e:
                                         self.log.warning(f"Failed to remove orphaned PDF cache file {cache_file_path}: {e}")
                             del self.pdf_index[desc_path]
                             pdf_index_changed = True # Ensure flag is set

                    # --- Save Final State Atomically ---
                    if lib_changed or pdf_index_changed:
                        self.log.info("Saving updated library and PDF index...")
                        self._write_library_internal() # Saves both lib and pdf_index atomically

                    # --- Reload Current Tensor If It Was Modified ---
                    if self.current_model and self.current_model['model_name'] in modified_tensors:
                         self.log.info(f"Reloading current model's ({self.current_model['model_name']}) tensor after cleanup.")
                         # Reload using internal method, consistency check should pass now
                         try:
                              self._load_tensor_internal(self.current_model['model_name'], check_consistency=True)
                         except (IcotqConsistencyError, IcotqCriticalError) as e:
                               # This should NOT happen if cleanup logic is correct
                               self.log.critical(f"CRITICAL: Consistency error after cleanup reload for model {self.current_model['model_name']}: {e}")
                               self.embeddings_matrix = None # Safety
                               raise IcotqCriticalError("Consistency check failed immediately after cleanup. Logic error suspected.") from e

                    self.log.info(f"Synchronization and cleanup completed. Library size: {len(self.lib)}")

                else:
                     # Should be unreachable if errors are raised correctly
                     self.log.error("Sync aborted due to errors during tensor processing. No changes committed.")

            elif lib_changed or pdf_index_changed:
                # Only library/pdf index changed (adds/simple updates), no tensor cleanup needed
                self.lib = new_lib
                self.log.info("Saving updated library and PDF index (no tensor cleanup required)...")
                self._write_library_internal()
                self.log.info(f"Synchronization completed. Library size: {len(self.lib)}")
            else:
                self.log.info("No changes detected during synchronization.")


    def generate_embeddings(self, save_every_sec: int = 180, purge: bool = False, model_name_override: str | None = None):
        """
        Generates embeddings for missing entries or all entries (if purge=True).
        Uses atomic saves periodically and at the end.
        Can operate on a specific model if `model_name_override` is provided.
        """
        with self._modify():
            self._generate_embeddings_internal(save_every_sec, purge, model_name_override)

    def _generate_embeddings_internal(self, save_every_sec: int = 180, purge: bool = False, model_name_override: str | None = None):
        """Internal implementation of generate_embeddings. Assumes lock is held."""

        model_to_use: EmbeddingsModel | None = None
        local_engine: SentenceTransformer | None = None
        local_embeddings_matrix: torch.Tensor | None = None
        target_model_name: str = ""

        # Determine which model and engine to use
        if model_name_override:
            # Find the model definition
            found_model = None
            for m in self.model_list:
                if m['model_name'] == model_name_override:
                    found_model = m
                    break
            if not found_model:
                 raise IcotqConfigurationError(f"Cannot generate embeddings: Overridden model name '{model_name_override}' not found in model list.")
            model_to_use = found_model
            target_model_name = model_to_use['model_name']
            self.log.info(f"Generating embeddings specifically for model: '{target_model_name}'")

            # Load this model temporarily if it's not the current one
            if not self.current_model or self.current_model['model_name'] != target_model_name:
                 self.log.warning(f"Temporarily loading model '{target_model_name}' for embedding generation.")
                 try:
                     # Use a temporary engine instance, don't change self.engine/self.current_model
                     temp_engine = SentenceTransformer(model_to_use['model_hf_name'],
                                                       trust_remote_code=self.config.get('embeddings_model_trust_code', True))
                     temp_device = self.resolve_device(self.config['embeddings_device'])
                     local_engine = temp_engine.to(torch.device(temp_device))
                 except Exception as e:
                      raise IcotqError(f"Failed to temporarily load model '{target_model_name}' for embedding: {e}") from e
            else:
                 # Use the currently loaded engine
                 local_engine = self.engine

            # Load or initialize the tensor for this specific model
            try:
                 # Try loading existing tensor for this model (even if current)
                 # No consistency check needed here, we might be purging
                 self._load_tensor_internal(target_model_name, check_consistency=False)
                 # If successful, self.embeddings_matrix is updated *if* it's the current model
                 # If not current model, we need to load it into a local var
                 if self.current_model and self.current_model['model_name'] == target_model_name:
                      local_embeddings_matrix = self.embeddings_matrix
                 else:
                      # Reload explicitly into local var if not current model
                      temp_tensor_path = self._get_tensor_path(target_model_name)
                      if os.path.exists(temp_tensor_path):
                            temp_device = self.resolve_device(self.config['embeddings_device'])
                            local_embeddings_matrix = torch.load(temp_tensor_path, map_location=torch.device(temp_device))
                      else:
                           local_embeddings_matrix = None

            except IcotqError as e:
                 self.log.warning(f"Could not load existing tensor for '{target_model_name}': {e}. Starting fresh if needed.")
                 local_embeddings_matrix = None

        elif self.current_model and self.engine:
            # Use the globally current model
            model_to_use = self.current_model
            target_model_name = model_to_use['model_name']
            local_engine = self.engine
            # Tensor is already loaded (or None) in self.embeddings_matrix
            local_embeddings_matrix = self.embeddings_matrix
            self.log.info(f"Generating embeddings for current model: '{target_model_name}'")
        else:
            raise IcotqError("Cannot generate embeddings: No current embeddings model loaded. Use 'load_model' first.")

        if purge:
            self.log.warning(f"Purging existing embeddings for model '{target_model_name}'.")
            local_embeddings_matrix = None
            # Clear pointers in the library for this model
            for i in range(len(self.lib)):
                if target_model_name in self.lib[i].get('emb_ptrs', {}):
                    del self.lib[i]['emb_ptrs'][target_model_name]
            # Immediate save of library state after purging pointers might be good
            try:
                 self._write_library_internal()
                 self.log.info(f"Library saved after purging pointers for {target_model_name}.")
            except IcotqCriticalError as e:
                 raise IcotqCriticalError("Failed to save library after purging pointers. Aborting embedding generation.") from e

        start_time: float = time.time()
        last_save_time: float = time.time()
        lib_changed_since_last_save = False
        tensor_changed_since_last_save = False

        total_entries = len(self.lib)
        processed_count = 0

        for ind, entry in enumerate(self.lib):
            print(f"\rEmbedding Progress ({target_model_name}): {ind+1}/{total_entries} ({processed_count} processed)", end="", flush=True)

            # Ensure 'emb_ptrs' dict exists
            if 'emb_ptrs' not in entry:
                self.lib[ind]['emb_ptrs'] = {}
                lib_changed_since_last_save = True # Technically changed

            # Skip if embeddings for this model already exist and not purging
            if target_model_name in entry['emb_ptrs'] and not purge:
                continue

            # --- Generate Embeddings for this entry ---
            processed_count += 1
            text_to_embed = entry.get('text')
            if not text_to_embed:
                # self.log.debug(f"Skipping empty text entry: {entry['desc_filename']}")
                 # Ensure no pointer exists if text is empty
                if target_model_name in self.lib[ind]['emb_ptrs']:
                     del self.lib[ind]['emb_ptrs'][target_model_name]
                     lib_changed_since_last_save = True
                continue

            try:
                text_chunks = self.get_chunks(text_to_embed, model_to_use['chunk_size'], model_to_use['chunk_overlap'])
                if not text_chunks:
                    # self.log.debug(f"No chunks generated for: {entry['desc_filename']}")
                     if target_model_name in self.lib[ind]['emb_ptrs']:
                        del self.lib[ind]['emb_ptrs'][target_model_name]
                        lib_changed_since_last_save = True
                     continue

                # self.log.info(f"Encoding {len(text_chunks)} chunks from {entry['desc_filename']}...")
                # Use the determined engine (local_engine)
                embeddings: list[torch.Tensor] = local_engine.encode(
                    sentences=text_chunks,
                    show_progress_bar=False, # Less noisy for many files
                    convert_to_numpy=False,
                    batch_size=32 # Adjust batch size as needed
                 ) # pyright: ignore[reportUnknownMemberType, reportAssignmentType]

                if not embeddings:
                     self.log.warning(f"Encoding produced no embeddings for {entry['desc_filename']}")
                     if target_model_name in self.lib[ind]['emb_ptrs']:
                         del self.lib[ind]['emb_ptrs'][target_model_name]
                         lib_changed_since_last_save = True
                     continue

                emb_matrix_chunk = torch.stack(embeddings).to(local_embeddings_matrix.device if local_embeddings_matrix is not None else torch.device(self.resolve_device(self.config['embeddings_device']))) # Ensure device match

                # Append to the tensor
                if local_embeddings_matrix is None:
                    start_ptr = 0
                    local_embeddings_matrix = emb_matrix_chunk
                else:
                    start_ptr = local_embeddings_matrix.shape[0]
                    local_embeddings_matrix = torch.cat([local_embeddings_matrix, emb_matrix_chunk])

                emb_len = emb_matrix_chunk.shape[0]
                del emb_matrix_chunk # Free memory
                # Update library entry with pointer
                self.lib[ind]['emb_ptrs'][target_model_name] = (start_ptr, emb_len)
                lib_changed_since_last_save = True
                tensor_changed_since_last_save = True

            except Exception as e:
                 self.log.error(f"\nFailed to generate embeddings for {entry['desc_filename']}: {e}", exc_info=True)
                 # Remove potentially partial pointer if error occurred
                 if target_model_name in self.lib[ind]['emb_ptrs']:
                      del self.lib[ind]['emb_ptrs'][target_model_name]
                      lib_changed_since_last_save = True
                 # Continue with the next entry

            # --- Periodic Save ---
            current_time = time.time()
            if save_every_sec > 0 and (current_time - last_save_time > save_every_sec):
                 print(f"\nPerforming periodic save ({target_model_name})...", end="", flush=True)
                 try:
                     if lib_changed_since_last_save:
                         self._write_library_internal() # Save lib + pdf index
                         lib_changed_since_last_save = False
                     if tensor_changed_since_last_save:
                          # Save the tensor for the specific model being processed
                          tensor_path = self._get_tensor_path(target_model_name)
                          self._atomic_save_tensor(local_embeddings_matrix, tensor_path)
                          tensor_changed_since_last_save = False

                     last_save_time = current_time
                     print(" Done.")
                 except IcotqCriticalError as e:
                      print("\nCRITICAL ERROR during periodic save. Aborting embedding generation.", flush=True)
                      # Decide how to handle state - maybe try one final save? Or just abort?
                      # Abort seems safer to avoid propagating potentially corrupt state.
                      raise IcotqCriticalError("Aborting due to periodic save failure.") from e
                 except Exception as e:
                     # Catch unexpected errors during save
                      print(f"\nUNEXPECTED ERROR during periodic save: {e}. Aborting embedding generation.", flush=True)
                      raise IcotqCriticalError("Aborting due to unexpected periodic save failure.") from e


        # --- Final Save ---
        print(f"\nFinalizing embedding generation ({target_model_name})...", end="", flush=True)
        try:
            if lib_changed_since_last_save:
                self._write_library_internal()
            if tensor_changed_since_last_save:
                tensor_path = self._get_tensor_path(target_model_name)
                self._atomic_save_tensor(local_embeddings_matrix, tensor_path)

            print(" Done.")
            self.log.info(f"Embedding generation for model '{target_model_name}' completed. Processed {processed_count} entries.")

            # --- Update global state if the processed model is the current model ---
            if self.current_model and self.current_model['model_name'] == target_model_name:
                self.embeddings_matrix = local_embeddings_matrix
                self.log.info(f"Current model's ({target_model_name}) in-memory tensor updated.")

        except IcotqCriticalError as e:
             print("\nCRITICAL ERROR during final save.", flush=True)
             # Re-raise the critical error
             raise
        except Exception as e:
             print(f"\nUNEXPECTED ERROR during final save: {e}", flush=True)
             raise IcotqCriticalError("Unexpected final save failure.") from e


    def check_clean(self, dry_run: bool = True):
        """
        Checks for inconsistencies in PDF cache, library pointers, and tensor shapes.
        If dry_run is False, attempts to fix detected inconsistencies where possible.
        """
        action = "Checking" if dry_run else "Checking and Cleaning"
        self.log.info(f"{action} IcoTqStore state...")
        with self._modify():
            issues_found = False
            require_reindex: set[str] = set() # Models needing full re-index

            # --- 1. PDF Cache Index vs. Library ---
            self.log.debug("Checking PDF cache index against library...")
            pdf_index_debris = []
            for pdf_desc in self.pdf_index:
                found_in_lib = any(entry['desc_filename'] == pdf_desc for entry in self.lib)
                if not found_in_lib:
                    pdf_index_debris.append(pdf_desc)
                    issues_found = True

            if pdf_index_debris:
                self.log.warning(f"Found {len(pdf_index_debris)} orphaned PDF cache index entries.")
                if not dry_run:
                    self.log.info("Removing orphaned PDF index entries...")
                    pdf_index_changed = False
                    for desc in pdf_index_debris:
                        if desc in self.pdf_index:
                            # Also try removing associated cache file
                            cache_filename = self.pdf_index[desc].get('filename')
                            if cache_filename:
                                cache_filepath = os.path.join(self.pdf_cache_path, os.path.basename(cache_filename))
                                if os.path.exists(cache_filepath):
                                    try:
                                        os.remove(cache_filepath)
                                        self.log.debug(f"Removed orphaned cache file: {cache_filepath}")
                                    except OSError as e:
                                        self.log.warning(f"Failed to remove orphaned cache file {cache_filepath}: {e}")
                            del self.pdf_index[desc]
                            pdf_index_changed = True
                    if pdf_index_changed:
                        # Save immediately after cleaning this part
                        try:
                           self._write_library_internal() # Saves both
                           self.log.info("Saved state after cleaning PDF index.")
                        except IcotqCriticalError as e:
                             self.log.error(f"Failed to save state after cleaning PDF index: {e}")
                             # Continue checking other things, but log the failure

            # --- 2. PDF Cache Files vs. Index ---
            self.log.debug("Checking PDF cache files against index...")
            pdf_cache_file_debris = []
            pdf_index_filenames = {idx['filename'] for idx in self.pdf_index.values() if idx.get('filename')}
            try:
                cache_files = [f for f in os.listdir(self.pdf_cache_path) if os.path.isfile(os.path.join(self.pdf_cache_path, f))]
                for filename in cache_files:
                     if filename.endswith('.json'): continue # Skip index file itself
                     if filename.endswith('.tmp'): # Clean up leftover temp files
                          self.log.warning(f"Found leftover temporary file: {filename}")
                          pdf_cache_file_debris.append(filename)
                          issues_found = True
                     elif os.path.basename(filename) not in pdf_index_filenames:
                          pdf_cache_file_debris.append(filename)
                          issues_found = True
            except FileNotFoundError:
                self.log.warning(f"PDF Cache directory not found at {self.pdf_cache_path}. Skipping file check.")
            except OSError as e:
                self.log.error(f"Error listing PDF cache directory {self.pdf_cache_path}: {e}")

            if pdf_cache_file_debris:
                 self.log.warning(f"Found {len(pdf_cache_file_debris)} orphaned or temporary PDF cache files.")
                 if not dry_run:
                      self.log.info("Removing orphaned/temporary PDF cache files...")
                      for filename in pdf_cache_file_debris:
                           filepath = os.path.join(self.pdf_cache_path, filename)
                           try:
                               os.remove(filepath)
                               self.log.debug(f"Removed cache file: {filepath}")
                           except OSError as e:
                                self.log.warning(f"Failed to remove cache file {filepath}: {e}")
                      # No state save needed here as only external files deleted


            # --- 3. Library Pointers and Tensor Consistency (Per Model) ---
            self.log.debug("Checking library pointers and tensor consistency...")
            all_known_models = {model['model_name'] for model in self.model_list}
            models_with_pointers = set()
            for entry in self.lib:
                 models_with_pointers.update(entry.get('emb_ptrs', {}).keys())

            models_to_check = all_known_models.intersection(models_with_pointers)
            if self.current_model and self.current_model['model_name'] not in models_to_check:
                 # Add current model even if no pointers yet, to check if tensor exists when it shouldn't
                 models_to_check.add(self.current_model['model_name'])

            for model_name in sorted(list(models_to_check)):
                self.log.debug(f"Checking consistency for model '{model_name}'...")
                tensor_path = self._get_tensor_path(model_name)
                tensor_exists = os.path.exists(tensor_path)
                expected_rows = 0
                has_pointers_in_lib = False
                pointer_issues_found = False
                max_index_reached = -1

                # Calculate expected size and check pointer validity
                temp_fat_check: dict[int, int] = {} # index -> count
                for entry in self.lib:
                    if model_name in entry.get('emb_ptrs', {}):
                        has_pointers_in_lib = True
                        start, length = entry['emb_ptrs'][model_name]

                        if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length < 0:
                            self.log.error(f"Model '{model_name}': Invalid pointer format {entry['emb_ptrs'][model_name]} in '{entry['desc_filename']}'.")
                            pointer_issues_found = True
                            issues_found = True
                            # Don't add to expected_rows, mark for re-index
                            require_reindex.add(model_name)
                            continue # Skip fat check for this invalid entry

                        if length == 0:
                             # Zero-length pointers might be okay if text was empty, but check
                             if entry.get('text'):
                                 self.log.warning(f"Model '{model_name}': Zero-length pointer found for non-empty text in '{entry['desc_filename']}'.")
                                 pointer_issues_found = True
                                 issues_found = True
                                 require_reindex.add(model_name)
                             continue # Skip fat check

                        expected_rows += length
                        max_index_reached = max(max_index_reached, start + length -1)

                        # Check for overlaps using FAT check
                        for i in range(start, start + length):
                            temp_fat_check[i] = temp_fat_check.get(i, 0) + 1


                # Check FAT array for overlaps or gaps (up to max index)
                overlaps = {i: count for i, count in temp_fat_check.items() if count > 1}
                gaps = {i for i in range(max_index_reached + 1) if i not in temp_fat_check} - set(overlaps.keys()) # Exclude overlaps from gaps

                if overlaps:
                    self.log.error(f"Model '{model_name}': Pointer overlaps detected at indices: {list(overlaps.keys())[:10]}...") # Show first few
                    pointer_issues_found = True
                    issues_found = True
                    require_reindex.add(model_name)
                # Gaps might be okay if cleanup failed, but usually indicate inconsistency
                # Be less strict about gaps unless size also mismatches?
                # Let's report gaps as warnings for now.
                if gaps:
                     self.log.warning(f"Model '{model_name}': Pointer gaps detected between 0 and {max_index_reached} at indices: {list(gaps)[:10]}...")
                     # Don't automatically trigger reindex for gaps alone unless size mismatches
                     # issues_found = True # Mark as issue? Maybe not critical


                # Check tensor existence and size
                actual_rows = -1
                if tensor_exists:
                    try:
                        # Quick size check without loading full tensor if possible (tricky with torch.load)
                        # Fallback to loading
                        # Temporarily load tensor header or small part if possible? torch.load loads all.
                        temp_tensor = torch.load(tensor_path, map_location='cpu') # Load to CPU to check shape
                        actual_rows = temp_tensor.shape[0]
                        del temp_tensor # Free memory
                    except Exception as e:
                         self.log.error(f"Model '{model_name}': Failed to load tensor {tensor_path} to check size: {e}")
                         actual_rows = -999 # Indicate load failure
                         issues_found = True
                         # If tensor exists but fails to load, it's corrupt -> requires reindex
                         require_reindex.add(model_name)

                    if actual_rows != expected_rows:
                         self.log.error(f"Model '{model_name}': Size inconsistency! Library expects {expected_rows} rows, tensor has {actual_rows} rows.")
                         issues_found = True
                         require_reindex.add(model_name)
                    elif pointer_issues_found: # Size matches but pointers overlap etc.
                         self.log.warning(f"Model '{model_name}': Tensor size matches library ({actual_rows} rows), but pointer errors (overlaps/invalid format) exist.")
                         # Already marked for reindex above
                    elif not has_pointers_in_lib and actual_rows > 0:
                         self.log.warning(f"Model '{model_name}': Tensor exists ({actual_rows} rows) but no library entries point to it.")
                         issues_found = True
                         # Should we delete the tensor or require reindex? Let's suggest reindex.
                         require_reindex.add(model_name)
                    elif not pointer_issues_found and not gaps:
                        self.log.info(f"Model '{model_name}': Consistent. Tensor size: {actual_rows}, Library pointers valid and contiguous.")

                elif has_pointers_in_lib: # Tensor doesn't exist, but library has pointers
                     self.log.error(f"Model '{model_name}': Missing tensor file! Library expects {expected_rows} rows, but {tensor_path} not found.")
                     issues_found = True
                     require_reindex.add(model_name)
                # Else: Tensor doesn't exist, library has no pointers -> Consistent state (just not indexed)


            # --- 4. Perform Fixes (if not dry_run) ---
            if not dry_run and require_reindex:
                 self.log.warning(f"Attempting to fix inconsistencies by re-indexing models: {list(require_reindex)}")
                 for model_name in require_reindex:
                      self.log.info(f"Running 'generate_embeddings(purge=True)' for model '{model_name}'...")
                      try:
                          # Call internal method directly as lock is held
                          self._generate_embeddings_internal(purge=True, model_name_override=model_name)
                          self.log.info(f"Re-indexing for model '{model_name}' completed.")
                      except IcotqError as e:
                           self.log.error(f"Failed to automatically re-index model '{model_name}': {e}. Manual intervention required.")
                           issues_found = True # Ensure issues_found remains true if fix fails
                      except Exception as e:
                           self.log.critical(f"Unexpected critical error during automatic re-index of '{model_name}': {e}", exc_info=True)
                           issues_found = True
                 # Reload current model tensor if it was re-indexed
                 if self.current_model and self.current_model['model_name'] in require_reindex:
                      self.log.info(f"Reloading current model '{self.current_model['model_name']}' tensor after re-index.")
                      try:
                          # Reload, consistency check should now pass
                          self._load_tensor_internal(self.current_model['model_name'], check_consistency=True)
                      except IcotqError as e:
                           self.log.critical(f"CRITICAL: Failed to reload tensor or inconsistency detected *after* automatic re-index of {self.current_model['model_name']}: {e}")
                           self.embeddings_matrix = None # Safety


            # --- Final Report ---
            if not issues_found:
                self.log.info(f"{action} completed. No issues found.")
            elif dry_run:
                self.log.warning(f"{action} completed. Issues found (see logs). Run without '--dry-run' to attempt fixes.")
            elif require_reindex and issues_found: # If reindex ran but issues persist (e.g., fix failed)
                 self.log.error(f"{action} completed. Some automatic fixes failed. Manual intervention likely required (e.g., 'index purge').")
            else: # Fixes attempted and seemed successful, or only cache cleaning done
                 self.log.info(f"{action} completed. Issues found and fixes attempted (see logs).")


    # === Search Functionality ===

    def _search_vect_internal(self, text: str) -> tuple[list[tuple[int, float]], torch.Tensor]:
        """Internal search vector generation. Assumes lock is held, engine/matrix valid."""
        if self.embeddings_matrix is None or self.engine is None:
             # This check should ideally be done by the caller, but double-check
             raise IcotqError("Search prerequisites not met: Embeddings matrix or engine not available.")

        # Ensure engine is on the correct device (should match matrix device)
        if str(self.engine.device) != str(self.embeddings_matrix.device):
             self.log.warning(f"Engine device ({self.engine.device}) differs from matrix device ({self.embeddings_matrix.device}). Moving engine.")
             try:
                 self.engine = self.engine.to(self.embeddings_matrix.device)
             except Exception as e:
                  raise IcotqError(f"Failed to move search engine to device {self.embeddings_matrix.device}: {e}") from e

        try:
             vects: list[torch.Tensor] = self.engine.encode(
                 sentences=[text],
                 show_progress_bar=False, # Usually fast for one sentence
                 convert_to_numpy=False
             ) # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
        except Exception as e:
             raise IcotqError(f"Failed to generate embedding for search text: {e}") from e

        if not vects:
            raise IcotqError("Failed to calculate embedding for search text (empty result).")

        search_vect: torch.Tensor = vects[0]
        if len(vects) > 1:
            self.log.warning("Search text generated multiple vectors; using only the first.")

        # Perform similarity search (matrix multiplication)
        try:
            # Ensure search_vect is on the same device as the matrix
            search_vect = search_vect.to(self.embeddings_matrix.device)
            similarities = torch.matmul(self.embeddings_matrix, search_vect)
            # Move results to CPU for sorting and returning
            simil_list: list[float] = similarities.cpu().numpy().tolist()
        except Exception as e:
             raise IcotqError(f"Error during similarity calculation: {e}") from e

        indexed_simil: list[tuple[int, float]] = list(enumerate(simil_list))
        sorted_simil: list[tuple[int, float]] = sorted(indexed_simil, key=lambda x: x[1], reverse=True)

        return sorted_simil, search_vect.cpu() # Return search vector on CPU


    def _yellow_line_it_internal(self, text: str, search_embeddings_cpu: torch.Tensor, context_length: int, context_steps: int) -> np.typing.NDArray[np.float32]:
        """Internal yellow liner. Assumes lock is held, engine valid."""
        if self.engine is None:
             raise IcotqError("Cannot perform yellow-lining: Engine not available.")

        if not text: return np.array([], dtype=np.float32)

        # Generate context snippets
        clr: list[str] = []
        text_len = len(text)
        for i in range(0, text_len, context_steps):
            i0 = max(0, i - context_length // 2)
            i1 = min(text_len, i + context_length // 2 + (context_length % 2)) # Ensure full context length

            # Adjust window if it hits boundaries to maintain length
            if i0 == 0 and i1 < text_len:
                 i1 = min(text_len, i0 + context_length)
            elif i1 == text_len and i0 > 0:
                 i0 = max(0, i1 - context_length)

            snippet = text[i0:i1]
            if snippet: # Avoid encoding empty strings if steps/length are odd
                clr.append(snippet)

        if not clr:
            # Fallback if no snippets generated (e.g., text shorter than context/steps)
            clr = [text]

        # Encode snippets
        try:
            target_device = torch.device(self.resolve_device(self.config['embeddings_device']))
            if str(self.engine.device) != str(target_device):
                 self.log.warning(f"Moving engine to {target_device} for yellow-lining.")
                 self.engine = self.engine.to(target_device)

            snippet_embeddings: list[torch.Tensor] = self.engine.encode(
                 sentences=clr,
                 show_progress_bar=False,
                 convert_to_numpy=False,
                 batch_size=128 # Larger batch size possible here
            ) # pyright: ignore[reportUnknownMemberType, reportAssignmentType]

            if not snippet_embeddings:
                 self.log.warning("Yellow-lining failed to generate snippet embeddings.")
                 return np.array([], dtype=np.float32)

            emb_matrix = torch.stack(snippet_embeddings) # On target_device
            search_embeddings = search_embeddings_cpu.to(emb_matrix.device) # Move search vec to snippet device

            # Calculate similarities
            yellow_vect: np.typing.NDArray[np.float32] = torch.matmul(emb_matrix, search_embeddings).cpu().numpy()

            # Normalize scores (optional but often useful)
            min_score, max_score = yellow_vect.min(), yellow_vect.max()
            if max_score > min_score:
                 yellow_vect = (yellow_vect - min_score) / (max_score - min_score)
            elif max_score > 0 : # All scores are the same positive value
                 yellow_vect.fill(1.0)
            else: # All scores are zero or negative
                 yellow_vect.fill(0.0)

            return yellow_vect

        except Exception as e:
             self.log.error(f"Error during yellow-lining: {e}", exc_info=True)
             return np.array([], dtype=np.float32) # Return empty on error


    def search(self, search_text:str, max_results:int=10, yellow_liner:bool=False, context_length:int=16, context_steps:int=4, compression_mode:str="none") -> list[SearchResult]:
        """
        Performs vector search, resolves results to library entries, handles merging,
        and optionally adds yellow-liner highlighting.
        """
        with self._lock: # Acquire lock for reading lib and matrix
            if self.current_model is None or self.embeddings_matrix is None or self.engine is None:
                self.log.error("Search cannot proceed: Model or embeddings not loaded/available.")
                return []
            if self.embeddings_matrix.shape[0] == 0:
                 self.log.warning("Search cannot proceed: Embeddings matrix is empty.")
                 return []

            target_model_name = self.current_model['model_name']

            try:
                # 1. Get Top-K Similarity Scores
                sorted_simil_all, search_embeddings_cpu = self._search_vect_internal(search_text)
                # Limit results early
                top_k_simil = sorted_simil_all[:max_results * 2] # Get more initially for potential merging

                # 2. Resolve Indices to Library Entries
                # Create a quick lookup map from index range to entry
                idx_to_entry_map: dict[int, tuple[LibEntry, int]] = {} # tensor_idx -> (entry, offset_in_entry)
                for entry in self.lib:
                     if target_model_name in entry.get('emb_ptrs', {}):
                         start, length = entry['emb_ptrs'][target_model_name]
                         if length > 0 and start >= 0:
                              for i in range(length):
                                   idx_to_entry_map[start + i] = (entry, i)

                resolved_list: list[tuple[str, int, float, LibEntry, int]] = [] # desc, tensor_idx, cosine, entry, offset_in_entry
                processed_tensor_indices = set() # Avoid duplicates if top_k has same index

                for tensor_idx, cosine in top_k_simil:
                    if tensor_idx in processed_tensor_indices: continue
                    processed_tensor_indices.add(tensor_idx)

                    if tensor_idx in idx_to_entry_map:
                         entry, offset = idx_to_entry_map[tensor_idx]
                         desc = entry['desc_filename']
                         resolved_list.append((desc, tensor_idx, cosine, entry, offset))
                    else:
                         # This should ideally not happen if matrix/lib are consistent
                         self.log.warning(f"Search result index {tensor_idx} (score {cosine:.4f}) could not be mapped to any library entry for model '{target_model_name}'.")

                # 3. Merge Consecutive Results within the Same Document
                # Sort by description, then by tensor index to group results by document
                resolved_list.sort(key=lambda x: (x[0], x[1]))

                merged_results: list[tuple[str, int, int, float, LibEntry]] = [] # desc, start_tensor_idx, count, max_cosine, entry
                if not resolved_list: return []

                # Initialize with the first result
                current_desc, current_idx, current_cos, current_entry, current_offset = resolved_list[0]
                current_count = 1
                max_cosine = current_cos

                for i in range(1, len(resolved_list)):
                    next_desc, next_idx, next_cos, next_entry, next_offset = resolved_list[i]

                    # Check if same document AND consecutive chunk index
                    if (next_desc == current_desc and
                        next_entry is current_entry and # Ensure same entry object
                        next_offset == current_offset + current_count): # Check if chunk index is consecutive

                        # Merge: increment count, update max cosine
                        current_count += 1
                        max_cosine = max(max_cosine, next_cos)
                        # Don't append yet, continue merging

                    else:
                        # Not mergeable: Append the completed merged block
                        merged_results.append((current_desc, current_idx - current_offset, current_count, max_cosine, current_entry)) # Store start idx of span

                        # Start a new block
                        current_desc, current_idx, current_cos, current_entry, current_offset = next_desc, next_idx, next_cos, next_entry, next_offset
                        current_count = 1
                        max_cosine = current_cos

                # Append the last block
                merged_results.append((current_desc, current_idx - current_offset, current_count, max_cosine, current_entry))

                # Sort merged results by cosine score (descending) and limit
                merged_results.sort(key=lambda x: x[3], reverse=True)
                final_merged_list = merged_results[:max_results]


                # 4. Format Final Results (Get Chunks, Compress, Yellow-line)
                search_results_final: list[SearchResult] = []
                for desc, start_tensor_idx, count, cosine, entry in final_merged_list:
                    # Calculate offset within the specific entry
                    entry_start_ptr, _ = entry['emb_ptrs'][target_model_name]
                    offset_in_entry = start_tensor_idx - entry_start_ptr

                    # Get the text chunk (potentially spanning multiple base chunks)
                    chunk_text: str = self.get_span_chunk(entry['text'],
                                                         offset_in_entry, # Start offset within this entry
                                                         count, # Number of base chunks to span
                                                         self.current_model['chunk_size'],
                                                         self.current_model['chunk_overlap'])

                    # Apply compression
                    if compression_mode == "light":
                        new_chunk = chunk_text.replace("\n", " ").replace("\r"," ").replace("\t", " ") # Replace whitespace chars
                        while "  " in new_chunk: new_chunk = new_chunk.replace("  ", " ") # Collapse spaces
                        chunk_text = new_chunk.strip()
                    elif compression_mode == "full":
                        new_chunk = chunk_text.replace("\n", " ").replace("\r"," ").replace("\t", " ") # Replace whitespace chars
                        while "  " in new_chunk: new_chunk = new_chunk.replace("  ", " ") # Collapse spaces
                        chunk_text = new_chunk.strip()
                        # Additional compression could be added here (e.g., removing punctuation?)

                    # Apply yellow liner if requested
                    yellow_liner_weights: np.typing.NDArray[np.float32] | None = None
                    if yellow_liner and chunk_text:
                         try:
                             yellow_liner_weights = self._yellow_line_it_internal(chunk_text, search_embeddings_cpu, context_length, context_steps)
                         except IcotqError as yl_e:
                             self.log.error(f"Failed to generate yellow-liner for '{desc}': {yl_e}")

                    sres: SearchResult = {
                        'cosine': cosine,
                        'index': start_tensor_idx, # Index of the first chunk in the span
                        'offset': offset_in_entry, # Offset of the first chunk within the document's chunks
                        'desc': desc,
                        'chunk': chunk_text,
                        'text': entry['text'], # Full text included
                        'yellow_liner': yellow_liner_weights
                    }
                    search_results_final.append(sres)

                return search_results_final

            except IcotqError as e:
                 self.log.error(f"Search failed: {e}", exc_info=True)
                 return []
            except Exception as e:
                 self.log.critical(f"Unexpected critical error during search: {e}", exc_info=True)
                 # Raise? Or just return empty list? Let's return empty for now.
                 return []


    # === Server Functionality ===

    async def search_handler(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handles incoming search requests via HTTP."""
        try:
            data: SearchRequest = await request.json()
            search_text = data.get('search_text')
            if not search_text:
                return aiohttp.web.json_response({"error": "Missing 'search_text' field"}, status=400)

            max_results = data.get('max_results', 10) # Default 10
            yellow_liner = data.get('yellow_liner', False)
            context_length = data.get('context_length', 16)
            context_steps = data.get('context_steps', 4)
            compression_mode = data.get('compression_mode', 'none')

            # Validate inputs
            if not isinstance(max_results, int) or max_results <= 0: max_results = 10
            if not isinstance(yellow_liner, bool): yellow_liner = False
            if not isinstance(context_length, int) or context_length <= 0: context_length = 16
            if not isinstance(context_steps, int) or context_steps <= 0: context_steps = 4
            if compression_mode not in ['none', 'light', 'full']: compression_mode = 'none'

            self.log.info(f"Received search request: text='{search_text[:50]}...', max={max_results}, yellow={yellow_liner}")

            # Perform search (this acquires the lock)
            search_results = self.search(
                search_text,
                max_results=max_results,
                yellow_liner=yellow_liner,
                context_length=context_length,
                context_steps=context_steps,
                compression_mode=compression_mode
            )

            self.log.info(f"Responding to search request with {len(search_results)} results.")

            # Convert numpy arrays in results for JSON serialization
            for result in search_results:
                if result.get('yellow_liner') is not None:
                    result['yellow_liner'] = result['yellow_liner'].tolist() # type: ignore

            return aiohttp.web.json_response(search_results) # type: ignore # Allow list directly

        except json.JSONDecodeError:
            return aiohttp.web.json_response({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            self.log.error(f"Error processing search request: {e}", exc_info=True)
            return aiohttp.web.json_response({"error": "Internal server error"}, status=500)


    def _server_task(self, host: str, port: int, in_thread: bool = False):
        """The asyncio server task runner."""
        self.log.info(f"Starting server task (in_thread={in_thread})...")
        loop = None
        try:
            app = aiohttp.web.Application()
            app.router.add_post('/search', self.search_handler)

            runner = aiohttp.web.AppRunner(app)

            if in_thread:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError: # May happen if no loop exists and not in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.loop = loop # Store the loop instance

            async def start_runner():
                await runner.setup()
                site = aiohttp.web.TCPSite(runner, host, port)
                await site.start()
                self.log.info(f"Server started successfully at http://{host}:{port}")
                # Keep running until self.server_running is False
                while self.server_running:
                    await asyncio.sleep(0.5) # Check periodically
                self.log.info("Server shutdown signal received.")
                await site.stop()
                self.log.info("Server site stopped.")

            async def cleanup_runner():
                await runner.cleanup()
                self.log.info("Server runner cleaned up.")

            # Run start_runner until server_running is False, then cleanup
            loop.run_until_complete(start_runner())
            loop.run_until_complete(cleanup_runner())

        except OSError as e:
             if "address already in use" in str(e).lower():
                  self.log.critical(f"Server failed to start: Port {port} on host '{host}' is already in use.")
             else:
                  self.log.critical(f"Server failed to start due to OS error: {e}", exc_info=True)
             # Ensure server_running is False if startup failed critically
             self.server_running = False
        except Exception as e:
            self.log.critical(f"Server task encountered a critical error: {e}", exc_info=True)
            self.server_running = False # Ensure flag is reset on unexpected exit
        finally:
            if loop and not loop.is_closed():
                 # Close the loop if it was created for a thread
                 if in_thread:
                     loop.close()
                 self.log.info("Server asyncio loop closed.")
            self.loop = None # Clear loop instance
            self.log.info("Server task finished.")


    def start_server(self, host: str = "0.0.0.0", port: int = 8080, background: bool = False):
        """Starts the search API server."""
        if self.server_running:
            self.log.warning("Server is already running or starting.")
            return

        self.server_running = True # Set flag early to prevent race conditions

        if background:
            self.log.info("Starting server in background thread...")
            self.server_thread = threading.Thread(
                target=self._server_task,
                args=(host, port, True), # Pass in_thread=True
                daemon=True # Allow program exit even if server thread is running
            )
            self.server_thread.start()
            # Give the server a moment to start up/fail
            time.sleep(1)
            if not self.server_running: # Check if startup failed
                 self.log.error("Server failed to start in background thread (see logs).")
                 self.server_thread = None
        else:
            self.log.info("Starting server in foreground (blocking)...")
            try:
                # Run directly in the current thread's event loop (or create one)
                self._server_task(host, port, False) # Pass in_thread=False
            except KeyboardInterrupt:
                 self.log.info("Server stopped by user (KeyboardInterrupt).")
            finally:
                 # Ensure flag is reset if run in foreground and it exits
                 self.server_running = False


    def stop_server(self):
        """Stops the running search API server."""
        if not self.server_running and self.server_thread is None:
            self.log.warning("Server is not running.")
            return

        self.log.info("Attempting to stop the server...")
        self.server_running = False # Signal the server loop to stop

        # Stop the asyncio loop if it's accessible and running
        if self.loop and self.loop.is_running():
             # self.loop.stop() # Simple stop might not be enough, need to wake it
             self.loop.call_soon_threadsafe(self.loop.stop) # Better way to stop from another thread
             self.log.debug("Requested asyncio loop stop.")

        # Wait for the server thread to finish if it exists
        if self.server_thread and self.server_thread.is_alive():
            self.log.info("Waiting for server thread to exit...")
            self.server_thread.join(timeout=10) # Wait up to 10 seconds
            if self.server_thread.is_alive():
                 self.log.warning("Server thread did not exit gracefully after 10 seconds.")
            else:
                 self.log.info("Server thread finished.")
            self.server_thread = None
        else:
             # If not running in a thread, flag should already be false or loop stopped
             pass

        # Final check
        if self.server_running:
             self.log.warning("Server flag indicates still running after stop attempt.")
             self.server_running = False # Force flag off

        self.log.info("Server stop sequence completed.")


    # === Static Utility Methods ===

    @staticmethod
    def get_chunk_ptr(index: int, chunk_size: int, chunk_overlap: int) -> int:
        """Calculates the start character index of a chunk."""
        # Ensure non-negative index
        index = max(0, index)
        # Ensure overlap is less than chunk size
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0:
            # Avoid division by zero or infinite loops if overlap >= size
            # Fallback to non-overlapping chunks
            step = chunk_size
        chunk_ptr: int = index * step
        return chunk_ptr

    @staticmethod
    def get_chunk(text: str, index: int, chunk_size: int, chunk_overlap: int ) -> str:
        """Extracts a single text chunk based on index."""
        if not text: return ""
        chunk_start: int = IcoTqStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        chunk = text[chunk_start : chunk_start + chunk_size]
        return chunk

    @staticmethod
    def get_span_chunk(text:str, index: int, count:int, chunk_size: int, chunk_overlap: int):
        """Extracts a span of text covering 'count' base chunks starting at 'index'."""
        if not text or count < 1:
            return ""
        # Ensure overlap is less than chunk size
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: step = chunk_size # Fallback

        chunk_start: int = IcoTqStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        # End position calculation needs care:
        # It's the start of the *last* chunk in the span + chunk_size
        # The last chunk in the span starts at index + (count - 1)
        # last_chunk_start = IcoTqStore.get_chunk_ptr(index + count - 1, chunk_size, chunk_overlap)
        # chunk_end = last_chunk_start + chunk_size
        # Simpler way: Start + (number of steps * step_size) + overlap
        # Number of steps = count - 1
        chunk_end = chunk_start + chunk_size + step * (count - 1)

        chunk = text[chunk_start : chunk_end]
        return chunk

    @staticmethod
    def get_chunks(text:str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Splits text into overlapping chunks."""
        if not text: return []
        # Ensure overlap is less than chunk size
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: step = chunk_size # Fallback

        text_len = len(text)
        chunks = []
        for i in range(0, text_len, step):
             chunk = text[i : i + chunk_size]
             if not chunk: break # Stop if we somehow get empty chunk
             # Only add chunk if it contains meaningful content (e.g., not just whitespace overlap)
             # This check might be too simple. Consider if overlap region matters.
             # Let's add all non-empty chunks for now.
             chunks.append(chunk)
             # Check if the last chunk fully covers the remaining text
             if i + chunk_size >= text_len:
                 break

        # Alternative calculation (less intuitive maybe):
        # num_chunks = 0
        # if text_len > chunk_size:
        #     num_chunks = (text_len - chunk_size + step -1) // step + 1 # Ceil division equiv
        # elif text_len > 0:
        #     num_chunks = 1
        # text_chunks = [IcoTqStore.get_chunk(text, i, chunk_size, chunk_overlap) for i in range(num_chunks)]

        return chunks

    def resolve_device(self, device: str | None = None) -> str:
        """Resolves 'auto' device to cuda, mps, or cpu."""
        dev = device if device else self.config.get('embeddings_device', 'auto')
        if dev == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                # MPS support can be flaky, maybe add explicit check or allow fallback?
                # For now, assume it's available if reported true.
                return 'mps'
            else:
                return 'cpu'
        elif dev in ['cuda', 'mps', 'cpu']:
             # Validate requested device
             if dev == 'cuda' and not torch.cuda.is_available():
                  self.log.warning("CUDA requested but not available, falling back to CPU.")
                  return 'cpu'
             if dev == 'mps' and not torch.backends.mps.is_available():
                  self.log.warning("MPS requested but not available, falling back to CPU.")
                  return 'cpu'
             return dev
        else:
             self.log.warning(f"Unsupported device '{dev}' requested, falling back to 'auto'.")
             # Recurse once to resolve 'auto'
             return self.resolve_device('auto')
