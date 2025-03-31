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

# Keep TypedDict, cast, NotRequired, Any, Generator for now
from typing import TypedDict, NotRequired, Any
from collections.abc import Generator
import pymupdf  # pyright: ignore[reportMissingTypeStubs]

try:
    import pymupdf4llm  # pyright: ignore[reportMissingTypeStubs]
    PYMUPDF4LLM_AVAILABLE:bool = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]


class TqSource(TypedDict):
    name: str
    tqtype: str
    path: str
    file_types: list[str] # Changed from List[str]

class IcotqConfig(TypedDict):
    icotq_path: str
    tq_sources: list[TqSource] # Changed from List[TqSource]
    embeddings_model_name: str
    embeddings_device: str
    embeddings_model_trust_code: bool
    auto_fix_inconsistency: bool

class LibEntry(TypedDict):
    source_name: str
    filename: str
    desc_filename: str
    text: str
    emb_ptrs: dict[str, tuple[int, int]]

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
    yellow_liner: np.typing.NDArray[np.float32] | None # Changed from Optional[...]

class EmbeddingsModel(TypedDict):
    model_hf_name: str
    model_name: str
    emb_dim: int
    max_input_token: int
    chunk_size: int
    chunk_overlap: int

class ModelInfo(TypedDict): # For listing models
    model_hf_name: str
    model_name: str

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
    # Accept config_file_override: str | None
    def __init__(self, config_file_override: str | None = None) -> None:
        self.log:logging.Logger = logging.getLogger("IcoTqStore")
        # Disable log spam
        tmp = logging.getLogger("transformers_modules")
        tmp.setLevel(logging.ERROR)
        tmp_st = logging.getLogger("sentence_transformers")
        tmp_st.setLevel(logging.WARNING)

        if PYMUPDF4LLM_AVAILABLE:
            self.log.info("pymupdf4llm library found, will be used as fallback for PDF text extraction.")
        else:
            self.log.warning("pymupdf4llm library not found. PDF text extraction will rely solely on pymupdf. Install with: pip install pymupdf4llm")

        # Determine config file path
        self.config_file: str
        if config_file_override:
            self.config_file = config_file_override
            self.log.info(f"Using overridden config file path: {self.config_file}")
        else:
            config_path = os.path.expanduser("~/IcoTqStore/config")
            if not os.path.isdir(config_path):
                try:
                    os.makedirs(config_path)
                except OSError as e:
                    self.log.error(f"Failed to create default config directory {config_path}: {e}")
            self.config_file = os.path.join(config_path, "icoqt.json")
            self.log.info(f"Using default config file path: {self.config_file}")

        # --- Core State ---
        self.lib: list[LibEntry] = [] # Changed from List[LibEntry]
        self.pdf_index:dict[str, PDFIndex] = {}
        self.config:IcotqConfig
        self.current_model: EmbeddingsModel | None = None # Changed from Optional[...]
        self.engine: SentenceTransformer | None = None # Changed from Optional[...]
        self.device: str | None = None # Changed from Optional[...]
        self.embeddings_matrix: torch.Tensor | None = None # Changed from Optional[...]
        self.root_path:str = ""
        self.embeddings_path: str = ""
        self.pdf_cache_path: str = "" # Added type hint
        self.model_list: list[EmbeddingsModel] = [] # Changed from List[EmbeddingsModel]

        # --- Concurrency Control ---
        self._lock: threading.Lock = threading.Lock()

        # --- Server State ---
        self.server_running:bool = False
        self.loop:asyncio.AbstractEventLoop | None = None # Changed from Optional[...]
        self.server_thread:threading.Thread | None = None # Changed from Optional[...]

        self._load_or_init_config()
        self._validate_config_paths()
        self._load_or_init_model_list()
        self._ensure_storage_dirs()

        # --- Load Initial State (Inside Lock) ---
        with self._lock:
            if self.config['embeddings_model_name']:
                try:
                    _ = self._load_model_internal(self.config['embeddings_model_name'],
                                            self.config['embeddings_device'],
                                            self.config['embeddings_model_trust_code'])
                except IcotqError as e:
                    self.log.error(f"Failed to load initial model specified in config: {e}")

            self._read_library_internal()
            if self.current_model:
                try:
                    _ = self._load_tensor_internal(model_name=self.current_model['model_name'])
                except IcotqConsistencyError as e:
                    self.log.error(f"Consistency Error loading tensor for initial model '{self.current_model['model_name']}': {e}")
                    if self.config.get('auto_fix_inconsistency', False):
                        self.log.warning(f"Attempting automatic fix (re-index) for model '{self.current_model['model_name']}' due to inconsistency.")
                        try:
                            self._generate_embeddings_internal(purge=True, model_name_override=self.current_model['model_name'])
                            self.log.info(f"Automatic re-index for '{self.current_model['model_name']}' completed.")
                            _ = self._load_tensor_internal(model_name=self.current_model['model_name'])
                        except IcotqError as fix_e:
                            self.log.error(f"Automatic fix failed for model '{self.current_model['model_name']}': {fix_e}")
                            self.embeddings_matrix = None
                            raise IcotqCriticalError(f"Failed initial load and automatic fix for model '{self.current_model['model_name']}'. Manual intervention likely required ('index purge').") from fix_e
                    else:
                        self.log.warning("Automatic fixing is disabled. Manual 'index purge' may be required.")
                        self.embeddings_matrix = None
                except IcotqError as e:
                     self.log.error(f"Error loading tensor for initial model: {e}")
                     self.embeddings_matrix = None

        self.log.info("IcoTqStore initialized.")
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
                    _ = iqc.setdefault('auto_fix_inconsistency', False)
                    self.config = iqc
                self.log.info(f"Loaded configuration from {self.config_file}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.log.error(f"Failed to load or parse config file {self.config_file}: {e}. Using default config.")
                self._create_default_config()
                try:
                    self._save_config_internal()
                except IcotqError as save_e:
                     self.log.error(f"Failed to save default config: {save_e}")
        else:
            self._create_default_config()
            self.log.warning(f"Created default configuration at {self.config_file}, please review!")
            try:
                self._save_config_internal()
            except IcotqError as save_e:
                self.log.error(f"Failed to save initial default config: {save_e}")

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
            'embeddings_model_name': 'ibm-granite/granite-embedding-107m-multilingual',
            'embeddings_device': 'auto',
            'embeddings_model_trust_code': True,
            'auto_fix_inconsistency': False
        })

    def _validate_config_paths(self):
        """Validates paths and sources in the configuration."""
        self.root_path = os.path.expanduser(self.config['icotq_path'])
        if not self.root_path:
             raise IcotqConfigurationError("`icotq_path` cannot be empty in configuration.")

        valid_sources: list[TqSource] = [] # Changed from List
        known_types: list[str] = ['txt', 'md', 'pdf'] # Changed from List
        known_tqtypes: list[str] = ['calibre_library', 'folder']

        for source in self.config['tq_sources']:
            valid = True
            source_path_expanded = os.path.expanduser(source.get('path', ''))
            for tp in source['file_types']:
                if tp not in known_types:
                    self.log.error(f"Source {source} has invalid file type {tp}, allowed are {known_types}, ignoring this source!")
                    valid = False
                    break
            if source['tqtype'] not in known_tqtypes:
                self.log.error(f"Source {source} has invalid tqtype {source['tqtype']}, valid are {known_tqtypes}, ignoring this source!")
                valid = False

            if valid:
                source['path'] = source_path_expanded
                valid_sources.append(source)
            else:
                self.log.warning(f"Please fix configuration file: {self.config_file}")

        if len(valid_sources) < len(self.config['tq_sources']):
            self.config['tq_sources'] = valid_sources
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
            { # granite-107m
                'model_hf_name': 'ibm-granite/granite-embedding-107m-multilingual',
                'model_name': 'granite-embedding-107m-multilingual',
                'emb_dim': 384, 
                'max_input_token': 512,
                'chunk_size': 2048, 
                'chunk_overlap': 2048 // 3
            },
            {
                'model_hf_name': 'ibm-granite/granite-embedding-278m-multilingual',
                'model_name': 'granite-embedding-278m-multilingual',
                'emb_dim': 768,
                'max_input_token': 512,
                'chunk_size': 2048,
                'chunk_overlap': 2048 // 3
            },
            { # nomic-v2
                'model_hf_name': 'nomic-ai/nomic-embed-text-v2-moe',
                'model_name': 'nomic-embed-text-v2-moe',
                'emb_dim': 768,  #  Matryoshka Embeddings
                'max_input_token': 512,
                'chunk_size': 2048,
                'chunk_overlap': 2048 // 3
            },
            { # nomic-v1.5
                'model_hf_name': 'nomic-ai/nomic-embed-text-v1.5',
                'model_name': 'nomic-embed-text-v1.5',
                'emb_dim': 768, 
                'max_input_token': 2048,
                'chunk_size': 4096, 
                'chunk_overlap': 4096 // 3
            },
            { # all-MiniLM-L6-v2 (Added for testing)
                'model_hf_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'model_name': 'all-MiniLM-L6-v2',
                'emb_dim': 384,
                'max_input_token': 512, # Use underlying model's limit or common practice
                'chunk_size': 1024,     # Adjust based on token limit and desired context
                'chunk_overlap': 1024 // 3
            },
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
        self.pdf_cache_path = os.path.join(self.root_path, "PDFTextCache")


    # === Atomic Save Helpers ===

    def _atomic_save_json(self, data: Any, final_path: str):  # pyright:ignore[reportExplicitAny, reportAny]
        """Atomically saves data as JSON using a temporary file and rename."""
        # ... (logic remains the same, no typing changes needed here) ...
        temp_fd, temp_path = None, None
        fd_needs_explicit_close = False
        try:
            temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix=os.path.basename(final_path) + '.tmp')
            fd_needs_explicit_close = True

            with os.fdopen(temp_fd, 'w') as f:
                fd_needs_explicit_close = False
                json.dump(data, f, indent=2)

            os.replace(temp_path, final_path)
            temp_path = None # Prevent removal in finally

        except (IOError, OSError, json.JSONDecodeError, TypeError) as e:
            self.log.error(f"Error during atomic save to {final_path}: {e}")
            if temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError as rm_e: self.log.error(f"Failed to remove temporary file {temp_path} after save error: {rm_e}")
            raise IcotqCriticalError(f"Failed to atomically save JSON to {final_path}: {e}\n{traceback.format_exc()}") from e
        finally:
             if fd_needs_explicit_close and temp_fd is not None:
                 try: os.close(temp_fd)
                 except OSError as close_e: self.log.warning(f"Ignoring error closing temp_fd {temp_fd} in finally: {close_e}")
             if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.log.warning(f"Removed leftover temp file in finally block: {temp_path}")
                except OSError as rm_e: self.log.error(f"Failed to remove leftover temp file {temp_path} in finally block: {rm_e}")


    def _atomic_save_tensor(self, tensor_data: torch.Tensor | None, final_path: str): # Changed Optional
        """Atomically saves a PyTorch tensor using a temporary file and rename."""
        # ... (logic remains the same) ...
        if tensor_data is None:
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                    self.log.info(f"Removed obsolete tensor file {final_path}")
                except OSError as e:
                     raise IcotqCriticalError(f"Failed to remove obsolete tensor file {final_path}: {e}") from e
            return

        temp_fd, temp_path = None, None
        try:
            (temp_fd, temp_path) = tempfile.mkstemp(dir=os.path.dirname(final_path), prefix=os.path.basename(final_path) + '.tmp')
            os.close(temp_fd)
            temp_fd = None
            torch.save(tensor_data, temp_path)  # pyright:ignore[reportUnknownMemberType]
            os.replace(temp_path, final_path)
            temp_path = None # Prevent removal
        except (IOError, OSError, RuntimeError) as e:
            self.log.error(f"Error during atomic tensor save to {final_path}: {e}")
            if temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError as rm_e: self.log.error(f"Failed to remove temporary tensor file {temp_path} after save error: {rm_e}")
            raise IcotqCriticalError(f"Failed to atomically save tensor to {final_path}: {e}\n{traceback.format_exc()}") from e
        finally:
             if temp_fd is not None:
                 try: os.close(temp_fd)
                 except OSError: pass
             if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.log.warning(f"Removed leftover temp tensor file in finally: {temp_path}")
                except OSError as rm_e: self.log.error(f"Failed to remove leftover temp tensor file {temp_path} in finally: {rm_e}")


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
        # ... (no typing changes needed) ...
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
        except IcotqCriticalError as e:
             raise IcotqCriticalError(f"Failed to save library or PDF index state: {e}") from e

    def _save_tensor_internal(self, model_name: str | None = None) -> bool: # Changed Optional
        """
        Saves the in-memory embeddings_matrix for the specified model atomically.
        Returns True on success, raises IcotqCriticalError on failure.
        """
        model_to_save: EmbeddingsModel | None = None # Changed Optional
        tensor_to_save: torch.Tensor | None = None # Changed Optional

        if model_name:
            if self.current_model and self.current_model['model_name'] == model_name:
                model_to_save = self.current_model
                tensor_to_save = self.embeddings_matrix
            else:
                 raise IcotqError(f"Cannot save tensor for non-current model '{model_name}' without explicit tensor data.")
        elif self.current_model:
            model_to_save = self.current_model
            tensor_to_save = self.embeddings_matrix
        else:
            raise IcotqError("Cannot save tensor: No model is currently loaded.")

        if model_to_save:
            embeddings_tensor_file = self._get_tensor_path(model_to_save['model_name'])
            try:
                self._atomic_save_tensor(tensor_to_save, embeddings_tensor_file)
                self.log.info(f"Embeddings tensor for '{model_to_save['model_name']}' saved to {embeddings_tensor_file}")
                return True
            except IcotqCriticalError as e:
                 self.log.error(f"Failed to save embeddings tensor for {model_to_save['model_name']} to {embeddings_tensor_file}: {e}")
                 raise
        else:
             raise IcotqError("Cannot save tensor: Model context lost unexpectedly.")  # pyright:ignore[reportUnreachable]


    def _load_tensor_internal(self, model_name: str, device_override: str | None = None, check_consistency: bool = True) -> bool: # Changed Optional
        """Loads tensor, handles consistency checks. Returns True if loaded."""
        embeddings_tensor_file = self._get_tensor_path(model_name)
        target_device = self.resolve_device(device_override if device_override else self.config['embeddings_device'])
        map_location = torch.device(target_device)
        loaded_tensor: torch.Tensor | None = None # Changed Optional

        if os.path.exists(embeddings_tensor_file):
            try:
                loaded_tensor = torch.load(embeddings_tensor_file, map_location=map_location)  # pyright:ignore[reportUnknownMemberType]
                self.log.info(f"Loaded tensor for model '{model_name}' onto device '{target_device}'. Shape: {loaded_tensor.shape if loaded_tensor is not None else 'N/A'}")

                if check_consistency and loaded_tensor is not None:
                    expected_rows = sum(entry['emb_ptrs'][model_name][1] for entry in self.lib if model_name in entry.get('emb_ptrs', {}))
                    actual_rows = loaded_tensor.shape[0]
                    self.log.info(f"Consistency check for '{model_name}': Tensor rows={actual_rows}, Library expected rows={expected_rows}")
                    if actual_rows != expected_rows:
                        consistency_msg = f"Embeddings tensor '{model_name}' is INCONSISTENT with library! Tensor has {actual_rows} rows, library expects {expected_rows}."
                        raise IcotqConsistencyError(consistency_msg)

            except FileNotFoundError:
                 self.log.warning(f"Tensor file {embeddings_tensor_file} vanished before loading.")
                 loaded_tensor = None
            except (RuntimeError, EOFError, Exception) as e:
                 self.log.error(f"Failed to load or process tensor file {embeddings_tensor_file}: {e}", exc_info=True)
                 raise IcotqCriticalError(f"Corrupted or unreadable tensor file: {embeddings_tensor_file}") from e
        else:
            self.log.warning(f"No embeddings tensor file found for model '{model_name}' at {embeddings_tensor_file}. Use 'index' to generate.")
            loaded_tensor = None

        if self.current_model and self.current_model['model_name'] == model_name:
            self.embeddings_matrix = loaded_tensor
            if self.device != target_device:
                self.log.info(f"Updating active device context to '{target_device}'")
                self.device = target_device
                if self.engine:
                    self.engine = self.engine.to(map_location)

        return loaded_tensor is not None


    def _save_config_internal(self):
        """Saves config atomically. Assumes lock is held."""
        try:
            self._atomic_save_json(self.config, self.config_file)
            self.log.info(f"Configuration changes saved to {self.config_file}")
        except IcotqCriticalError as e:
             raise IcotqCriticalError(f"Failed to save configuration: {e}") from e

    # === PDF Handling (Internal, Assumes Lock Held) ===

    def _get_pdf_text_internal(self, desc:str, full_path:str) -> tuple[str | None, bool]: # Changed Optional
        """
        Gets PDF text, using cache if possible.
        Attempts pymupdf first, then pymupdf4llm (if available) as fallback.
        Assumes lock is held.
        """
        text: str | None = None
        changed: bool = False # Indicates if pdf_index was modified

        # Check cache validity
        if desc in self.pdf_index:
            cached_info = self.pdf_index[desc]
            try:
                cur_file_size = os.path.getsize(full_path)

                # Condition to skip extraction: Size matches AND it was a known previous failure
                if (cur_file_size == cached_info['file_size'] and
                        cached_info['previous_failure']):
                    self.log.debug(f"Skipping PDF {desc}: Size matches and previously failed extraction.")
                    return None, False # Return None, index not changed (still marked as failure)

                # Condition to use cache: Size matches AND it was *not* a previous failure AND cache file exists
                elif (cur_file_size == cached_info['file_size'] and
                      not cached_info['previous_failure'] and
                      cached_info.get('filename')):

                    basename = os.path.basename(cached_info['filename'])
                    local_path = os.path.join(self.pdf_cache_path, basename)

                    if os.path.exists(local_path):
                        try:
                            with open(local_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            # self.log.debug(f"Read PDF cache for {desc}")
                            return text, False # Return cached text, index not changed
                        except Exception as e:
                            self.log.warning(f"Failed to read PDF cache file {local_path} for {desc}: {e}. Re-extracting.")
                            text = None # Proceed to re-extract
                    else:
                         self.log.warning(f"PDF cache index points to non-existent file {local_path} for {desc}. Re-extracting.")
                         text = None # Proceed to re-extract

                elif cur_file_size != cached_info['file_size']:
                    self.log.info(f"PDF file size changed for {desc}, re-importing text.")
                    text = None # Force re-extraction, ignore previous_failure status

                else: # Size matches, not failed, but filename missing? Inconsistent.
                    self.log.warning(f"Inconsistent PDF cache state for {desc}. Re-extracting.")
                    text = None # Proceed to re-extract

            except FileNotFoundError:
                self.log.warning(f"Original PDF file {full_path} not found while checking cache for {desc}. Cannot get text.")
                if desc in self.pdf_index: 
                    del self.pdf_index[desc]
                    changed = True
                return None, changed
            except Exception as e:
                self.log.error(f"Error accessing file {full_path} or its cache info for {desc}: {e}. Cannot get text.")
                return None, False # Don't change index on access error

        # If text is still None (cache miss, size change, or read failure), attempt extraction
        if text is None:
            extracted_text: str | None = None
            extraction_method: str = "None" # Track which method succeeded

            # --- Attempt 1: pymupdf ---
            self.log.debug(f"Attempting PDF extraction for {desc} using pymupdf...")
            try:
                doc = pymupdf.open(full_path)
                extracted_pages = []
                for page_num, page in enumerate(doc):  # pyright:ignore[reportArgumentType, reportUnknownVariableType]
                    try:
                        page_text = page.get_text()  # pyright:ignore[reportUnknownMemberType, reportUnknownVariableType]
                        if isinstance(page_text, str): 
                            extracted_pages.append(page_text)  # pyright: ignore[reportUnknownMemberType]
                        else: 
                            self.log.warning(f"Non-string text on page {page_num+1} of {full_path}.")
                    except Exception as page_e: 
                        self.log.warning(f"Failed extraction on page {page_num+1} of {full_path}: {page_e}")
                doc.close()

                if extracted_pages:
                    combined_text = "\n".join(extracted_pages)  # pyright:ignore[reportUnknownArgumentType]
                    if combined_text.strip(): # Check if not just whitespace
                        extracted_text = combined_text
                        extraction_method = "pymupdf"
                        self.log.info(f"Successfully extracted text from {desc} using pymupdf.")
                    else:
                        self.log.info(f"Extracted only whitespace from {desc} using pymupdf.")
                else:
                    self.log.info(f"No text could be extracted from {desc} using pymupdf.")

            except FileNotFoundError:
                 self.log.error(f"PDF file {full_path} not found during pymupdf extraction for {desc}.")
                 if desc in self.pdf_index:
                    del self.pdf_index[desc]
                    changed = True
                 return None, changed # Abort extraction for this file
            except Exception as e:
                self.log.error(f"Failed pymupdf extraction for {desc}: {e}", exc_info=True)
                # Don't set extracted_text, proceed to fallback if available

            # --- Attempt 2: pymupdf4llm (Fallback) ---
            if extracted_text is None and PYMUPDF4LLM_AVAILABLE:
                self.log.info(f"pymupdf failed for {desc}, attempting fallback using pymupdf4llm...")
                try:
                    # Use pymupdf4llm to convert the document to markdown
                    # Requires pymupdf4llm to be installed: pip install pymupdf4llm
                    md_text = pymupdf4llm.to_markdown(full_path)  # pyright:ignore[reportPossiblyUnboundVariable, reportUnknownMemberType]

                    if md_text and md_text.strip():
                        extracted_text = md_text # Use the markdown text
                        extraction_method = "pymupdf4llm"
                        self.log.info(f"Successfully extracted markdown text from {desc} using pymupdf4llm.")
                    else:
                        self.log.info(f"pymupdf4llm returned empty/whitespace text for {desc}.")

                except FileNotFoundError:
                    # Should have been caught by pymupdf attempt, but handle defensively
                    self.log.error(f"PDF file {full_path} not found during pymupdf4llm fallback for {desc}.")
                    if desc in self.pdf_index:
                        del self.pdf_index[desc]
                        changed = True
                    return None, changed # Abort extraction
                except Exception as e:
                    self.log.error(f"Failed pymupdf4llm fallback extraction for {desc}: {e}", exc_info=True)
                    # extracted_text remains None

            # --- Process Extraction Result ---
            final_failure = (extracted_text is None)
            cache_filename = ""
            temp_pdf_path: str | None = None # For temp file in atomic save

            if not final_failure:
                 cache_filename = str(uuid.uuid4()) + ".txt" # Use .txt even for markdown for simplicity

            # Get current file size for the index entry
            current_size = -1
            try:
                if os.path.exists(full_path):
                    current_size = os.path.getsize(full_path)
            except OSError as e:
                 self.log.warning(f"Could not get size of {full_path} for index: {e}")


            new_pdf_ind: PDFIndex = {
                'filename': cache_filename,
                'file_size': current_size,
                'previous_failure': final_failure # Mark failure only if *both* methods failed
            }

            # Write new cache file if extraction succeeded
            if not final_failure:
                cache_file_path = os.path.join(self.pdf_cache_path, cache_filename)
                temp_fd_pdf, temp_pdf_path = None, None
                try:
                    # Atomically save the extracted text
                    temp_fd_pdf, temp_pdf_path = tempfile.mkstemp(dir=self.pdf_cache_path, prefix=cache_filename + '.tmp')
                    with os.fdopen(temp_fd_pdf, 'w', encoding='utf-8') as f:
                        _ = f.write(extracted_text)
                    os.replace(temp_pdf_path, cache_file_path)
                    temp_pdf_path = None # Prevent removal in finally
                    self.log.info(f"Added/Updated {desc} in PDF cache ({cache_filename}) using {extraction_method}.")
                    text = extracted_text # Set return text
                except (IOError, OSError) as e:
                     self.log.error(f"Failed to write PDF cache file {cache_file_path} for {desc}: {e}. Extraction result lost.")
                     # Revert state to failure if write failed
                     new_pdf_ind['previous_failure'] = True
                     new_pdf_ind['filename'] = ""
                     text = None
                except Exception as e:
                    self.log.error(f"Unexpected error writing PDF cache file {cache_file_path} for {desc}: {e}", exc_info=True)
                    new_pdf_ind['previous_failure'] = True
                    new_pdf_ind['filename'] = ""
                    text = None
                finally:
                     # Ensure temp file is cleaned up on error
                     if temp_pdf_path and os.path.exists(temp_pdf_path):
                         try: os.remove(temp_pdf_path)
                         except OSError as rm_e: self.log.error(f"Failed remove temp pdf cache {temp_pdf_path}: {rm_e}")

            # Remove old cache file if necessary
            old_filename = self.pdf_index.get(desc, {}).get('filename')
            if old_filename and old_filename != cache_filename:
                old_cache_path = os.path.join(self.pdf_cache_path, os.path.basename(old_filename))
                if os.path.exists(old_cache_path):
                    try: 
                        os.remove(old_cache_path)
                        self.log.debug(f"Removed old PDF cache file {old_cache_path}")
                    except OSError as e: 
                        self.log.warning(f"Failed to remove old PDF cache file {old_cache_path}: {e}")

            # Update the index entry, even if extraction failed (to record the failure status)
            self.pdf_index[desc] = new_pdf_ind
            changed = True # Index was modified

            if final_failure:
                self.log.warning(f"Both pymupdf and fallback failed to extract text from: {desc}")

        # Return the extracted text (if successful) and whether the index changed
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
        with self._lock:
            if not self.config['tq_sources']: 
                print("No valid sources configured.")
                return
            print("Configured Sources:")
            for i, source in enumerate(self.config['tq_sources']): 
                print(f"  {i:02d}: Name='{source['name']}', Type='{source['tqtype']}', Path='{source['path']}', Files={source['file_types']}")

    def get_status_info(self) -> tuple[int, str | None, list[str], list[TqSource]]:
        """
        Returns key status information in a thread-safe manner.

        Returns:
            tuple containing:
                - library_size (int)
                - current_model_name (str | None)
                - available_model_names (list[str])
                - sources (list[TqSource])
        """
        with self._lock: # Internal lock acquisition
            lib_size = len(self.lib)
            current_model_name = self.current_model['model_name'] if self.current_model else None
            available_model_names = [m['model_name'] for m in self.model_list]
            sources = self.config.get('tq_sources', []) # Read config safely under lock
            return lib_size, current_model_name, available_model_names, sources

    def get_available_model_info(self) -> list[ModelInfo]:
        """Returns a list of available models' info."""
        with self._lock:
            return [
                ModelInfo(model_hf_name=m['model_hf_name'], model_name=m['model_name'])
                for m in self.model_list
            ]

    def read_library(self):
        """Public method to reload library and PDF index from disk."""
        with self._modify():
            self._read_library_internal()
            if self.current_model:
                try:
                    _ = self._load_tensor_internal(self.current_model['model_name'])
                except IcotqError as e:
                    self.log.error(f"Error reloading tensor for current model after library reload: {e}")
                    self.embeddings_matrix = None


    def load_model(self, name: str, device:str="auto", trust_remote_code:bool=False) -> bool:
        """Loads an embeddings model and its corresponding tensor."""
        with self._modify():
            try:
                loaded_model_def = self._load_model_internal(name, device, trust_remote_code)
                if loaded_model_def:
                     try:
                         _ = self._load_tensor_internal(loaded_model_def['model_name'])
                     except IcotqConsistencyError as e:
                         self.log.error(f"Consistency Error loading tensor for new model '{name}': {e}")
                         if self.config.get('auto_fix_inconsistency', False):
                             self.log.warning(f"Attempting automatic fix (re-index) for model '{name}'.")
                             try:
                                 self._generate_embeddings_internal(purge=True, model_name_override=name)
                                 self.log.info(f"Automatic re-index for '{name}' completed.")
                                 _ = self._load_tensor_internal(name)
                             except IcotqError as fix_e:
                                 self.log.error(f"Automatic fix failed for model '{name}': {fix_e}")
                                 self.embeddings_matrix = None
                                 self.log.critical(f"Model '{name}' loaded, but embeddings are inconsistent and could not be fixed automatically. Manual 'index purge' required.")
                         else:
                             self.log.warning("Automatic fixing is disabled. Embeddings for this model are inconsistent. Manual 'index purge' may be required.")
                             self.embeddings_matrix = None
                     except IcotqCriticalError as e:
                          self.log.error(f"Critical Error loading tensor for model '{name}': {e}")
                          self.embeddings_matrix = None
                          self.log.critical(f"Model '{name}' loaded, but its embeddings tensor is corrupt or unreadable. Indexing needed.")
                     except IcotqError as e:
                          self.log.error(f"Error loading tensor for model '{name}': {e}")
                          self.embeddings_matrix = None

                     self.config['embeddings_model_name'] = name
                     self.config['embeddings_device'] = device
                     self.config['embeddings_model_trust_code'] = trust_remote_code
                     self._save_config_internal()
                     return True
                else:
                    return False

            except IcotqError as e:
                self.log.error(f"Failed to load model '{name}': {e}")
                self.engine = None
                self.current_model = None
                self.embeddings_matrix = None
                self.device = None
                return False
            except Exception as e:
                 self.log.critical(f"Unexpected critical error loading model '{name}': {e}", exc_info=True)
                 self.engine = None
                 self.current_model = None
                 self.embeddings_matrix = None
                 self.device = None
                 raise IcotqCriticalError(f"Unexpected failure loading model {name}") from e


    def _load_model_internal(self, name: str, device_str:str="auto", trust_remote_code:bool=False) -> EmbeddingsModel | None: # Changed Optional
        """Internal model loading logic. Assumes lock is held."""
        self.log.info(f"Attempting to load model '{name}'...")
        selected_model: EmbeddingsModel | None = None # Changed Optional
        for model in self.model_list:
            if model.get('model_hf_name') == name or model.get('model_name') == name:
                selected_model = model
                break

        if not selected_model:
            raise IcotqConfigurationError(f"Model '{name}' is unknown, not found in model_list.json")

        hf_name = selected_model['model_hf_name']
        resolved_device = self.resolve_device(device_str)
        target_torch_device = torch.device(resolved_device)

        # --- Store reference to old tensor before clearing ---
        old_tensor = self.embeddings_matrix
        old_tensor_device = old_tensor.device if old_tensor is not None else None

        try:
            # --- Load New Engine ---
            engine = SentenceTransformer(hf_name, trust_remote_code=trust_remote_code)
            engine = engine.to(target_torch_device)

            # --- Update State (Engine, Device, Current Model) ---
            self.engine = engine
            self.device = resolved_device
            self.current_model = selected_model
            # --- Clear old tensor reference *after* new engine loaded ---
            self.embeddings_matrix = None

            self.log.info(f"Model '{name}' ({hf_name}) loaded successfully onto device '{resolved_device}'.")

            # --- START: Explicitly Clear Old Tensor Memory ---
            if old_tensor is not None:
                self.log.info(f"Clearing previous model's embedding tensor from memory (Device: {old_tensor_device})...")
                del old_tensor # Remove reference
                # Try to clear the cache on the specific device the old tensor was on
                if old_tensor_device and old_tensor_device.type == 'cuda':
                    try:
                        # target specific device if possible, otherwise general
                        # cuda_device_idx = old_tensor_device.index if old_tensor_device.index is not None else torch.cuda.current_device()
                        # with torch.cuda.device(cuda_device_idx): # Less reliable if context changes
                        torch.cuda.empty_cache()
                        self.log.debug("Cleared CUDA cache.")
                    except Exception as cache_e:
                        self.log.warning(f"Failed attempt to clear CUDA cache: {cache_e}")
                elif old_tensor_device and old_tensor_device.type == 'mps':
                     try:
                         torch.mps.empty_cache() # type: ignore # Requires specific torch version/build
                         self.log.debug("Cleared MPS cache.")
                     except AttributeError:
                          self.log.warning("torch.mps.empty_cache() not available in this PyTorch version.")
                     except Exception as cache_e:
                          self.log.warning(f"Failed attempt to clear MPS cache: {cache_e}")
            # --- END: Explicitly Clear Old Tensor Memory ---

            return selected_model # Return successfully loaded model info

        except Exception as e:
            # If loading the new engine fails, restore the old tensor reference if it existed
            # to maintain previous state as much as possible.
            self.log.error(f"Failed to load or initialize model '{name}' ({hf_name}): {e}", exc_info=True)
            self.embeddings_matrix = old_tensor # Restore old tensor if load failed (its not unbound!) # pyright: ignore[reportPossiblyUnboundVariable]
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
            processed_desc_paths: set[str] = set()

            # --- Initialize tensor_debris before Pass 1 ---
            tensor_debris: dict[str, list[tuple[int, int]]] = {} # model_name -> list[(start, len)]

            # --- Pass 1: Scan sources and update/add to in-memory library ---
            self.log.debug(f"Sync Pass 1: Scanning sources, initial library size: {initial_lib_size}...")
            lib_map = {entry['desc_filename']: entry for entry in self.lib}
            new_lib: list[LibEntry] = []
            abort_scan = False
            current_imports = 0

            for source in self.config['tq_sources']:
                if abort_scan: 
                    break
                source_path = source['path']
                self.log.info(f"Scanning source '{source['name']}' at '{source_path}'...")

                for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: self.log.warning(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                    if abort_scan: 
                        break
                    for filename in files:
                        if abort_scan: 
                            break
                        current_imports +=1

                        base, ext_with_dot = os.path.splitext(filename)
                        ext = ext_with_dot[1:].lower() if ext_with_dot else ""
                        if ext not in source['file_types']: 
                            continue

                        preferred_ext_exists = False
                        preferred_order = ['txt', 'md']
                        if ext == 'pdf':
                             for pref_ext in preferred_order:
                                 if os.path.exists(os.path.join(root, base + '.' + pref_ext)):
                                     preferred_ext_exists = True
                                     break
                        if preferred_ext_exists: continue

                        full_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(full_path, source_path)
                        desc_path = "{" + source['name'] + "}" + relative_path
                        processed_desc_paths.add(desc_path)

                        current_text: str | None = None
                        pdf_changed_during_get = False
                        try:
                            if ext in ['md', 'txt']:
                                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f: current_text = f.read()
                            elif ext == 'pdf':
                                # PDF handling might add complexity, ensure it works if used
                                current_text, pdf_changed_during_get = self._get_pdf_text_internal(desc_path, full_path)
                                if pdf_changed_during_get: pdf_index_changed = True
                        except FileNotFoundError: 
                            self.log.warning(f"File vanished: {full_path}.")
                            continue
                        except Exception as e: 
                            self.log.error(f"Error reading {full_path}: {e}.")
                            continue

                        existing_entry = lib_map.get(desc_path)
                        if existing_entry:
                            # Update handling
                            needs_update = False
                            old_pointers_to_collect: dict[str, tuple[int, int]] = {}

                            if current_text is not None and existing_entry.get('text') != current_text:
                                self.log.info(f"Updating text for {desc_path}")
                                needs_update = True
                                old_pointers_to_collect = existing_entry.get('emb_ptrs', {}).copy()
                                existing_entry['text'] = current_text
                            elif current_text is None: #  and existing_entry.get('text') is not None:
                                self.log.warning(f"Text unreadable for {desc_path}. Clearing.")
                                needs_update = True
                                old_pointers_to_collect = existing_entry.get('emb_ptrs', {}).copy()
                                existing_entry['text'] = ""

                            if needs_update:
                                existing_entry['emb_ptrs'] = {} # Clear pointers *after* copying
                                lib_changed = True
                                # --- Collect debris immediately ---
                                if old_pointers_to_collect:
                                     self.log.info(f"DEBUG Pass 1 Update: Collecting pointers {old_pointers_to_collect} for {desc_path}")
                                     for model_name, ptr_info in old_pointers_to_collect.items():
                                         if model_name not in tensor_debris: 
                                            tensor_debris[model_name] = []
                                         # Avoid adding duplicates if somehow processed twice
                                         if ptr_info not in tensor_debris[model_name]:
                                             tensor_debris[model_name].append(ptr_info)

                            if existing_entry['filename'] != full_path:
                                existing_entry['filename'] = full_path
                                lib_changed = True

                            new_lib.append(existing_entry) # Add (potentially modified) entry to new list

                        elif current_text is not None:
                            # Add handling
                            self.log.info(f"Adding new entry for {desc_path}")
                            entry: LibEntry = LibEntry({'source_name': source['name'], 'desc_filename': desc_path, 'filename': full_path, 'text': current_text, 'emb_ptrs': {}})
                            new_lib.append(entry)
                            lib_changed = True
                        else:
                            self.log.debug(f"Skipping add for new file {desc_path} as text is not available.")

            if max_imports is not None and len(new_lib) > max_imports:
                 self.log.warning(f"Library size ({len(new_lib)}) exceeds max_imports ({max_imports}). Pruning...")
                 entries_to_prune = new_lib[max_imports:]
                 new_lib = new_lib[:max_imports]
                 for entry in entries_to_prune: 
                    processed_desc_paths.discard(entry['desc_filename'])
                 lib_changed = True

            # --- Pass 2: Identify Debris (Simplified: Only Deletions/Pruning now) ---
            self.log.debug("Sync Pass 2: Identifying deleted/pruned entries...")
            debris_lib_entries: list[LibEntry] = []
            debris_pdf_indices: list[str] = []
            # tensor_debris already initialized and populated for updates in Pass 1.

            original_lib_descs = set(lib_map.keys())
            current_lib_descs = {entry['desc_filename'] for entry in new_lib}
            deleted_or_pruned_descs = original_lib_descs - current_lib_descs

            for desc_path in deleted_or_pruned_descs:
                entry = lib_map[desc_path] # Get original entry state
                self.log.info(f"Detected deleted/pruned library entry: {desc_path}")
                debris_lib_entries.append(entry) # Keep track for potential PDF cache cleanup
                # Collect pointers for deletion cleanup
                for model_name, ptr_info in entry.get('emb_ptrs', {}).items():
                     if model_name not in tensor_debris: 
                        tensor_debris[model_name] = []
                     # Avoid adding duplicates if it was updated then deleted
                     if ptr_info not in tensor_debris[model_name]:
                          self.log.info(f"DEBUG Pass 2 Deletion: Collecting pointers {ptr_info} for {desc_path}")
                          tensor_debris[model_name].append(ptr_info)
                     else:
                          self.log.info(f"DEBUG Pass 2 Deletion: Pointers {ptr_info} for {desc_path} already collected (from prior update).")
                lib_changed = True

            # PDF index check (remains the same)
            all_current_lib_descs = {entry['desc_filename'] for entry in new_lib}
            for desc_path in list(self.pdf_index.keys()):
                 if desc_path not in all_current_lib_descs:
                     self.log.info(f"Detected orphaned PDF cache index entry: {desc_path}")
                     debris_pdf_indices.append(desc_path)
                     pdf_index_changed = True

            # --- Pass 3: Execute Cleanup ---
            self.log.debug("Sync Pass 3: Executing cleanup...")
            self.log.info(f"DEBUG: Entering Pass 3. tensor_debris keys: {list(tensor_debris.keys())}, debris_lib_entries: {len(debris_lib_entries)}, debris_pdf_indices: {len(debris_pdf_indices)}")
            if tensor_debris: 
                self.log.info(f"DEBUG: tensor_debris content: {tensor_debris}")

            if tensor_debris or debris_lib_entries or debris_pdf_indices: # Check if *any* cleanup needed
                # Make sure the self.lib = new_lib assignment happens *before* saving
                # and *after* pointer adjustments are applied to new_lib
                modified_tensors: dict[str, torch.Tensor] = {}
                new_pointer_maps: dict[str, dict[int, tuple[int, int]]] = {}
                all_models_processed_ok = True

                for model_name, removals in tensor_debris.items():
                    if not removals: continue
                    self.log.info(f"Calculating cleanup for tensor '{model_name}' ({len(removals)} chunks to remove).")
                    try:
                        # --- Load Tensor ---
                        temp_tensor_path = self._get_tensor_path(model_name)
                        if not os.path.exists(temp_tensor_path):
                            # ... (handling for missing tensor remains the same) ...
                            continue

                        temp_device = self.resolve_device(self.config['embeddings_device'])
                        original_tensor: torch.Tensor = torch.load(temp_tensor_path, map_location=torch.device(temp_device))  # pyright: ignore[reportUnknownMemberType]
                        num_rows = original_tensor.shape[0]

                        # --- Calculate Boolean Keep Mask (still needed for offset calculation) ---
                        keep_mask = torch.ones(num_rows, dtype=torch.bool, device=original_tensor.device)
                        self.log.info(f"DEBUG Cleanup {model_name}: Original rows={num_rows}, removing {len(removals)} chunk(s).") # Simplified log
                        removed_indices_count = 0
                        sorted_removals_for_mask = sorted(removals, key=lambda x: x[0], reverse=True)

                        for start, length in sorted_removals_for_mask:
                            if start < 0 or start + length > num_rows:
                                self.log.error(f"Invalid removal range [{start}:{start+length}]. Skipping.")
                                continue
                            # Check for overlap is less critical now, but warning is ok
                            if torch.any(keep_mask[start : start + length] == False):
                                self.log.warning(f"Overlapping removal detected near index {start}.")
                            keep_mask[start : start + length] = False
                            removed_indices_count += length

                        self.log.debug(f"DEBUG Cleanup {model_name}: Marked {removed_indices_count} rows for removal.")

                        # --- START: Efficient Tensor Creation using Integer Indexing ---
                        # Get the integer indices of rows to keep
                        keep_indices = torch.nonzero(keep_mask).squeeze(dim=1) # Squeeze to make it 1D

                        # Index the original tensor using the integer indices
                        # This is generally more memory-efficient than boolean mask indexing
                        modified_tensor = original_tensor.index_select(dim=0, index=keep_indices)
                        # Alternative (often equivalent): modified_tensor = original_tensor[keep_indices]
                        # index_select might sometimes be slightly more optimized.

                        # Free memory potentially held by the indices tensor if large (optional)
                        # del keep_indices
                        # --- END: Efficient Tensor Creation using Integer Indexing ---


                        self.log.info(f"DEBUG Cleanup {model_name}: Calculated modified tensor shape: {modified_tensor.shape}")
                        # Sanity check shape
                        expected_final_rows = num_rows - removed_indices_count
                        if modified_tensor.shape[0] != expected_final_rows:
                            self.log.error(f"DEBUG Cleanup {model_name}: MISMATCH between calculated shape {modified_tensor.shape[0]} and expected rows {expected_final_rows}!")
                            raise IcotqCriticalError(f"Tensor shape mismatch after indexing for {model_name}. Aborting cleanup.")
                        modified_tensors[model_name] = modified_tensor

                        # --- Calculate Pointer Adjustments (using keep_mask) ---
                        # This part remains the same as it needs the original indexing context
                        removed_lengths_cumsum = torch.cumsum(~keep_mask, dim=0)
                        current_model_pointers: dict[int, tuple[int,int]] = {}
                        # ... (rest of pointer adjustment logic using removed_lengths_cumsum remains the same) ...
                        for entry in new_lib: # Iterate over the *target* library state
                            if model_name in entry.get('emb_ptrs', {}):
                                old_start, old_length = entry['emb_ptrs'][model_name]
                                if old_start < 0 or old_start >= num_rows:
                                    self.log.error(f"Entry '{entry['desc_filename']}' invalid old start pointer {old_start}. Removing."); del entry['emb_ptrs'][model_name]; lib_changed = True; continue

                                # Check if this entry's original range survived (using keep_mask)
                                if not torch.all(keep_mask[old_start : old_start + old_length]):
                                     self.log.error(f"Logic Error: Entry '{entry['desc_filename']}' (in new_lib) points to rows [{old_start}:{old_start+old_length}] marked for removal. Removing pointer.")
                                     del entry['emb_ptrs'][model_name]
                                     lib_changed = True
                                     continue

                                offset: int = int(removed_lengths_cumsum[old_start - 1].item()) if old_start > 0 else 0
                                new_start = old_start - offset
                                if new_start < 0:
                                    self.log.critical(f"CRITICAL ERROR pointer adjustment for '{entry['desc_filename']}' model '{model_name}': New start negative ({new_start}).")
                                    raise IcotqCriticalError(f"Pointer adjustment failed critically for {model_name}.")
                                current_model_pointers[old_start] = (new_start, old_length)
                        new_pointer_maps[model_name] = current_model_pointers

                    except IcotqCriticalError: 
                        all_models_processed_ok = False
                        raise
                    except Exception as e: 
                        all_models_processed_ok = False
                        raise IcotqCriticalError(f"Failure during tensor processing for {model_name}.") from e

                # --- Commit Phase ---
                if all_models_processed_ok:
                    # Save modified tensors
                    for model_name, tensor_data in modified_tensors.items():
                        try:
                            self._atomic_save_tensor(tensor_data, self._get_tensor_path(model_name))
                            self.log.info(f"Atomically saved cleaned tensor for '{model_name}'. Shape: {tensor_data.shape}")
                        except IcotqCriticalError as e:
                            raise IcotqCriticalError(f"Failed to commit cleaned tensor for {model_name}.") from e

                    # Apply pointer adjustments to new_lib (before assigning to self.lib)
                    for model_name, ptr_map in new_pointer_maps.items():
                        for i, entry in enumerate(new_lib):
                             if model_name in entry.get('emb_ptrs', {}):
                                 old_start = entry['emb_ptrs'][model_name][0]
                                 if old_start in ptr_map:
                                     new_ptr_tuple = ptr_map[old_start]
                                     if new_lib[i]['emb_ptrs'][model_name] != new_ptr_tuple:
                                         new_lib[i]['emb_ptrs'][model_name] = new_ptr_tuple
                                         lib_changed = True
                                 else:
                                     # This entry's original start wasn't in the map - should not happen if it wasn't deleted
                                     self.log.error(f"Pointer consistency error: Old start {old_start} for {entry['desc_filename']} model {model_name} not found in adjustment map after cleanup. Removing pointer.")
                                     del new_lib[i]['emb_ptrs'][model_name]
                                     lib_changed = True

                    # Update in-memory library *now*
                    self.lib = new_lib

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
                             pdf_index_changed = True

                    # Save final library and PDF index state
                    if lib_changed or pdf_index_changed:
                        self.log.info("Saving updated library and PDF index after cleanup...")
                        self._write_library_internal()

                    if self.current_model and self.current_model['model_name'] in modified_tensors:
                        current_model_name = self.current_model['model_name']
                        self.log.info(f"Updating current model's ({current_model_name}) in-memory tensor after cleanup.")

                        # --- Assign directly instead of reloading ---
                        self.embeddings_matrix = modified_tensors[current_model_name]
                        # --- End Assignment ---

                        # Optional: Perform consistency check on the new in-memory matrix
                        try:
                            expected_rows = sum(entry['emb_ptrs'][current_model_name][1] for entry in self.lib if current_model_name in entry.get('emb_ptrs', {}))
                            actual_rows = self.embeddings_matrix.shape[0]
                            if actual_rows != expected_rows:
                                # This would be a critical logic error if it happens here
                                consistency_msg = f"CRITICAL Inconsistency *after* direct assignment! Lib expects {expected_rows}, matrix has {actual_rows}."
                                self.log.critical(consistency_msg)
                                self.embeddings_matrix = None # Safety
                                raise IcotqCriticalError(consistency_msg)
                            else:
                                self.log.info(f"In-memory tensor for '{current_model_name}' updated and passed consistency check ({actual_rows} rows).")
                        except Exception as check_e:
                             # Handle potential errors during the check itself
                             self.log.critical(f"CRITICAL Error during post-assignment consistency check for {current_model_name}: {check_e}")
                             self.embeddings_matrix = None # Safety
                             # Re-raise as critical? Or just log and clear matrix? Re-raise seems safer.
                             raise IcotqCriticalError(f"Post-assignment check failed for {current_model_name}") from check_e

                    # Reload current tensor if modified
                    # if self.current_model and self.current_model['model_name'] in modified_tensors:
                    #      self.log.info(f"Reloading current model's ({self.current_model['model_name']}) tensor after cleanup.")
                    #      try:
                    #          # Check consistency *after* reload
                    #          _ = self._load_tensor_internal(self.current_model['model_name'], check_consistency=True)
                    #      except (IcotqConsistencyError, IcotqCriticalError) as e:
                    #          # This is where the previous crash happened - need to ensure it passes now
                    #          self.log.critical(f"CRITICAL: Consistency error *after* cleanup reload for {self.current_model['model_name']}: {e}")
                    #          self.embeddings_matrix = None # Safety
                    #          raise IcotqCriticalError("Consistency check failed immediately after cleanup. Logic error suspected.") from e

                    self.log.info(f"Synchronization and cleanup completed. Library size: {len(self.lib)}")
                else:
                     self.log.error("Sync aborted due to errors during tensor processing.")

            elif lib_changed or pdf_index_changed:
                # Only library/pdf index changed (adds/simple updates), no tensor cleanup needed
                # Assign new_lib to self.lib before saving
                self.lib = new_lib
                self.log.info("Saving updated library and PDF index (no tensor cleanup required)...")
                self._write_library_internal()
                self.log.info(f"Synchronization completed. Library size: {len(self.lib)}")
            else:
                self.log.info("No changes detected during synchronization.")


    def generate_embeddings(self, save_every_sec: int = 180, purge: bool = False, model_name_override: str | None = None): # Changed Optional
        """Generates embeddings. Uses atomic saves."""
        with self._modify():
            self._generate_embeddings_internal(save_every_sec, purge, model_name_override)

    def _generate_embeddings_internal(self, save_every_sec: int = 180, purge: bool = False, model_name_override: str | None = None): # Changed Optional
        """Internal implementation of generate_embeddings. Assumes lock is held."""
        model_to_use: EmbeddingsModel | None = None # Changed Optional
        local_engine: SentenceTransformer | None = None # Changed Optional
        local_embeddings_matrix: torch.Tensor | None = None # Changed Optional
        target_model_name: str = ""

        if model_name_override:
            found_model = next((m for m in self.model_list if m['model_name'] == model_name_override), None)
            if not found_model: 
                raise IcotqConfigurationError(f"Overridden model name '{model_name_override}' not found.")
            model_to_use = found_model
            target_model_name = model_to_use['model_name']
            self.log.info(f"Generating embeddings specifically for model: '{target_model_name}'")
            if not self.current_model or self.current_model['model_name'] != target_model_name:
                 self.log.warning(f"Temporarily loading model '{target_model_name}'.")
                 try:
                     temp_engine = SentenceTransformer(model_to_use['model_hf_name'], trust_remote_code=self.config.get('embeddings_model_trust_code', True))
                     temp_device = self.resolve_device(self.config['embeddings_device'])
                     local_engine = temp_engine.to(torch.device(temp_device))
                 except Exception as e: 
                    raise IcotqError(f"Failed to temporarily load model '{target_model_name}': {e}") from e
            else: 
                local_engine = self.engine

            try:
                 _ = self._load_tensor_internal(target_model_name, check_consistency=False)
                 if self.current_model and self.current_model['model_name'] == target_model_name: 
                    local_embeddings_matrix = self.embeddings_matrix
                 else:
                    temp_tensor_path = self._get_tensor_path(target_model_name)
                    if os.path.exists(temp_tensor_path):
                        temp_device = self.resolve_device(self.config['embeddings_device'])
                        local_embeddings_matrix = torch.load(temp_tensor_path, map_location=torch.device(temp_device))  # pyright:ignore[reportUnknownMemberType]
                    else: 
                        local_embeddings_matrix = None
            except IcotqError as e: 
                self.log.warning(f"Could not load existing tensor for '{target_model_name}': {e}.")
                local_embeddings_matrix = None

        elif self.current_model and self.engine:
            model_to_use = self.current_model
            target_model_name = model_to_use['model_name']
            local_engine = self.engine
            local_embeddings_matrix = self.embeddings_matrix
            self.log.info(f"Generating embeddings for current model: '{target_model_name}'")
        else:
            raise IcotqError("Cannot generate embeddings: No current embeddings model loaded.")

        if purge:
            self.log.warning(f"Purging existing embeddings for model '{target_model_name}'.")
            local_embeddings_matrix = None
            for i in range(len(self.lib)):
                if target_model_name in self.lib[i].get('emb_ptrs', {}): 
                    del self.lib[i]['emb_ptrs'][target_model_name]
            try: 
                self._write_library_internal()
                self.log.info(f"Library saved after purging pointers for {target_model_name}.")
            except IcotqCriticalError as e: 
                raise IcotqCriticalError("Failed to save library after purging pointers.") from e

        # start_time = time.time()
        last_save_time = time.time()
        lib_changed_since_last_save = False
        tensor_changed_since_last_save = False
        total_entries = len(self.lib)
        processed_count = 0

        for ind, entry in enumerate(self.lib):
            print(f"\rEmbedding Progress ({target_model_name}): {ind+1}/{total_entries} ({processed_count} processed)", end="", flush=True)
            if 'emb_ptrs' not in entry: 
                self.lib[ind]['emb_ptrs'] = {}
                lib_changed_since_last_save = True
            if target_model_name in entry['emb_ptrs'] and not purge: 
                continue

            processed_count += 1
            text_to_embed = entry.get('text')
            if not text_to_embed:
                 if target_model_name in self.lib[ind]['emb_ptrs']: 
                    del self.lib[ind]['emb_ptrs'][target_model_name]
                    lib_changed_since_last_save = True
                 continue

            try:
                text_chunks = self.get_chunks(text_to_embed, model_to_use['chunk_size'], model_to_use['chunk_overlap'])
                if not text_chunks:
                     if target_model_name in self.lib[ind]['emb_ptrs']: 
                        del self.lib[ind]['emb_ptrs'][target_model_name]
                        lib_changed_since_last_save = True
                     continue

                embeddings: list[torch.Tensor] = local_engine.encode(  # pyright:ignore[reportUnknownMemberType, reportOptionalMemberAccess]
                    sentences=text_chunks, show_progress_bar=False, convert_to_numpy=False, batch_size=32
                 )

                if not embeddings:
                     self.log.warning(f"Encoding produced no embeddings for {entry['desc_filename']}")
                     if target_model_name in self.lib[ind]['emb_ptrs']: 
                        del self.lib[ind]['emb_ptrs'][target_model_name]
                        lib_changed_since_last_save = True
                     continue

                emb_matrix_chunk = torch.stack(embeddings).to(local_embeddings_matrix.device if local_embeddings_matrix is not None else torch.device(self.resolve_device(self.config['embeddings_device'])))

                if local_embeddings_matrix is None: 
                    start_ptr = 0
                    local_embeddings_matrix = emb_matrix_chunk
                else: 
                    start_ptr = local_embeddings_matrix.shape[0]
                    local_embeddings_matrix = torch.cat([local_embeddings_matrix, emb_matrix_chunk])

                emb_len = emb_matrix_chunk.shape[0]
                del emb_matrix_chunk
                self.lib[ind]['emb_ptrs'][target_model_name] = (start_ptr, emb_len)
                lib_changed_since_last_save = True
                tensor_changed_since_last_save = True

            except Exception as e:
                 self.log.error(f"\nFailed to generate embeddings for {entry['desc_filename']}: {e}", exc_info=True)
                 if target_model_name in self.lib[ind]['emb_ptrs']: 
                    del self.lib[ind]['emb_ptrs'][target_model_name]
                    lib_changed_since_last_save = True

            current_time = time.time()
            if save_every_sec > 0 and (current_time - last_save_time > save_every_sec):
                print(f"\nPerforming periodic save ({target_model_name})...", end="", flush=True)
                try:
                    if lib_changed_since_last_save: 
                        self._write_library_internal()
                        lib_changed_since_last_save = False
                    if tensor_changed_since_last_save:
                        tensor_path = self._get_tensor_path(target_model_name)
                        self._atomic_save_tensor(local_embeddings_matrix, tensor_path)
                        tensor_changed_since_last_save = False
                    last_save_time = current_time
                    print(" Done.")
                except IcotqCriticalError as e: 
                    print("\nCRITICAL ERROR during periodic save.")
                    raise IcotqCriticalError("Aborting due to periodic save failure.") from e
                except Exception as e: 
                    print(f"\nUNEXPECTED ERROR during periodic save: {e}.")
                    raise IcotqCriticalError("Aborting due to unexpected periodic save failure.") from e

        print(f"\nFinalizing embedding generation ({target_model_name})...", end="", flush=True)
        try:
            if lib_changed_since_last_save: 
                self._write_library_internal()
            if tensor_changed_since_last_save: 
                self._atomic_save_tensor(local_embeddings_matrix, self._get_tensor_path(target_model_name))
            print(" Done.")
            self.log.info(f"Embedding generation for '{target_model_name}' completed. Processed {processed_count} entries.")
            if self.current_model and self.current_model['model_name'] == target_model_name:
                self.embeddings_matrix = local_embeddings_matrix
                self.log.info(f"Current model's ({target_model_name}) in-memory tensor updated.")
        except IcotqCriticalError as e: 
            print("\nCRITICAL ERROR during final save.")
            raise
        except Exception as e: 
            print(f"\nUNEXPECTED ERROR during final save: {e}.")
            raise IcotqCriticalError("Unexpected final save failure.") from e

    def check_clean(self, dry_run: bool = True):
        """Checks for inconsistencies. Attempts fixes if dry_run is False."""
        action = "Checking" if dry_run else "Checking and Cleaning"
        self.log.info(f"{action} IcoTqStore state...")
        with self._modify():
            issues_found = False
            initial_issues_found = False # Track if issues existed before fixes
            require_reindex: set[str] = set()

            # --- 1. PDF Cache Index vs. Library ---
            self.log.debug("Checking PDF cache index against library...")
            pdf_index_debris = [desc for desc in self.pdf_index if not any(entry['desc_filename'] == desc for entry in self.lib)]
            if pdf_index_debris:
                self.log.warning(f"Found {len(pdf_index_debris)} orphaned PDF cache index entries.")
                if not issues_found: initial_issues_found = True # Record first issue
                issues_found = True
            if not dry_run and pdf_index_debris:
                self.log.info("Removing orphaned PDF index entries...")
                pdf_index_changed = False
                for desc in pdf_index_debris:
                    if desc in self.pdf_index:
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
                    try: 
                        self._write_library_internal()
                        self.log.info("Saved state after cleaning PDF index.")
                    except IcotqCriticalError as e: 
                        self.log.error(f"Failed to save state after cleaning PDF index: {e}")

            # --- 2. PDF Cache Files vs. Index ---
            self.log.debug("Checking PDF cache files against index...")
            pdf_cache_file_debris: list[str] = [] # Changed from List
            pdf_index_filenames = {idx['filename'] for idx in self.pdf_index.values() if idx.get('filename')}
            try:
                cache_files = [f for f in os.listdir(self.pdf_cache_path) if os.path.isfile(os.path.join(self.pdf_cache_path, f))]
                for filename in cache_files:
                    if filename.endswith('.json'): continue
                    basename = os.path.basename(filename)
                    if basename.endswith('.tmp'): 
                        self.log.warning(f"Found leftover temporary file: {filename}")
                        pdf_cache_file_debris.append(filename)
                        issues_found = True
                    elif basename not in pdf_index_filenames: 
                        pdf_cache_file_debris.append(filename)
                        issues_found = True
            except FileNotFoundError:
                self.log.warning(f"PDF Cache directory not found at {self.pdf_cache_path}.")
            except OSError as e:
                self.log.error(f"Error listing PDF cache directory {self.pdf_cache_path}: {e}")

            if pdf_cache_file_debris:
                self.log.warning(f"Found {len(pdf_cache_file_debris)} orphaned or temporary PDF cache files.")
                if not issues_found: initial_issues_found = True
                issues_found = True
            if not dry_run and pdf_cache_file_debris:
                self.log.info("Removing orphaned/temporary PDF cache files...")
                for filename in pdf_cache_file_debris:
                    filepath = os.path.join(self.pdf_cache_path, filename)
                    try: 
                        os.remove(filepath)
                        self.log.debug(f"Removed cache file: {filepath}")
                    except OSError as e:
                        self.log.warning(f"Failed to remove cache file {filepath}: {e}")

            # --- 3. Library Pointers and Tensor Consistency ---
            self.log.debug("Checking library pointers and tensor consistency...")
            all_known_models = {model['model_name'] for model in self.model_list}
            models_with_pointers: set[str] = set().union(*(entry.get('emb_ptrs', {}).keys() for entry in self.lib))
            models_to_check = all_known_models.intersection(models_with_pointers)
            if self.current_model and self.current_model['model_name'] not in models_to_check: 
                models_to_check.add(self.current_model['model_name'])

            for model_name in sorted(list(models_to_check)):
                self.log.debug(f"Checking consistency for model '{model_name}'...")
                tensor_path = self._get_tensor_path(model_name)
                tensor_exists = os.path.exists(tensor_path)
                expected_rows = 0
                has_pointers_in_lib = False
                pointer_issues_found = False
                max_index_reached = -1
                temp_fat_check: dict[int, int] = {}

                for entry in self.lib:
                    if model_name in entry.get('emb_ptrs', {}):
                        has_pointers_in_lib = True
                        start, length = entry['emb_ptrs'][model_name]
                        if start < 0 or length < 0:
                            self.log.error(f"Model '{model_name}': Invalid pointer format {entry['emb_ptrs'][model_name]} in '{entry['desc_filename']}'.")
                            pointer_issues_found = True
                            issues_found = True
                            require_reindex.add(model_name)
                            continue
                        if length == 0:
                            if entry.get('text'): 
                                self.log.warning(f"Model '{model_name}': Zero-length pointer for non-empty text in '{entry['desc_filename']}'.")
                                pointer_issues_found = True
                                issues_found = True
                                require_reindex.add(model_name)
                            continue
                        expected_rows += length
                        max_index_reached = max(max_index_reached, start + length -1)
                        for i in range(start, start + length): 
                            temp_fat_check[i] = temp_fat_check.get(i, 0) + 1

                overlaps = {i: count for i, count in temp_fat_check.items() if count > 1}
                gaps = {i for i in range(max_index_reached + 1) if i not in temp_fat_check} - set(overlaps.keys())
                if overlaps: 
                    self.log.error(f"Model '{model_name}': Pointer overlaps detected at indices: {list(overlaps.keys())[:10]}...")
                    pointer_issues_found = True
                    issues_found = True
                    require_reindex.add(model_name)
                if gaps: 
                    self.log.warning(f"Model '{model_name}': Pointer gaps detected between 0 and {max_index_reached} at indices: {list(gaps)[:10]}...")

                actual_rows = -1
                if tensor_exists:
                    try:
                        temp_tensor: torch.Tensor = torch.load(tensor_path, map_location='cpu')  # pyright:ignore[reportUnknownMemberType]
                        actual_rows = temp_tensor.shape[0]
                        del temp_tensor
                    except Exception as e: 
                        self.log.error(f"Model '{model_name}': Failed to load tensor {tensor_path}: {e}")
                        actual_rows = -999
                        issues_found = True
                        require_reindex.add(model_name)

                    if actual_rows != expected_rows: 
                        self.log.error(f"Model '{model_name}': Size inconsistency! Library={expected_rows}, tensor={actual_rows}.")
                        issues_found = True
                        require_reindex.add(model_name)
                    elif pointer_issues_found: 
                        self.log.warning(f"Model '{model_name}': Tensor size matches ({actual_rows}), but pointer errors exist.")
                    elif not has_pointers_in_lib and actual_rows > 0: 
                        self.log.warning(f"Model '{model_name}': Tensor exists ({actual_rows}) but no library pointers found.")
                        issues_found = True
                        require_reindex.add(model_name)
                    elif not pointer_issues_found and not gaps: 
                        self.log.info(f"Model '{model_name}': Consistent. Tensor size: {actual_rows}.")
                elif has_pointers_in_lib: 
                    self.log.error(f"Model '{model_name}': Missing tensor file! Library expects {expected_rows}.")
                    issues_found = True
                    require_reindex.add(model_name)

            # --- 4. Perform Fixes ---
            if not dry_run and require_reindex:
                self.log.warning(f"Attempting to fix inconsistencies by re-indexing models: {list(require_reindex)}")
                for model_name in require_reindex:
                    self.log.info(f"Running 'generate_embeddings(purge=True)' for model '{model_name}'...")
                    try: 
                        self._generate_embeddings_internal(purge=True, model_name_override=model_name)
                        self.log.info(f"Re-indexing for '{model_name}' completed.")
                    except IcotqError as e: 
                        self.log.error(f"Failed automatic re-index for '{model_name}': {e}.")
                        issues_found = True
                    except Exception as e: 
                        self.log.critical(f"Unexpected error during re-index of '{model_name}': {e}", exc_info=True)
                        issues_found = True
                if self.current_model and self.current_model['model_name'] in require_reindex:
                    self.log.info(f"Reloading current model '{self.current_model['model_name']}' tensor after re-index.")
                    try: 
                        _ = self._load_tensor_internal(self.current_model['model_name'], check_consistency=True)
                    except IcotqError as e: 
                        self.log.critical(f"CRITICAL: Failed reload/consistency check *after* re-index of {self.current_model['model_name']}: {e}")
                        self.embeddings_matrix = None

                    try:
                        # Minimal re-check after fix (just size consistency)
                        _ = self._load_tensor_internal(model_name, check_consistency=True)  # pyright:ignore[reportPossiblyUnboundVariable]
                        self.log.info(f"Post-fix check for '{model_name}' passed.")  # pyright:ignore[reportPossiblyUnboundVariable]
                        # If check passed, we might consider removing model_name from require_reindex
                        # But let's keep it simple, the final report logic handles persisting issues
                    except (IcotqConsistencyError, IcotqCriticalError) as post_fix_e:
                        if 'model_name' in locals():
                            self.log.error(f"Post-fix consistency check FAILED for '{model_name}': {post_fix_e}")  # pyright:ignore[reportPossiblyUnboundVariable]
                        else:
                            self.log.error(f"Post-fix consistency check FAILED (model name unknown): {post_fix_e}")
                        # Ensure issues_found remains True
                        issues_found = True
                    except Exception as post_fix_e:
                        if 'model_name' in locals():
                            self.log.error(f"Unexpected error during post-fix check for '{model_name}': {post_fix_e}")  # pyright:ignore[reportPossiblyUnboundVariable]
                        else:
                            self.log.error(f"Unexpected error during post-fix check for unknown model_name: {post_fix_e}")
                        issues_found = True

            # --- Final Report (using initial_issues_found and issues_found) ---
            if not initial_issues_found:
                self.log.info(f"{action} completed. No issues found.")
            elif dry_run:
                self.log.warning(f"{action} completed. Issues found (dry run). Run without '--dry-run' to attempt fixes.")
            # Check if issues *still* exist after fixes were attempted
            elif issues_found: # Use the flag reflecting state *after* potential fixes
                if require_reindex: # Check if the remaining issues involved reindexing
                    self.log.error(f"{action} completed. Automatic fixes ran but issues persist (check reindex models: {list(require_reindex)}). Manual intervention likely required.")
                else: # Only non-reindex issues remain (or fix failed)
                    self.log.error(f"{action} completed. Issues persist after attempted fixes (see logs).")
            else: # Initial issues found, but no issues remain after fixes
                self.log.info(f"{action} completed. Issues found and fixes applied successfully.")

    # === Search Functionality ===

    def _search_vect_internal(self, text: str) -> tuple[list[tuple[int, float]], torch.Tensor]: # Changed from List
        """Internal search vector generation. Assumes lock is held, engine/matrix valid."""
        if self.embeddings_matrix is None or self.engine is None: 
            raise IcotqError("Search prerequisites not met.")
        if str(self.engine.device) != str(self.embeddings_matrix.device):
             self.log.warning(f"Engine device ({self.engine.device}) differs from matrix device ({self.embeddings_matrix.device}). Moving engine.")
             try: 
                self.engine = self.engine.to(self.embeddings_matrix.device)
             except Exception as e: 
                raise IcotqError(f"Failed to move search engine: {e}") from e

        try: 
            vects: list[torch.Tensor] = self.engine.encode(sentences=[text], show_progress_bar=False, convert_to_numpy=False)  # pyright:ignore[reportAssignmentType, reportUnknownMemberType]
        except Exception as e: 
            raise IcotqError(f"Failed to generate search embedding: {e}") from e
        if not vects: raise IcotqError("Failed to calculate search embedding (empty result).")

        search_vect: torch.Tensor = vects[0];
        if len(vects) > 1: 
            self.log.warning("Search text generated multiple vectors, using only the first.")

        try:
            search_vect = search_vect.to(self.embeddings_matrix.device)
            similarities = torch.matmul(self.embeddings_matrix, search_vect)
            simil_list: list[float] = similarities.cpu().numpy().tolist()  # pyright: ignore[reportAssignmentType, reportUnknownMemberType]
        except Exception as e: 
            raise IcotqError(f"Error during similarity calculation: {e}") from e

        indexed_simil: list[tuple[int, float]] = list(enumerate(simil_list)) # Changed from List
        sorted_simil: list[tuple[int, float]] = sorted(indexed_simil, key=lambda x: x[1], reverse=True) # Changed from List
        return sorted_simil, search_vect.cpu()


    def _yellow_line_it_internal(self, text: str, search_embeddings_cpu: torch.Tensor, context_length: int, context_steps: int) -> np.typing.NDArray[np.float32]:
        """Internal yellow liner. Assumes lock is held, engine valid."""
        if self.engine is None: 
            raise IcotqError("Cannot perform yellow-lining: Engine not available.")
        if not text: return np.array([], dtype=np.float32)

        # target_device = self.engine.device # Just use the engine's current device

        clr: list[str] = [] # Changed from List
        text_len = len(text)
        for i in range(0, text_len, context_steps):
            i0 = max(0, i - context_length // 2)
            i1 = min(text_len, i + context_length // 2 + (context_length % 2))
            if i0 == 0 and i1 < text_len: 
                i1 = min(text_len, i0 + context_length)
            elif i1 == text_len and i0 > 0: 
                i0 = max(0, i1 - context_length)
            snippet = text[i0:i1];
            if snippet: 
                clr.append(snippet)
        if not clr: 
            clr = [text]

        try:
            snippet_embeddings: list[torch.Tensor] = self.engine.encode(sentences=clr, show_progress_bar=False, convert_to_numpy=False, batch_size=128)  # pyright:ignore[reportAssignmentType, reportUnknownMemberType]
            if not snippet_embeddings: 
                self.log.warning("Yellow-lining failed: no snippet embeddings.")
                return np.array([], dtype=np.float32)

            emb_matrix = torch.stack(snippet_embeddings)
            search_embeddings = search_embeddings_cpu.to(emb_matrix.device)
            yellow_vect: np.typing.NDArray[np.float32] = torch.matmul(emb_matrix, search_embeddings).cpu().numpy()  # pyright:ignore[reportUnknownMemberType]

            min_score: float
            max_score: float
            (min_score, max_score) = yellow_vect.min(), yellow_vect.max()
            if max_score > min_score: 
                yellow_vect = (yellow_vect - min_score) / (max_score - min_score)
            elif max_score > 0 : 
                yellow_vect.fill(1.0)
            else: 
                yellow_vect.fill(0.0)
            return yellow_vect
        except Exception as e: 
            self.log.error(f"Error during yellow-lining: {e}", exc_info=True)
            return np.array([], dtype=np.float32)

    def search(self, search_text:str, max_results:int=10, yellow_liner:bool=False, context_length:int=16, context_steps:int=4, compression_mode:str="none") -> list[SearchResult]: # Changed from List
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
            # --- Determine target device based on the main embeddings matrix ---
            target_device = self.embeddings_matrix.device

            try:
                # 1. Get Top-K Similarity Scores (ensures engine is on target_device)
                sorted_simil_all, search_embeddings_cpu = self._search_vect_internal(search_text)
                # Now self.engine is guaranteed to be on target_device if _search_vect_internal succeeded

                top_k_simil = sorted_simil_all[:max_results * 2] # Get more initially for potential merging

                idx_to_entry_map: dict[int, tuple[LibEntry, int]] = {}
                for entry in self.lib:
                     if target_model_name in entry.get('emb_ptrs', {}):
                         start, length = entry['emb_ptrs'][target_model_name]
                         if length > 0 and start >= 0:
                              for i in range(length): idx_to_entry_map[start + i] = (entry, i)

                resolved_list: list[tuple[str, int, float, LibEntry, int]] = [] # Changed from List
                processed_tensor_indices: set[int] = set()
                for tensor_idx, cosine in top_k_simil:
                    if tensor_idx in processed_tensor_indices: 
                        continue
                    processed_tensor_indices.add(tensor_idx)
                    if tensor_idx in idx_to_entry_map:
                         entry, offset = idx_to_entry_map[tensor_idx]
                         resolved_list.append((entry['desc_filename'], tensor_idx, cosine, entry, offset))
                    else: 
                        self.log.warning(f"Search index {tensor_idx} (score {cosine:.4f}) unmapped for model '{target_model_name}'.")

                resolved_list.sort(key=lambda x: (x[0], x[1]))
                merged_results: list[tuple[str, int, int, float, LibEntry]] = [] # Changed from List
                if not resolved_list: 
                    return []

                current_desc, current_idx, current_cos, current_entry, current_offset = resolved_list[0]
                current_count = 1
                max_cosine = current_cos
                for i in range(1, len(resolved_list)):
                    next_desc, next_idx, next_cos, next_entry, next_offset = resolved_list[i]
                    if (next_desc == current_desc and next_entry is current_entry and next_offset == current_offset + current_count):
                        current_count += 1
                        max_cosine = max(max_cosine, next_cos)
                    else:
                        merged_results.append((current_desc, current_idx - current_offset, current_count, max_cosine, current_entry))
                        current_desc, current_idx, current_cos, current_entry, current_offset = next_desc, next_idx, next_cos, next_entry, next_offset
                        current_count = 1
                        max_cosine = current_cos
                merged_results.append((current_desc, current_idx - current_offset, current_count, max_cosine, current_entry))

                merged_results.sort(key=lambda x: x[3], reverse=True)
                final_merged_list = merged_results[:max_results]

                # --- Ensure engine is on the correct device ONCE before the yellow-lining loop ---
                # (Redundant if _search_vect_internal already did it, but safe explicit check)
                # Compare torch.device objects directly
                if self.engine.device != target_device:
                     self.log.info(f"Ensuring search engine is on device {target_device} for yellow-lining.")
                     try:
                         self.engine = self.engine.to(target_device)
                     except Exception as e:
                          # If move fails here, we can't yellow-line
                          self.log.error(f"Failed to move engine to {target_device} for yellow-lining: {e}. Skipping highlighting.")
                          yellow_liner = False # Disable yellow-lining if move fails

                search_results_final: list[SearchResult] = [] # Changed from List
                for desc, start_tensor_idx, count, cosine, entry in final_merged_list:
                    entry_start_ptr, _ = entry['emb_ptrs'][target_model_name]
                    offset_in_entry = start_tensor_idx - entry_start_ptr
                    chunk_text: str = self.get_span_chunk(entry['text'], offset_in_entry, count, self.current_model['chunk_size'], self.current_model['chunk_overlap'])

                    if compression_mode == "light":
                        new_chunk = chunk_text.replace("\n", " ").replace("\r"," ").replace("\t", " ")
                        while "  " in new_chunk: new_chunk = new_chunk.replace("  ", " ")
                        chunk_text = new_chunk.strip()
                    elif compression_mode == "full":
                         new_chunk = chunk_text.replace("\n", " ").replace("\r"," ").replace("\t", " ")
                         while "  " in new_chunk: new_chunk = new_chunk.replace("  ", " ")
                         chunk_text = new_chunk.strip()

                    yellow_liner_weights: np.typing.NDArray[np.float32] | None = None # Changed Optional
                    if yellow_liner and chunk_text:
                         try:
                             # Pass the known correct target_device to avoid re-checking inside
                             yellow_liner_weights = self._yellow_line_it_internal(
                                 chunk_text,
                                 search_embeddings_cpu,
                                 context_length,
                                 context_steps,
                                 # Pass the correct device explicitly
                                 # expected_device=target_device
                             )
                         except IcotqError as yl_e:
                             self.log.error(f"Failed to generate yellow-liner for '{desc}': {yl_e}")

                    sres: SearchResult = {'cosine': cosine, 'index': start_tensor_idx, 'offset': offset_in_entry, 'desc': desc, 'chunk': chunk_text, 'text': entry['text'], 'yellow_liner': yellow_liner_weights}
                    search_results_final.append(sres)
                return search_results_final

            except IcotqError as e: 
                self.log.error(f"Search failed: {e}", exc_info=True)
                return []
            except Exception as e: 
                self.log.critical(f"Unexpected critical error during search: {e}", exc_info=True)
                return []


    # === Server Functionality ===

    async def search_handler(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handles incoming search requests via HTTP."""
        try:
            data: SearchRequest = await request.json()
            search_text = data.get('search_text')
            if not search_text: 
                return aiohttp.web.json_response({"error": "Missing 'search_text' field"}, status=400)
            # ... (input validation) ...
            max_results=data.get('max_results', 10)
            yellow_liner=data.get('yellow_liner', False)
            context_length=data.get('context_length', 16)
            context_steps=data.get('context_steps', 4)
            compression_mode=data.get('compression_mode', 'none')
            if max_results <= 0: 
                max_results = 10

            self.log.info(f"Received search request: text='{search_text[:50]}...', max={max_results}, yellow={yellow_liner}")
            search_results = self.search(search_text, max_results=max_results, yellow_liner=yellow_liner, context_length=context_length, context_steps=context_steps, compression_mode=compression_mode)
            self.log.info(f"Responding with {len(search_results)} results.")
            for result in search_results:
                if result.get('yellow_liner') is not None: 
                    result['yellow_liner'] = result['yellow_liner'].tolist()  # pyright: ignore[reportOptionalMemberAccess, reportGeneralTypeIssues]
            return aiohttp.web.json_response(search_results) # type: ignore

        except json.JSONDecodeError: 
            return aiohttp.web.json_response({"error": "Invalid JSON format"}, status=400)
        except Exception as e: 
            self.log.error(f"Error processing search request: {e}", exc_info=True)
            return aiohttp.web.json_response({"error": "Internal server error"}, status=500)


    def _server_task(self, host: str, port: int, in_thread: bool = False):
        """The asyncio server task runner."""
        # ... (logic remains the same, no typing changes needed here) ...
        self.log.info(f"Starting server task (in_thread={in_thread})...")
        loop = None
        try:
            app = aiohttp.web.Application()
            _ = app.router.add_post('/search', self.search_handler)
            runner = aiohttp.web.AppRunner(app)
            if in_thread: 
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                try: 
                    loop = asyncio.get_event_loop()
                except RuntimeError: 
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            self.loop = loop

            async def start_runner():
                await runner.setup()
                site = aiohttp.web.TCPSite(runner, host, port)
                await site.start()
                self.log.info(f"Server started successfully at http://{host}:{port}")
                while self.server_running: await asyncio.sleep(0.5)
                self.log.info("Server shutdown signal received.")
                await site.stop()
                self.log.info("Server site stopped.")
            async def cleanup_runner(): 
                await runner.cleanup()
                self.log.info("Server runner cleaned up.")

            loop.run_until_complete(start_runner())
            loop.run_until_complete(cleanup_runner())

        except OSError as e:
             if "address already in use" in str(e).lower(): 
                self.log.critical(f"Server failed: Port {port} on '{host}' in use.")
             else: 
                self.log.critical(f"Server failed OS error: {e}", exc_info=True)
             self.server_running = False
        except Exception as e: 
            self.log.critical(f"Server task critical error: {e}", exc_info=True)
            self.server_running = False
        finally:
            if loop and not loop.is_closed():
                 if in_thread: 
                    loop.close()
                 self.log.info("Server asyncio loop closed.")
            self.loop = None
            self.log.info("Server task finished.")


    def start_server(self, host: str = "0.0.0.0", port: int = 8080, background: bool = False):
        """Starts the search API server."""
        if self.server_running: 
            self.log.warning("Server already running/starting.")
            return
        self.server_running = True
        if background:
            self.log.info("Starting server in background thread...")
            self.server_thread = threading.Thread(target=self._server_task, args=(host, port, True), daemon=True)
            self.server_thread.start()
            time.sleep(1)
            if not self.server_running: 
                self.log.error("Server failed to start in background.")
                self.server_thread = None
        else:
            self.log.info("Starting server in foreground (blocking)...")
            try: 
                self._server_task(host, port, False)
            except KeyboardInterrupt: 
                self.log.info("Server stopped by user.")
            finally: 
                self.server_running = False


    def stop_server(self):
        """Stops the running search API server."""
        if not self.server_running and self.server_thread is None: 
            self.log.warning("Server not running.")
            return
        self.log.info("Attempting to stop server...")
        self.server_running = False
        if self.loop and self.loop.is_running(): 
            _ = self.loop.call_soon_threadsafe(self.loop.stop)
            self.log.debug("Requested asyncio loop stop.")
        if self.server_thread and self.server_thread.is_alive():
            self.log.info("Waiting for server thread exit...")
            self.server_thread.join(timeout=10)
            if self.server_thread.is_alive(): 
                self.log.warning("Server thread join timed out.")
            else: 
                self.log.info("Server thread finished.")
            self.server_thread = None
        if self.server_running: 
            self.log.warning("Server flag still true after stop.")
            self.server_running = False
        self.log.info("Server stop sequence completed.")


    # === Static Utility Methods ===

    @staticmethod
    def get_chunk_ptr(index: int, chunk_size: int, chunk_overlap: int) -> int:
        """Calculates the start character index of a chunk."""
        index = max(0, index)
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: 
            step = chunk_size
        return index * step

    @staticmethod
    def get_chunk(text: str, index: int, chunk_size: int, chunk_overlap: int ) -> str:
        """Extracts a single text chunk based on index."""
        if not text: 
            return ""
        chunk_start = IcoTqStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        return text[chunk_start : chunk_start + chunk_size]

    @staticmethod
    def get_span_chunk(text:str, index: int, count:int, chunk_size: int, chunk_overlap: int):
        """Extracts a span of text covering 'count' base chunks."""
        if not text or count < 1: 
            return ""
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: 
            step = chunk_size
        chunk_start = IcoTqStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        chunk_end = chunk_start + chunk_size + step * (count - 1)
        return text[chunk_start : chunk_end]

    @staticmethod
    def get_chunks(text:str, chunk_size: int, chunk_overlap: int) -> list[str]: # Changed from List
        """Splits text into overlapping chunks."""
        if not text: 
            return []
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: 
            step = chunk_size
        text_len = len(text)
        chunks: list[str] = []
        for i in range(0, text_len, step):
             chunk = text[i : i + chunk_size]
             if not chunk: 
                break
             chunks.append(chunk)
             if i + chunk_size >= text_len: 
                break
        return chunks

    def resolve_device(self, device: str | None = None) -> str: # Changed Optional
        """Resolves 'auto' device to cuda, mps, or cpu."""
        dev = device if device else self.config.get('embeddings_device', 'auto')
        if dev == 'auto':
            if torch.cuda.is_available(): 
                return 'cuda'
            elif torch.backends.mps.is_available(): 
                return 'mps'
            else: 
                return 'cpu'
        elif dev in ['cuda', 'mps', 'cpu']:
            if dev == 'cuda' and not torch.cuda.is_available(): 
                self.log.warning("CUDA requested but not available, fallback CPU.")
                return 'cpu'
            if dev == 'mps' and not torch.backends.mps.is_available(): 
                self.log.warning("MPS requested but not available, fallback CPU.")
                return 'cpu'
            return dev
        else:
            self.log.warning(f"Unsupported device '{dev}', fallback 'auto'.")
            return self.resolve_device('auto')