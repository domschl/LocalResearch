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

import torch
from sentence_transformers import SentenceTransformer

from typing import TypedDict, cast, NotRequired
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


class IcoTqStore:
    def __init__(self) -> None:
        self.log:logging.Logger = logging.getLogger("IcoTqStore")
        # Disable log spam
        tmp = logging.getLogger("transformers_modules")
        tmp.setLevel(logging.ERROR)
        config_path = os.path.expanduser("~/.config/icotq")  # Turquoise icosaeder
        if os.path.isdir(config_path) is False:
            os.makedirs(config_path)
        self.lib: list[LibEntry] = []
        self.pdf_index:dict[str, PDFIndex] = {}
        self.config_file:str = os.path.join(config_path, "icoqt.json")
        self.config:IcotqConfig
        self.server_running:bool = False
        self.loop:asyncio.AbstractEventLoop
        self.server_thread:threading.Thread | None = None
        
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            self.config = IcotqConfig({
                'icotq_path': '~/IcoTqStore',
                'tq_sources': [
                    TqSource({
                    'name': 'Calibre',
                    'tqtype': 'calibre_library',
                    'path': '~/ReferenceLibrary/Calibre Library',
                    'file_types': ['txt', 'pdf']
                    }),
                    TqSource({
                        'name': 'Notes',
                        'tqtype': 'folder',
                        'path': '~/Notes',
                        'file_types': ['md']
                    })],
                'embeddings_model_name': 'granite-embedding-107m-multilingual',
                'embeddings_device': 'auto',
                'embeddings_model_trust_code': True
                })
            self.save_config()
            self.log.warning(f"Created default configuration at {self.config_file}, please review!")
        self.root_path:str = os.path.expanduser(self.config['icotq_path'])
        if os.path.exists(self.root_path) is False:
            os.makedirs(self.root_path)
            self.log.warning(f"Creating IcoTq storage path at {self.root_path}, all IcoTq data will reside there. Modify {self.config_file} to chose another location!")
        model_list_path = os.path.join(self.root_path, "model_list.json")
        self.model_list: list[EmbeddingsModel] = []
        if os.path.exists(model_list_path) is True:
            with open(model_list_path, 'r') as f:
                self.model_list = json.load(f)
        else:
            self.model_list = [
                {
                    'model_hf_name': 'nomic-ai/nomic-embed-text-v2-moe',
                    'model_name': 'nomic-embed-text-v2-moe',
                    'emb_dim': 768,  #  Matryoshka Embeddings
                    'max_input_token': 512,
                    'chunk_size': 2048,
                    'chunk_overlap': 2048 // 3
                },
                {
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
                }
            ]
            with open(model_list_path, 'w') as f:
                json.dump(self.model_list, f, indent=4)
            self.log.warning(f"Initialized {model_list_path} with default embeddings model list. Please verify.")
        self.current_model: EmbeddingsModel | None = None
        self.engine: SentenceTransformer | None = None
        self.device: str | None = None
        self.embeddings_matrix: torch.Tensor | None = None
        if self.config['embeddings_model_name'] != "":
            _ = self.load_model(self.config['embeddings_model_name'], self.config['embeddings_device'], self.config['embeddings_model_trust_code'])
        config_subdirs = ['Embeddings', 'PDFTextCache']
        for cdir in config_subdirs:
            full_path = os.path.join(self.root_path, cdir)
            if os.path.isdir(full_path) is False:
                os.makedirs(full_path)
        self.embeddings_path: str = os.path.join(self.root_path, "Embeddings")
        for source in self.config['tq_sources']:
            valid:bool = True
            known_types: list[str] = ['txt', 'md', 'pdf']
            for tp in source['file_types']:
                if tp not in known_types:
                    self.log.error(f"Source {source} has invalid file type {tp}, allowed are {known_types}, ignoring this source!")
                    valid = False
                    break
            if os.path.exists(os.path.expanduser(source['path'])) is False:
                self.log.error(f"Source {source} has invalid file path {source['path']}, ignoring this source!")
                valid = False
            known_tqtypes = ['calibre_library', 'folder']
            if source['tqtype'] not in known_tqtypes:
                self.log.error(f"Source {source} has invalid tqtype {source['tqtype']}, valid are {known_tqtypes}, ignoring this source!")
                valid = False
            if valid is False:
                self.config['tq_sources'].remove(source)
                self.log.warning(f"Please fix configuration file {self.config_file}")
        self.read_library()
        if self.current_model is not None:
            _ = self.load_tensor()
        
    def list_sources(self) -> None:
        for id, source in enumerate(self.config['tq_sources']):
            print(f"{id:02d} {source['name']}, {source['tqtype']}, {source['path']}, {source['file_types']}")

    def read_library(self):
        print("\rLoading library...", end="", flush=True)
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        if os.path.exists(lib_path) is True:
            with open(lib_path, 'r') as f:
                self.lib = json.load(f)
            print("\r", end="", flush=True)
            self.log.info(f"Library loaded, {len(self.lib)} entries")
        else:
            print("\r", end="", flush=True)
            self.log.info(f"No current library state at {lib_path}")
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        pdf_cache_index = os.path.join(pdf_cache, "pdf_index.json")
        print("\rLoading PDF cache...", end="", flush=True)
        if os.path.exists(pdf_cache_index):
            with open(pdf_cache_index, 'r') as f:
                self.pdf_index = json.load(f)
            print("\r", end="", flush=True)
            self.log.info(f"PDF text cache loaded, {len(self.pdf_index.keys())} entries")
        else:
            self.pdf_index = {}
            print("\r", end="", flush=True)

    def load_config(self):
        with open(self.config_file, 'r') as f:
            iqc: IcotqConfig = json.load(f)
            self.config = iqc

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.log.info(f"Configuration changes saved to {self.config_file}")

    def save_pdf_cache_state(self):
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        pdf_cache_index = os.path.join(pdf_cache, "pdf_index.json")
        with open(pdf_cache_index, 'w') as f:
            json.dump(self.pdf_index, f, indent=2)

    def save_tensor(self) -> bool:
        if self.current_model is None:
            self.log.error("Can't save embeddings tensor: no current model information available!")
            return False
        embeddings_tensor_file = os.path.join(self.embeddings_path, f"embeddings_{self.current_model['model_name']}.pt")
        if self.embeddings_matrix is not None:
            try:
                torch.save(self.embeddings_matrix, embeddings_tensor_file)  # pyright: ignore[reportUnknownMemberType]
            except Exception as e:
                self.log.error(f"Failed to save embeddings tensor to {embeddings_tensor_file}: {e}")
                return False
            self.log.info(f"Embeddings tensor saved to {embeddings_tensor_file}")
            return True
        else:
            if os.path.exists(embeddings_tensor_file) is True:
                os.remove(embeddings_tensor_file)
                self.log.error(f"No embeddings available to save, removing obsolete tensor {embeddings_tensor_file}")
            else:
                self.log.error("No embeddings available to save")
            return False

    def resolve_device(self, device:str) -> str:
        if device=='auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            return device

    def load_tensor(self, model_name:str|None=None, device:str|None=None, warn:bool=True) -> bool:
        if device is None:
            device = self.config['embeddings_device']
        if model_name is None and self.current_model is None:
            self.log.error("Can't save embeddings tensor: no current model information available!")
            return False
        if model_name is None and self.current_model is not None:
            model_name = self.current_model['model_name']
        if model_name is None:
            self.log.error("Model not specified")
            return False
        embeddings_tensor_file = os.path.join(self.embeddings_path, f"embeddings_{model_name}.pt")
        map_location = torch.device(self.resolve_device(device))
        if os.path.exists(embeddings_tensor_file):
            self.embeddings_matrix = torch.load(embeddings_tensor_file, map_location=map_location)  # pyright: ignore[reportUnknownMemberType]
        else:
            self.embeddings_matrix = None
        if self.embeddings_matrix is not None and self.current_model is not None:
            sum = 0
            model_name = self.current_model['model_name']
            for entry in self.lib:
                if model_name in entry['emb_ptrs']:
                    sum += entry['emb_ptrs'][model_name][1]
            if warn == True:
                self.log.info(f"Matrix: {self.embeddings_matrix.shape}, chunks: {sum}, texts: {len(self.lib)}")
                if sum != self.embeddings_matrix.shape[0]:
                    self.log.warning(f"Embeddings-matrix index incompatible with text library! User 'index purge' to rebuild index! Info: Sum: {sum}, EmbMat: {self.embeddings_matrix.shape}")
            return True
        else:
            self.log.warning("No embeddings index available! Use 'sync' to import text, 'index' to generate embeddings.")
            return False

    def write_library(self):
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        with open(lib_path, 'w') as f:
            json.dump(self.lib, f, indent=2)
        self.save_pdf_cache_state()

    def get_pdf_text(self, desc:str, full_path:str) -> tuple[str | None, bool]:
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        text: str | None = None
        if desc in self.pdf_index:
            cur_file_size = os.path.getsize(full_path)
            if cur_file_size == self.pdf_index[desc]['file_size'] and self.pdf_index[desc]['previous_failure'] is False:
                basename = os.path.basename(self.pdf_index[desc]['filename'])
                local_path = os.path.join(pdf_cache, basename)
                try:
                    with open(local_path, 'r') as f:
                        text = f.read()
                        return text, False
                except Exception as e:
                    self.log.warning(f"Failed to read PDF cache file for {desc}: {e}")
                    del self.pdf_index[desc]
                    text = None
            else:
                if cur_file_size != self.pdf_index[desc]['file_size']:
                    self.log.info(f"PDF file {full_path} has changed, re-importing")
                    del self.pdf_index[desc]
                else:
                    # self.log.info(f"PDF file {full_path} has no text (extract failed before), ignoring")
                    return None, False  # Known failure case
        changed: bool = False
        if text is None:
            doc = pymupdf.open(full_path)
            text = ""
            for page in doc:
                page_text = page.get_text()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                if isinstance(page_text, str) is False:
                    self.log.error(f"Can't read page of {full_path}, ignoring page")
                    continue
                page_text = cast(str, page_text)
                text += page_text
            if text == "":
                text = None
                failure = True
                cache_filename = ""
                self.log.info(f"Failed to extract text from: {desc}")
            else:
                cache_filename = str(uuid.uuid4())
                failure = False
            pdf_ind: PDFIndex = {
                'filename': cache_filename,
                'file_size': os.path.getsize(full_path),
                'previous_failure': failure
            }
            if failure is False and text is not None:
                with open(os.path.join(pdf_cache, pdf_ind['filename']), 'w') as f:
                    _ = f.write(text)
                    self.log.info(f"Added {desc} to PDF cache, size: {len(self.pdf_index.keys())}")
            self.pdf_index[desc] = pdf_ind
            changed = True
        return text, changed

    def sync_texts(self, max_imports: int|None = None):
        if len(self.config['tq_sources']) == 0:
            self.log.error(f"No valid sources defined in config, can't import")
            return
        lib_changed = False
        tensor_debris: dict[str, list[tuple[int, int]]] = {}     # {model_name -> list of (emb_ptr, emb_len)}
        abort_import = False
        debris_candidates: list[str] = []
        lib_counter = 0
        for source in self.config['tq_sources']:
            if abort_import is True:
                break
            source_path = os.path.expanduser(source['path'])
            for entry in self.lib:
                if entry['source_name'] == source['name']:
                    debris_candidates.append(entry['desc_filename'])
            for root, _dir, files in os.walk(source_path):
                for filename in files:
                    if abort_import is True:
                        break
                    if max_imports is not None and lib_counter >= max_imports:
                        if len(self.lib) > max_imports:
                            self.log.warning(f"Pruning library to {max_imports} entries!")
                            for entry in self.lib[max_imports:]:
                                if entry['emb_ptrs'] != {}:
                                    for entry_model_name in entry['emb_ptrs']:
                                        if entry_model_name not in tensor_debris:
                                            tensor_debris[entry_model_name] = []
                                        tensor_debris[entry_model_name].append(entry['emb_ptrs'][entry_model_name])
                            self.lib = self.lib[:max_imports]
                        lib_changed = True
                        self.log.warning(f"Import reached max {max_imports}, library: {len(self.lib)} entries")
                        abort_import = True
                        break
                    parts = os.path.splitext(filename)
                    file_base = parts[0]
                    if len(parts[1]) > 0:
                        ext = parts[1][1:].lower()  # remove leading '.'
                    else:
                        ext = ""
                    if ext not in source['file_types']:
                        continue
                    alt_exists = False
                    if ext in ['epub', 'pdf']:  # Check if the file is available in a better file format (e.g. txt)
                        for alt in ['txt', 'epub']:
                            if alt == ext:
                                continue
                            alt_file = os.path.join(root, file_base + '.' + alt)
                            if os.path.exists(alt_file):
                                alt_exists = True
                        if alt_exists is True:  # better format of same file exist, so skip this one
                            continue                    
                    full_path = os.path.join(root, filename)
                    desc_path = "{"+ source['name'] + "}" + full_path[len(source_path):]
                    if desc_path in debris_candidates:
                        debris_candidates.remove(desc_path)
                    lib_counter += 1
                    in_lib = False
                    lib_text: str|None = None
                    lib_index: int|None = None
                    for ind, entry in enumerate(self.lib):
                        if entry['desc_filename'] == desc_path:
                            in_lib = True
                            lib_index = ind
                            lib_text = entry['text']
                            break
                    text = None
                    if ext in ['md', 'py', 'txt']:
                        with open(full_path, 'r') as f:
                            text = f.read()
                        if lib_text is not None:
                            if text == lib_text:
                                continue
                            else:
                                self.log.warning(f"Text for {desc_path} has changed!")
                                lib_changed = True
                        else:
                            lib_changed = True
                    elif ext == 'pdf':
                        text, changed = self.get_pdf_text(desc_path, full_path)
                        if in_lib is True and text == lib_text:
                            continue
                        else:
                            if changed is True:
                                lib_changed = True
                    else:
                        self.log.error(f"Unsupported conversion {ext} to text at {desc_path}")
                        lib_counter -= 1
                        continue
                    if text is not None:
                        if in_lib is True and lib_index is not None:
                            self.lib[lib_index]['text'] = text
                            if self.lib[lib_index]['emb_ptrs'] != {}:
                                for entry_model_name in self.lib[lib_index]['emb_ptrs']:
                                    if entry_model_name not in tensor_debris:
                                        tensor_debris[entry_model_name] = []
                                    tensor_debris[entry_model_name].append(self.lib[lib_index]['emb_ptrs'][entry_model_name])
                            self.lib[lib_index]['emb_ptrs'] = {}
                        else:
                            entry: LibEntry = LibEntry({
                                'source_name': source['name'],
                                'desc_filename': desc_path,
                                'filename': full_path,
                                'text': text,
                                'emb_ptrs': {}
                            })
                            self.lib.append(entry)
                        lib_changed = True
        current_model_name:str|None = None
        if self.current_model is not None:
            current_model_name = self.current_model['model_name']
        pdf_cache_path = os.path.join(self.root_path, "PDFTextCache")
        for debris_desc in debris_candidates:
            self.log.info(f"Removing debris {debris_desc}")
            if debris_desc in self.pdf_index:
                cache_name = self.pdf_index[debris_desc]['filename']
                if cache_name != "" and os.path.exists(os.path.join(pdf_cache_path, cache_name)):
                    os.remove(os.path.join(pdf_cache_path, cache_name))
                    self.log.info(f"Cached entry {debris_desc}, file {cache_name} removed")
                else:
                    if cache_name != "" and self.pdf_index[debris_desc]['previous_failure'] is False:
                        self.log.warning(f"Cache entry for {debris_desc} at {cache_name} does not exist, inconsistent cache!")
                del self.pdf_index[debris_desc]
                lib_changed = True
        for entry in self.lib:
            if entry['desc_filename'] in debris_candidates:
                for entry_model_name in entry['emb_ptrs']:
                    if entry_model_name not in tensor_debris:
                        tensor_debris[entry_model_name] = []
                    tensor_debris[entry_model_name].append(entry['emb_ptrs'][entry_model_name])
                self.log.info(f"Library entry {entry['desc_filename']} removed")
                self.lib.remove(entry)
                lib_changed = True
        if len(tensor_debris.keys()) > 0:
            self.log.warning("Rewriting embeddings tensors to remove obsolete entries")
            for model_name in tensor_debris:
                # Pass 1 cut emb_vectors
                self.log.info(f"Removing {len(tensor_debris[model_name])} chunks from tensor {model_name}")
                if len(tensor_debris[model_name]) == 0:
                    continue
                tensor_valid:bool = False
                removals = sorted(tensor_debris[model_name], reverse=True)
                if self.load_tensor(model_name=model_name, warn=False) is False or self.embeddings_matrix is None:
                    self.log.warning(f"Failed to load tensor for model {model_name}, no embeddings have been created yet, use 'embed' command to create them.")
                else:
                    tensor_valid = True
                if tensor_valid and self.embeddings_matrix is not None:
                    tensor_changed = False
                    for start, length in removals:  # sorted in reverse start idx already
                        self.embeddings_matrix = torch.cat([self.embeddings_matrix[:start,:], self.embeddings_matrix[start+length:,:]])
                        tensor_changed = True
                    if tensor_changed is True:
                        _ = self.save_tensor()
                # Pass 2 shift emb_ptr starts
                for ind, entry in enumerate(self.lib):
                    if model_name in entry['emb_ptrs']:
                        if tensor_valid is False:
                            del self.lib[ind]['emb_ptrs'][model_name]
                        else:
                            offs = 0
                            start, _ = entry['emb_ptrs'][model_name]
                            for st, ln in tensor_debris[model_name]:
                                if st<start:
                                    offs += ln
                            if offs > 0:
                                new_offs = entry['emb_ptrs'][model_name][0] - offs
                                # self.log.info("Moving emb_ptr start by {offs} to {new_offs}")
                                if new_offs < 0:
                                    self.log.error(f"Things went horribly wrong processing {model_name} at {entry['desc_filename']}, offset went to {new_offs} (delta: {offs})")
                                    self.log.error("INCONSISTENT STATE!")
                                    return
                                self.lib[ind]['emb_ptrs'][model_name] = (new_offs, entry['emb_ptrs'][model_name][1])
                                lib_changed = True
            if current_model_name is not None:
                _ = self.load_tensor(model_name = current_model_name)
        if lib_changed is True:
            self.write_library()
            self.log.info(f"Changed library saved: {len(self.lib)} entries")

    @staticmethod
    def get_chunk_ptr(index: int, chunk_size: int, chunk_overlap: int) -> int:
        chunk_ptr: int = index * (chunk_size - chunk_overlap)
        return chunk_ptr

    @staticmethod
    def get_chunk(text: str, index: int, chunk_size: int, chunk_overlap: int ) -> str:
        chunk_start: int = IcoTqStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        chunk = text[chunk_start : chunk_start + chunk_size]
        return chunk

    @staticmethod
    def get_span_chunk(text:str, index: int, count:int, chunk_size: int, chunk_overlap: int):
        if count < 1:
            return ""
        chunk_start: int = IcoTqStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        offset = chunk_size - chunk_overlap
        chunk = text[chunk_start : chunk_start + chunk_size+offset * (count-1)]
        return chunk

    @staticmethod
    def get_chunks(text:str, chunk_size: int, chunk_overlap: int) -> list[str]:
        chunks = (len(text) - 1) // (chunk_size - chunk_overlap) + 1 
        text_chunks = [IcoTqStore.get_chunk(text, i, chunk_size, chunk_overlap) for i in range(chunks) ]
        return text_chunks
                
    def load_model(self, name: str, device:str="auto", trust_remote_code:bool=False) -> bool:
        self.log.info(f"Loading model {name}...")
        for model in self.model_list:
            if model['model_hf_name'] == name or model['model_name'] == name:
                try:
                    self.engine = SentenceTransformer(model['model_hf_name'], 
                                                      trust_remote_code=trust_remote_code)
                    self.device = self.resolve_device(device)
                    self.engine = self.engine.to(torch.device(self.device))
                    self.current_model = model
                    self.log.info(f"Model {name} loaded.")
                    self.config['embeddings_model_name'] = name
                    self.save_config()
                    return True
                except Exception as e:
                    self.log.error(f"Failed to load model {model}: {e}")
                    return False
        self.log.error(f"Model {name} is unknown, not in model list")
        return False

    def check_clean(self, dry_run:bool=True):
        self.log.info("Checking PDF cache...")
        lib_changed:bool = False
        dirty: bool = False
        errors: bool = False
        index_backup: dict[str, PDFIndex] = {}
        index_backup_valid:bool = False
        if dry_run is True:
            index_backup = self.pdf_index.copy()
            index_backup_valid = True
            
        debris: list[str] = []
        bad_cnt:int = 0
        for pdf_desc in self.pdf_index:
            if self.pdf_index[pdf_desc]['previous_failure'] is True:
                bad_cnt += 1
                continue
            found = False
            for entry in self.lib:
                if entry['desc_filename'] == pdf_desc:
                    found = True
                    break
            if found is False:
                debris.append(pdf_desc)
        if dry_run is True:
            self.log.info(f"PDF index contains {len(self.pdf_index.keys())} entries, {bad_cnt} PDFs with no extractable text, from which {len(debris)} are debris.")
        else:
            self.log.warning(f"Deleting {len(debris)} entries from PDF cache index")
            if len(debris) > 0:
                lib_changed = True
                dirty = True
        for pdf_desc in debris:
            del self.pdf_index[pdf_desc]
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        pdf_cache_index = "pdf_index.json"
        cache_files = [f for f in os.listdir(pdf_cache) if os.path.isfile(os.path.join(pdf_cache, f))]
        cnt = 0
        debris = []
        for filename in cache_files:
            if filename == pdf_cache_index:
                continue
            cnt += 1
            found = False
            for cf in self.pdf_index:
                if self.pdf_index[cf]['filename'] == filename:
                    found = True
                    break
            if found is False:
                debris.append(filename)
                dirty = True
        if dry_run is True:
            self.log.info(f"PDF cache contains {cnt} files, {bad_cnt} pointers to PDFs without text, {len(debris)} are debris and would be deleted")
        else:
            self.log.warning(f"Deleting {len(debris)} files from PDF cache.")
            for filename in debris:
                os.remove(os.path.join(pdf_cache, filename))

        if self.embeddings_matrix is not None and self.current_model is not None:
            sum = 0
            model_name = self.current_model['model_name']
            for entry in self.lib:
                if model_name in entry['emb_ptrs']:
                    sum += entry['emb_ptrs'][model_name][1]
            self.log.info(f"Matrix: {self.embeddings_matrix.shape}, chunks: {sum}, texts: {len(self.lib)}")
            if sum != self.embeddings_matrix.shape[0]:
                self.log.warning(f"Embeddings-matrix index incompatible with text library! User 'index purge' to rebuild index! Info: Sum: {sum}, EmbMat: {self.embeddings_matrix.shape}")
                errors = True
            else:
                fat = [0] * sum
                for entry in self.lib:
                    if model_name in entry['emb_ptrs']:
                        start, length = entry['emb_ptrs'][model_name]
                        for ind in range(start, start+length):
                            fat[ind] += 1
                    else:
                        self.log.warning(f"Entry {entry['desc_filename']} has not yet been indexed, please use 'index'")
                        errors = True
                for ind, fi in enumerate(fat):
                    if fi != 1:
                        errors = True
                        if fi == 0:
                            self.log.warning(f"Unused index slot at {ind}, this should not happen!")
                        else:
                            self.log.warning(f"Multiple allocations for index slot at {ind}: {fi} duplicate references!")
                if errors is False:
                    self.log.info(f"Embeddings-index is consistent with library, Tensor-shape: {self.embeddings_matrix.shape}")
        
        if dry_run is True and index_backup_valid is True:
            self.pdf_index = index_backup
        if dry_run is False and lib_changed is True:
            self.write_library()
        if dirty is True and dry_run is True:
            self.log.warning("Problems encounter, consider running 'clean' to fix.")
        if errors is True:
            self.log.warning("Embeddings-index is inconsistent with library, re-index using 'index [purge]' to fix! use 'purge' option to rebuild index from scratch.")
        if dirty is False and errors is False:
            self.log.info("No problems found.")

    def generate_embeddings(self, save_every_sec:int = 180, purge:bool=False):
        if self.current_model is None or self.engine is None:
            self.log.error("No current embeddings model loaded!")
            return
        if purge is True:
            self.embeddings_matrix = None
        start_time: float = time.time()
        for ind, entry in enumerate(self.lib):
            # self.log.info(f"Embedding: {ind+1}/{len(self.lib)}")
            if 'emb_ptrs' not in entry:
                self.lib[ind]['emb_ptrs'] = {}
            if self.current_model['model_name'] in entry['emb_ptrs'] and purge is False:
                continue
            text_chunks = self.get_chunks(entry['text'], self.current_model['chunk_size'], self.current_model['chunk_overlap'])
            if len(text_chunks) == 0:
                self.log.error(f"Cannot encode empty text list at {entry['desc_filename']}")
                continue
            self.log.info(f"Encoding {len(text_chunks)} chunks from {entry['desc_filename']}...")
            embeddings: list[torch.Tensor] = self.engine.encode(sentences=text_chunks, show_progress_bar=True, convert_to_numpy=False)  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
            emb_matrix = torch.stack(embeddings)

            if self.embeddings_matrix is None:
                start_ptr = 0
                self.embeddings_matrix = emb_matrix
            else:
                start_ptr = self.embeddings_matrix.shape[0]
                self.embeddings_matrix = torch.cat([self.embeddings_matrix, emb_matrix])
            emb_len = len(embeddings)
            del emb_matrix
            self.lib[ind]['emb_ptrs'][self.current_model['model_name']] = (start_ptr, emb_len)
            if save_every_sec > 0 and time.time() - start_time > save_every_sec:
                self.write_library()
                _ = self.save_tensor()
                start_time = time.time()
        self.write_library()
        _ = self.save_tensor()

    def search_vect(self, text:str) -> tuple[list[tuple[int, float]], torch.Tensor]:
        if self.embeddings_matrix is None or self.engine is None:
            self.log.error("No embeddings available!")
            return [], torch.Tensor([])
        vect: list[torch.Tensor] = self.engine.encode(sentences=[text], show_progress_bar=True, convert_to_numpy=False)   # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
        if len(vect) == 0:
            self.log.error("Failed to calculate embedding for search")
            return [], torch.Tensor([])
        if len(vect) > 1:
            self.log.warning("Result contains more than one vector, ignoring additional ones")
        search_vect:torch.Tensor = vect[0]
        simil: list[tuple[int, float]] = enumerate(torch.matmul(self.embeddings_matrix, search_vect).cpu().numpy().tolist())  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
        sorted_simil:list[tuple[int, float]] = sorted(simil, key=lambda x: x[1], reverse=True)
        return sorted_simil, search_vect

    def yellow_line_it(self, text: str, search_embeddings: torch.Tensor, context_length:int=16, context_steps:int=1) -> np.typing.NDArray[np.float32]:
        if self.embeddings_matrix is None or self.engine is None:
            self.log.error("No embeddings available at yellow-lining!")
            return np.array([], dtype=np.float32)
        clr: list[str] = []
        for i in range(0, len(text), context_steps):
            i0 = i - context_length // 2
            i1 = i + context_length // 2
            if i0 < 0:
                i1 = i1 - i0
                i0 = 0
            if i1 > len(text):
                i0 = i0 - (i1 - len(text))
                i1 = len(text)
            clr.append(text[i0:i1])
        if clr == []:
            clr = [text]
        embeddings: list[torch.Tensor] = self.engine.encode(sentences=clr, show_progress_bar=True, convert_to_numpy=False)  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
        emb_matrix = torch.stack(embeddings)
        yellow_vect: np.typing.NDArray[np.float32] = torch.matmul(emb_matrix, search_embeddings).cpu().numpy()  # pyright: ignore[reportUnknownMemberType]
        return yellow_vect

    def search(self, search_text:str, max_results:int=2, yellow_liner:bool=False, context_length:int=16, context_steps:int=4, compression_mode:str="none"):
        if self.current_model is None:
            self.log.error("No current model!")
            res:list[SearchResult] = []
            return res
        sorted_simil_all, search_embeddings = self.search_vect(search_text)
        sorted_simil = sorted_simil_all[:max_results]
        search_results: list[SearchResult] = []
        resolved_list: list[tuple[str, int, int, float, LibEntry]] = []
        yellow_liner_weights: np.typing.NDArray[np.float32] | None
        for result in sorted_simil:
            idx = result[0]
            cosine = result[1]
            for entry in self.lib:
                if self.current_model['model_name'] in entry['emb_ptrs']:
                    start, length = entry['emb_ptrs'][self.current_model['model_name']]
                    if idx >= start and idx < start + length:
                        print(f"{entry['desc_filename']}: {cosine}")
                        resolved_list.append((entry['desc_filename'], idx, 1, cosine, entry))
        srla = sorted(resolved_list)
        for ind, sra in reversed(list(enumerate(srla))):
            if ind+1 == len(srla):
                continue
            if sra[0] == srla[ind+1][0]:  # same desc
                start, length = sra[4]['emb_ptrs'][self.current_model['model_name']]
                start2, _length2 = srla[ind+1][4]['emb_ptrs'][self.current_model['model_name']]
                if start + length >= start2:  # Overlapping consequtive
                # if sra[4]['emb_ten_idx'] + sra[4]['emb_ten_size'] >= srla[ind+1][4]['emb_ten_idx']:  # Overlapping consequtive
                    del srla[ind+1]
                    cnt:int = srla[ind][2] + 1
                    cosine: float = sra[3]
                    if ind+1 < len(srla) and sra[3] < srla[ind+1][3]:
                        cosine = srla[ind+1][3]  # get the better score
                    srla[ind] = (srla[ind][0], srla[ind][1], cnt, cosine, srla[ind][4])
                    self.log.info(f"Merged two consequtive search postions into a span-chunk for {sra[0]}")
        for sra in srla:
            desc, idx, count, cosine, entry = sra
            start, _length = entry['emb_ptrs'][self.current_model['model_name']]
            chunk:str = self.get_span_chunk(entry['text'], idx - start, count, self.current_model['chunk_size'], self.current_model['chunk_overlap'])
            if compression_mode == "light":
                new_chunk = chunk
                old_chunk = None
                while new_chunk != old_chunk:
                    old_chunk = new_chunk
                    new_chunk = old_chunk.replace("  ", " ").replace("\n\n", "\n")
                chunk = new_chunk
            elif compression_mode == "full":
                new_chunk = chunk
                old_chunk = None
                while new_chunk != chunk:
                    old_chunk = new_chunk
                    new_chunk = old_chunk.replace("\n", " ").replace("\r"," ").replace("\t", " ").replace("  ", " ")
                chunk = new_chunk
            if yellow_liner is True:
                yellow_liner_weights = self.yellow_line_it(chunk, search_embeddings, context_length=context_length, context_steps=context_steps)
            else:
                yellow_liner_weights = None
            sres:SearchResult = {
                'cosine': cosine,
                'index': idx,
                'offset': idx - start,
                'desc': desc,
                'chunk': chunk,  
                'text': entry['text'],
                'yellow_liner': yellow_liner_weights
            }
            search_results.append(sres)
        search_results = sorted(search_results, key=lambda x: x['cosine'], reverse=True)
        return search_results


    async def search_handler(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        try:
            data:SearchRequest = await request.json()
        except Exception as e:
            self.log.error(f"Search request format failure: {e}")
            return aiohttp.web.json_response(data={}, status=400)
        self.log.info(f"Search request received: {data}")
        search_text:str = data['search_text']
        max_results:int = data.get('max_results', 2)
        yellow_liner:bool = data.get('yellow_liner', False)
        context_length:int = data.get('context_length', 16)
        context_steps:int = data.get('context_steps', 4)
        compression_mode:str = data.get('compression_mode', 'none')
        search_results = self.search(search_text, max_results=max_results, yellow_liner=yellow_liner, context_length=context_length, context_steps=context_steps, compression_mode=compression_mode)
        self.log.info(f"Remote search request: {len(search_results)} answers")
        return aiohttp.web.json_response(search_results)

    def _server_task(self, host:str, port:int, in_thread:bool=False):
        self.log.info(f"_server_task, in_thread={in_thread}")
        app = aiohttp.web.Application()
        _ = app.router.add_post('/search', self.search_handler)

        runner = aiohttp.web.AppRunner(app)
        if in_thread is True:
            self.loop = asyncio.new_event_loop()
        else:
            self.loop = asyncio.get_event_loop()
        self.server_running = True

        async def start():
            await runner.setup()
            site = aiohttp.web.TCPSite(runner, host, port)
            await site.start()
            self.log.info(f"Server started at {host}:{port}")
            while self.server_running:
                await asyncio.sleep(0.1)
            await site.stop()
            await runner.cleanup()
            self.log.info(f"Server stopped")

        self.loop.run_until_complete(start())
        while self.server_running:
            time.sleep(0.1)
    
    def start_server(self, host:str="0.0.0.0", port:int=8080, background:bool=False):
        if self.server_running is True:
            self.log.warning("Server already running!")
            return
        if background is False:
            self.log.info("Starting blocking server task")
            self._server_task(host, port)
        else:
            self.log.info("Starting server thread")
            in_thread:bool = True
            self.server_thread = threading.Thread(target=self._server_task, args=(host, port, in_thread), daemon=True)
            self.server_thread.start()


    def stop_server(self):
        if self.server_running is False:
            self.log.warning("Server is not running!")
            return
        self.server_running = False
        self.loop.stop()
        # stop thread, if it is running
        if self.server_thread is not None and self.server_thread.is_alive():
            self.server_thread.join()
            self.server_thread = None
        
        
        
