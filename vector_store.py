import os
import logging
import json
import time
import datetime
import hashlib
import tempfile
from typing import TypedDict, cast, Any, Callable
import subprocess
import colorsys
import math
import numpy as np

import pymupdf  # pyright: ignore[reportMissingTypeStubs]
import pymupdf4llm  # pyright: ignore[reportMissingTypeStubs]  # XXX currently locked to 0.19, otherwise export returns empty docs, requires investigation!
import torch
import transformers
from sentence_transformers import SentenceTransformer

# INTEL XPU incantation:
# uv pip install -U --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu

from research_defs import MetadataEntry
from research_tools import DocumentTable
from markdown_handler import MarkdownTools
from orgmode_handler import OrgmodeTools
from calibre_handler import CalibreTools

support_dim3d = False
try:
    import umap  # pyright: ignore[reportMissingTypeStubs]
    support_dim3d = True
except ImportError:
    umap = None
    pass


class DocumentSource(TypedDict):
    type: str
    path: str
    file_types: list[str]


class DocumentConfig(TypedDict):
    version: int
    document_sources: dict[str, DocumentSource]
    vars: dict[str, tuple[str,str]]
    publish_path: str


class VectorConfig(TypedDict):
    version: int
    embeddings_model_name: str
    embeddings_device: str
    embeddings_model_trust_code: bool
    batch_base_multiplier: int
    chunk_size: int
    chunk_overlap: int
    chunk_batch_size: int
    oom_recoveries: int
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str


class TextLibraryEntry(TypedDict):
    source_name: str
    descriptor: str
    text: str


class PDFIndex(TypedDict):
    previous_failure: bool
    file_size: int


class ModelCheck(TypedDict):
    document_count: int
    embedding_count: int
    embedding_dim: int
    debris_count: int
    deleted_count: int
    missing_count: int
    model_name: str
    enabled: bool
    selected: bool


class EmbeddingModel(TypedDict):
    model_hf_name: str
    model_name: str
    batch_multiplier: int
    enabled: bool

    
class SequenceVersion(TypedDict):
    sequence: int


class SearchResultEntry(TypedDict):
    cosine: float
    hash: str
    chunk_index: int
    entry: TextLibraryEntry
    text: str|None
    significance: list[float]|None


class ProgressState(TypedDict):
    issues: int
    state: str
    percent_completion: float
    vars: dict[str, str]
    finished: bool

    
class Sha256CacheEntry(TypedDict):
    size: int
    modified: float
    sha256: str


def get_files_of_extensions(path:str, extensions: list[str]):
    result: list[str] = []
    if os.path.isdir(path) is False:
        return result
    for file in os.listdir(path):
        if not os.path.isdir(file):
            ext = os.path.splitext(file)[1]
            if ext and len(ext)>1:
                ext = ext[1:]
            else:
                continue
            if ext in extensions:
                result.append(file)
    return result


class VectorStore:
    def __init__(self, storage_path:str, config_path:str):
        self.current_version: int = 5
        self.log: logging.Logger = logging.getLogger("VectorStore")
        self.storage_path:str = storage_path
        self.config_path:str = config_path
        if os.path.isdir(self.config_path) is False:
            os.makedirs(self.config_path)
        self.config_changed: bool = False
        self.config_file:str = os.path.join(self.config_path, "vector_store.json")
        self.model_file:str = os.path.join(self.config_path, "model_list.json")
        self.config: VectorConfig = self.get_config()
        self.embeddings_path:str = os.path.join(self.storage_path, "embeddings")
        if os.path.isdir(self.embeddings_path) is False:
            os.makedirs(self.embeddings_path, exist_ok=True)
        self.visualization_3d:str = os.path.join(self.storage_path, "visualization_3d")
        if os.path.isdir(self.visualization_3d) is False:
            os.makedirs(self.visualization_3d)
        self.model_list: list[EmbeddingModel]
        self.get_model_list()
        for model in self.model_list:
            model_path = self.model_embedding_path(model['model_name'])
            if os.path.isdir(model_path) is False:
                os.makedirs(model_path, exist_ok=True)
        self.model: EmbeddingModel | None = None
        self.engine: SentenceTransformer | None = None
        self.perf: dict[str, float] = {}
        self.device: torch.device = torch.device(self.resolve_device())
        self.log.info(f">{self.config['embeddings_model_name']}< on device >{self.device}< using transformers {transformers.__version__} on torch {torch.__version__}")
        required_transformers_version = "4.57.0"
        if self.check_version(transformers.__version__, required_transformers_version) is False:
            self.log.error(f"Required minimal version {required_transformers_version} for transformers not found, you are on your own!")
            self.log.info("While .57 is not yet released, use: pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview")

    def check_version(self, current:str, required_minimal:str) -> bool:
        cur = current.split('.')
        req = required_minimal.split('.')
        if len(cur)<len(req):
            it = len(cur)
        else:
            it = len(req)
        for i in range(it):
            if int(cur[i]) < int(req[i]):
                return False
            if int(cur[i]) > int(req[i]):
                return True
        return True
        
    def model_embedding_path(self, model_name: str) -> str:
        return os.path.join(self.embeddings_path, model_name)

    def get_model_list(self):
        self.model_list = []
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, "r") as f:
                    self.model_list = cast(list[EmbeddingModel], json.load(f))
            except Exception as e:
                self.log.error(f"Failed to load model list: {e}, reverting to default")
        if len(self.model_list) == 0:
            self.model_list = [
                {
                    'model_hf_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'model_name': 'all-MiniLM-L6-v2',
                    'batch_multiplier': 128,
                    'enabled': True,
                },
                {
                    'model_hf_name': 'ibm-granite/granite-embedding-107m-multilingual',
                    'model_name': 'granite-embedding-107m-multilingual',
                    'batch_multiplier': 64,
                    'enabled': True,
                },
                {
                    'model_hf_name': 'ibm-granite/granite-embedding-278m-multilingual',
                    'model_name': 'granite-embedding-278m-multilingual',
                    'batch_multiplier': 32,
                    'enabled': True,
                },
                {
                    'model_hf_name': 'nomic-ai/nomic-embed-text-v2-moe',
                    'model_name': 'nomic-embed-text-v2-moe',
                    'batch_multiplier': 32,
                    'enabled': True,
                },
                {
                    'model_hf_name': 'google/embeddinggemma-300m',
                    'model_name': 'embeddinggemma',
                    'batch_multiplier': 1,
                    'enabled': True,
                },
                {
                    'model_hf_name': 'Qwen/Qwen3-Embedding-0.6B',
                    'model_name': 'Qwen3-Embedding-0.6B',
                    'batch_multiplier': 1,
                    'enabled': True,
                },
            ]
            self.save_model_list()

    def save_model_list(self):
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.model_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(self.model_list, temp_file, indent=4)
            os.replace(temp_path, self.model_file)  # atomic update
        except Exception as e:
            self.log.error("Model-list-file update interrupted, not updated.")
            os.remove(temp_path)
            raise e

    def get_config(self) -> VectorConfig:
        valid = False
        config: VectorConfig | None = None
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config = cast(VectorConfig, json.load(f))
                    valid = True
                    if config['version'] < self.current_version:
                        self.log.error(f"Version of config file {self.config_file} is outdated, upgrading to default version {self.current_version}")
                        valid = False
            except Exception as e:
                self.log.error(f"Failed to read config {self.config_file}: {e}, resetting to default configuration!")
                valid = False
        if valid is False or config is None:
            config = VectorConfig({
                'version': self.current_version,
                'embeddings_model_name': 'granite-embedding-107m-multilingual',
                'embeddings_device': 'auto',
                'embeddings_model_trust_code': True,
                'batch_base_multiplier': 1,
                'chunk_size': 3072,
                'chunk_overlap': 1024,
                'chunk_batch_size': 2,
                'oom_recoveries': 2,
                'umap_n_neighbors': 15,
                'umap_min_dist': 0.1,
                'umap_metric': 'cosine',
            })
            self.save_config(config)
            self.log.warning(f"Default configuration created at {self.config_file}, please review!")
        self.config_changed = False
        return config

    def save_config(self, config: VectorConfig):
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.config_file))
        config['version'] = self.current_version
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:
                json.dump(config, temp_file, indent=4)
            os.replace(temp_path, self.config_file)  # atomic update
        except Exception as e:
            self.log.error("Config-file update interrupted, not updated.")
            os.remove(temp_path)
            raise e

    def get_model_index(self, model_name:str) -> int|None:
        for ind, model in enumerate(self.model_list):
            if model['model_name'] == model_name:
                return ind+1
        return None

    def check_indices(self, doc_hashes:list[str], clean:bool) -> list[ModelCheck]:
        all_deleted = 0
        model_check:list[ModelCheck] = []
        
        for model in self.model_list:
            cnt = 0
            debris_cnt = 0
            deleted_cnt = 0
            if model['enabled'] is True:
                indices_path = self.model_embedding_path(model['model_name'])
                emb_cnt, dim = self.get_embeddings_size(indices_path)
                file_list = get_files_of_extensions(indices_path, ["pt"])
                # hash_list = [os.path.splitext(name)[0] for name in file_list]
                for filename in file_list:
                    hash = os.path.splitext(filename)[0]
                    if hash in doc_hashes:
                        cnt += 1
                    else:
                        debris_cnt += 1
                        if clean is True:
                            os.remove(os.path.join(indices_path, filename))
                            deleted_cnt += 1
                            all_deleted += 1
                if model['model_name'] == self.config['embeddings_model_name']:
                    selected = True
                else:
                    selected = False
                missing_cnt = len(doc_hashes) - cnt
                model_check.append(ModelCheck({'document_count': cnt,
                                               'embedding_count':emb_cnt,
                                               'embedding_dim': dim,
                                               'debris_count': debris_cnt,
                                               'deleted_count': deleted_cnt,
                                               'missing_count': missing_cnt,
                                               'model_name': model['model_name'],
                                               'enabled': model['enabled'],
                                               'selected': selected}))
            else:
                model_check.append(ModelCheck({'document_count':0,
                                               'embedding_count': 0,
                                               'embedding_dim': 0,
                                               'debris_count': 0,
                                               'deleted_count': 0,
                                               'missing_count': 0,
                                               'model_name': model['model_name'],
                                               'enabled': False,
                                               'selected': False}))                
        return model_check
                    
    def select(self, ind: int) -> str | None:
        if ind<1 or ind>len(self.model_list):
            self.log.error(f"Invalid model ID {ind}, use 'list models' to get valid IDs")
            return None
        if self.model_list[ind-1]['enabled'] is False:
            if self.engine is not None:
                del self.engine
                self.engine = None
            if self.model is not None:
                del self.model
                self.model = None
            self.log.error(f"Model {self.model_list[ind-1]['model_name']} is disabled, use 'enable <ID>' to enable")
            return None
        new_model = self.model_list[ind-1]['model_name']
        if new_model == self.config['embeddings_model_name'] and self.model is not None and self.engine is not None:
            self.log.info(f"Model {new_model} was already active")
            return None
        if self.model is not None:
            del self.model
            self.model = None
        if self.engine is not None:
            del self.engine
            self.engine = None
        self.log.info(f"Model {new_model} active")
        self.config['embeddings_model_name'] = new_model
        self.config_changed = True
        self.save_config(self.config)
        return new_model

    def enable(self, ind: int) -> str | None:
        if ind<1 or ind>len(self.model_list):
            self.log.error(f"Invalid model ID {ind}, use 'list models' to get valid IDs")
            return None
        if self.model_list[ind-1]['enabled'] is False:
            self.model_list[ind-1]['enabled'] = True
            self.save_model_list()
            self.log.info(f"Model {self.model_list[ind-1]['model_name']} enabled, use 'select {ind}' to start using it, 'list models' for overview'")
            return self.model_list[ind-1]['model_name']
        else:
            self.log.warning(f"Model {self.model_list[ind-1]['model_name']} was already enabled")

    def disable(self, ind: int) -> str | None:
        if ind<1 or ind>len(self.model_list):
            self.log.error(f"Invalid model ID {ind}, use 'list models' to get valid IDs")
            return None
        if self.model_list[ind-1]['enabled'] is True:
            self.model_list[ind-1]['enabled'] = False
            self.save_model_list()
            self.log.info(f"Model {self.model_list[ind-1]['model_name']} disabled")
            if self.model_list[ind-1]['model_name'] == self.config['embeddings_model_name']:
                if self.model is not None:
                    del self.model
                    self.model = None
                if self.engine is not None:
                    del self.engine
                    self.engine = None
                self.log.warning(f"Model {self.model_list[ind-1]['model_name']} was currently active, use 'list models' and 'select <ID>' to active alternative")
            return self.model_list[ind-1]['model_name']
        else:
            self.log.warning(f"Model {self.model_list[ind-1]['model_name']} was already disabled")

    def get_embedding_filename(self, hash:str) -> str:
        current_embeddings_path = os.path.join(self.embeddings_path, self.config['embeddings_model_name'])
        return os.path.join(current_embeddings_path, hash+".pt")

    def resolve_device(self, device_name:str|None = None) -> str:
        if device_name is None:
            dev = self.config.get('embeddings_device', 'auto')
        else:
            dev = device_name
        if dev == 'auto':
            if torch.cuda.is_available(): 
                return 'cuda'
            elif torch.xpu.is_available():
                return 'xpu'
            elif torch.backends.mps.is_available(): 
                return 'mps'
            else: 
                return 'cpu'
        elif dev in ['cuda', 'mps', 'cpu', 'xpu']:
            if dev == 'cuda' and not torch.cuda.is_available(): 
                self.log.warning("CUDA requested but not available, fallback CPU.")
                return 'cpu'
            if dev == 'mps' and not torch.backends.mps.is_available(): 
                self.log.warning("MPS requested but not available, fallback CPU.")
                return 'cpu'
            if dev == 'xpu' and not torch.xpu.is_available():
                self.log.warning("XPU requested but not avaiable, fallback to CPU.")
                return 'cpu'
            return dev
        else:
            self.log.warning(f"Undefined device {dev}, using 'cpu' fallback")
            return 'cpu'

    def set_device(self, device_name:str):
        device_name = self.resolve_device(device_name)
        if device_name != self.config['embeddings_device'] or device_name != str(self.device):
            self.device = torch.device(device_name)
            self.config['embeddings_device'] = device_name
            self.save_config(self.config)
            self.log.info(f"(Re-)loading model on device {device_name}")
            self.load_model()
        
    def load_model(self, reload:bool = False):
        if self.model is None:
            for model in self.model_list:
                if model['model_name'] == self.config['embeddings_model_name']:
                    if model['enabled'] is False:
                        if self.engine is not None:
                            del self.engine
                            self.engine = None
                        if self.model is not None:
                            del self.model
                            self.model = None
                        self.log.error(f"Model {self.config['embeddings_model_name']} is disabled, use 'list models' and 'select <ID>' to activate a different model.")
                        return
                    else:
                        self.model = model                    
        if self.model is None:
            self.log.error(f"Invalid model {self.config['embeddings_model_name']} could not be identified, load_model failed!")
            return
        if self.engine is None or reload is True:
            if self.engine is not None:
                del self.engine
            self.engine = SentenceTransformer(self.model['model_hf_name'], device=str(self.device), trust_remote_code=self.config['embeddings_model_trust_code']) # .to(self.device)
            current_device = next(self.engine.parameters()).device
            self.log.info(f"{self.config['embeddings_model_name']} loaded to device {current_device}")
            
    def get_embeddings_size(self, embeddings_path:str) -> tuple[int,int]:
        file_list = get_files_of_extensions(embeddings_path, ["pt"])
        x:int = 0
        y:int = 0
        for filename in file_list:
            tensor_path = os.path.join(embeddings_path, filename)
            tensor_i = cast(torch.Tensor, torch.load(tensor_path, map_location='cpu'))
            dx, ny = tensor_i.shape
            del tensor_i
            x += dx
            if y==0:
                y=ny
            else:
                if ny != y:
                    self.log.error("Tensor dimensions in axis 1 differ! That can't be! Rebuild index completely!")
                    raise ValueError
        return x,y

    def get_embeddings_matrix(self, model_name:str|None=None) -> tuple[np.typing.NDArray[np.float32], list[tuple[str,int]]]:
        emb_array: np.typing.NDArray[np.float32] = np.array([], dtype=np.float32)
        hashes: list[tuple[str,int]] = []
        if model_name is None:
            model_name = self.config['embeddings_model_name']
        for _ind, model in enumerate(self.model_list):
            if model_name != model['model_name']:
                continue
            if model['enabled'] is True:
                indices_path = self.model_embedding_path(model['model_name'])
                x, y = self.get_embeddings_size(indices_path)
                cx:int= 0
                emb_array = np.zeros((x,y), dtype=np.float32)
                file_list = get_files_of_extensions(indices_path, ["pt"])
                for filename in file_list:
                    hash = os.path.splitext(filename)[0]
                    tensor_path = os.path.join(indices_path, filename)
                    emb_tensor = cast(torch.Tensor, torch.load(tensor_path, map_location='cpu'))
                    dx, _dy = emb_tensor.shape
                    emb_part = cast(np.typing.NDArray[np.float32], emb_tensor.numpy(force=True))
                    emb_array[cx:cx+dx, :] = emb_part
                    cx += dx
                    hashes += [(hash, ind) for ind in range(cast(int, emb_part.shape[0]))]
                self.log.info(f"Tensor size: {x}x{y}, loaded")
        return emb_array, hashes

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
        chunk_start = VectorStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        return text[chunk_start : chunk_start + chunk_size]

    @staticmethod
    def get_chunk_context_aware(text:str, index: int, chunk_size: int, chunk_overlap: int) -> str:
        """Extracts a single text chunk based on index, try to extend to sentence-completness"""
        if not text: 
            return ""
        chunk_start = VectorStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        chunk_end = chunk_start + chunk_size
        if index>0:
            prev_chunk = VectorStore.get_chunk_ptr(index-1, chunk_size, chunk_overlap)
        else:
            prev_chunk = chunk_start
        prev_max = chunk_start - prev_chunk
        if index < VectorStore.get_chunk_count(text, chunk_size, chunk_overlap) - 1:
            next_chunk = VectorStore.get_chunk_ptr(index + 1, chunk_size, chunk_overlap)
        else:
            next_chunk = chunk_end
        next_max = prev_max + chunk_size
        extended_text = text[prev_chunk:next_chunk+chunk_size]
        act_start = prev_max
        for ind in range(prev_max, 0, -1):
            if extended_text[ind] == '.':
                act_start = ind
                while extended_text[act_start] in ['.', ' ']:
                    act_start += 1
                break
        act_end = next_max
        for ind in range(next_max, len(extended_text)):
            if extended_text[ind] == '.':
                act_end = ind+1
                break
        context_text = extended_text[act_start:act_end].strip()
        return context_text

    @staticmethod
    def get_span_chunk(text:str, index: int, count:int, chunk_size: int, chunk_overlap: int):
        """Extracts a span of text covering 'count' base chunks."""
        if not text or count < 1: 
            return ""
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: 
            step = chunk_size
        chunk_start = VectorStore.get_chunk_ptr(index, chunk_size, chunk_overlap)
        chunk_end = chunk_start + chunk_size + step * (count - 1)
        return text[chunk_start : chunk_end]

    @staticmethod
    def get_chunks(text:str, chunk_size: int, chunk_overlap: int):
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

    @staticmethod
    def get_chunk_count(text:str, chunk_size: int, chunk_overlap: int) -> int:
        overlap = min(chunk_overlap, chunk_size - 1)
        step = chunk_size - overlap
        if step <= 0: 
            step = chunk_size
        text_len = len(text)
        count = 0
        for i in range(0, text_len, step):
            chunk = text[i : i + chunk_size]
            if not chunk: 
               break
            count += 1
            if i + chunk_size >= text_len: 
               break
        return count

    def save_tensor(self, tensor:torch.Tensor, filename:str):
        temp_path: str | None = None
        temp_fd: int | None = None
        try:
            (temp_fd, temp_path) = tempfile.mkstemp(dir=os.path.dirname(filename))
            os.close(temp_fd)
            torch.save(tensor, temp_path)
            os.replace(temp_path, filename)
            temp_path = None # Prevent removal
        except (IOError, OSError, RuntimeError) as e:
            self.log.error(f"Error during atomic tensor save to {filename}: {e}")
            if temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError as rm_e: self.log.error(f"Failed to remove temporary tensor file {temp_path} after save error: {rm_e}")
            raise e
        finally:
             if temp_fd is not None:
                 try: os.close(temp_fd)
                 except OSError: pass
             if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.log.warning(f"Removed leftover temp tensor file in finally: {temp_path}")
                except OSError as rm_e: self.log.error(f"Failed to remove leftover temp tensor file {temp_path} in finally: {rm_e}")
        
    def index(self, text_library:dict[str,TextLibraryEntry], progress_callback:Callable[[ProgressState], None ]|None=None, abort_check_callback:Callable[[], bool]|None=None) -> list[str]:
        errors:list[str] = []
        self.load_model()
        if self.model is None or self.engine is None:
            self.log.error("Failed to load model, cannot index!")
            errors.append("Failed to load model, cannot index!")
            return errors
        new_chunks: int = 0
        retries: int = 0
        for hash in text_library:
            filename = self.get_embedding_filename(hash)
            name = text_library[hash]['descriptor']
            if os.path.exists(filename):
                continue
            new_chunks += VectorStore.get_chunk_count(text_library[hash]['text'], self.config['chunk_size'], self.config['chunk_overlap'])

        if new_chunks == 0:
            self.log.info("Index already complete, no new documents found")
            errors.append("Index already complete, no new documents found")
            return errors
        
        current_chunks:int = 0
        start_time = time.time()
        last_status = time.time()
        for hash in text_library:
            if abort_check_callback is not None and abort_check_callback() is True:
                return errors
            filename = self.get_embedding_filename(hash)
            name = text_library[hash]['descriptor']
            if os.path.exists(filename):
                continue
            # self.save_embeddings_tensor(text_library[hash]['text'], filename)
            batch_size = self.config['batch_base_multiplier'] * self.model['batch_multiplier']
            chunks: list[str] = VectorStore.get_chunks(text_library[hash]['text'], self.config['chunk_size'], self.config['chunk_overlap'])
            chunk_batch_size = self.config['chunk_batch_size'] * self.model['batch_multiplier']
            embeddings_tensor: torch.Tensor | None = None
            for ind in range(0, len(chunks), chunk_batch_size):
                perc:float = current_chunks / new_chunks
                current_time:float = time.time()
                delta:float = current_time - start_time
                if delta > 5.0 and perc > 0:
                    # eta_t:float = start_time + delta/perc
                    eta_s:float = delta/perc - delta
                    h = int(eta_s // 3600)
                    m = int((eta_s % 3600) // 60)
                    s = int(eta_s % 60)
                    eta_str = (datetime.datetime.now()+datetime.timedelta(seconds=int(eta_s))).strftime("%H:%M:%S")                
                    eta = f"{eta_str}, in {h}:{m:02d}:{s:02d}"
                else:
                    eta = "calculating..."
                if time.time() - last_status > 1 or current_chunks == new_chunks:
                    state = f"Indexing: {name[-80:]:80s}, eta={eta}"
                    if progress_callback is not None:
                        progress_state = ProgressState({'issues': len(errors), 'state': state, 'percent_completion': perc, 'vars': {}, 'finished': False})
                        progress_callback(progress_state)
                    last_status = time.time()
                sub_chunks = chunks[ind:ind+chunk_batch_size]
                try:
                    embeddings_sub_tensor = cast(torch.Tensor, self.engine.encode_document(sub_chunks, device=str(self.device), convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size, normalize_embeddings=True))  # pyright:ignore[reportUnknownMemberType]
                except torch.OutOfMemoryError as e:
                    self.log.warning(f"Out of Memory at {name}[{ind}]")
                    errors.append(f"Out-of-memory trying to index: {name} with {self.model['model_name']}")
                    retries += 1
                    if retries > self.config['oom_recoveries']:
                        self.log.error(f"Out-of-memory errors for {self.model['model_name']} exceeded oom_recoveries count, aborting: {e}")
                        raise
                    else:
                        if embeddings_tensor is not None:
                            del embeddings_tensor
                        embeddings_tensor = None
                        break
                except:
                    raise
                
                if embeddings_tensor is None:
                    embeddings_tensor = embeddings_sub_tensor
                else:
                    embeddings_tensor = torch.cat((embeddings_tensor, embeddings_sub_tensor), dim=0)
                current_chunks += len(sub_chunks)
            if embeddings_tensor is not None:
                self.save_tensor(embeddings_tensor, filename)
                del embeddings_tensor
                embeddings_tensor = None
        if progress_callback is not None:
            progress_state = ProgressState({'issues': len(errors), 'state': "Index complete", 'percent_completion': 1.0, 'vars': {}, 'finished': True})
            progress_callback(progress_state)
        self.log.info("Index completed")
        return errors

    def index_all(self, text_library:dict[str, TextLibraryEntry], progress_callback:Callable[[ProgressState], None ]|None=None, abort_check_callback:Callable[[], bool]|None=None) -> list[str]:
        errors:list[str] = []
        current_old = self.config['embeddings_model_name']
        current_index = self.get_model_index(current_old)
        for ind in range(len(self.model_list)):
            if self.model_list[ind]['enabled'] is False:
                continue
            name = self.select(ind+1)
            self.log.info(f"{ind+1}., indexing {name}")
            errors += self.index(text_library, progress_callback, abort_check_callback)
        if current_index is not None:
            name = self.select(current_index)
            self.log.info(f"Reactivated {name} after indexing all.")
        return errors
    
    def get_significance(self, text: str, search_tensor: torch.Tensor, context_length: int, context_steps: int, cutoff:float=0.0) -> list[float]:
        clr: list[str] = []
        if self.model is None:
            self.log.error("No active model")

            return []
        batch_size = self.config['batch_base_multiplier'] * self.model['batch_multiplier']
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

        if self.engine is None:
            raise ValueError
        context_tensor: torch.Tensor = cast(torch.Tensor, self.engine.encode_document(clr, device=str(self.device), convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size, normalize_embeddings=True))  # pyright:ignore[reportUnknownMemberType]
        cosines: list[float] = cast(list[float], self.engine.similarity(context_tensor, search_tensor).reshape((-1,)).tolist())  # pyright:ignore[reportUnknownMemberType]

        min_cos = 1.0
        max_cos = 0.0
        for cos in cosines:
            if cos < min_cos:
                min_cos = cos
            if cos > max_cos:
                max_cos = cos
        if max_cos - min_cos > 0.0:
            for ind, cos in enumerate(cosines):
                cosines[ind] = (cos - min_cos) / (max_cos - min_cos)

            for ind, cos in enumerate(cosines):
                cosines[ind] = (math.exp(cos) - 1) / (math.exp(1) - 1)
                if cosines[ind] < cutoff:
                    cosines[ind] = 0.0
        return cosines

    def search(self, search_text:str, text_library:dict[str,TextLibraryEntry], max_results:int=10, highlight:bool=False,
               highlight_cutoff:float=0.0, highlight_dampening:float=1.0, context_length:int=16, context_steps:int=4) -> list[SearchResultEntry]:
        self.load_model()
        if self.model is None or self.engine is None:
            self.log.error("Failed to load model, cannot index!")
            return []
        start_time = time.time()
        search_results: list[SearchResultEntry] = []
        device = torch.device(self.resolve_device())
        search_tensor = cast(torch.Tensor, self.engine.encode_query(search_text, device=str(self.device), convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True))  # pyright:ignore[reportUnknownMemberType]
        path = self.model_embedding_path(self.model['model_name'])
        tensor_file_list = get_files_of_extensions(path, ['pt'])
        best_min_cosine: float | None = None
        for tensor_file in tensor_file_list:
            tensor_path = os.path.join(path, tensor_file)
            hash = os.path.splitext(tensor_file)[0]
            tensor:torch.Tensor = cast(torch.Tensor, torch.load(tensor_path, map_location=device))
            # Cosines = torch.matmul(search_tensor, tensor.T).T
            cosines = self.engine.similarity(tensor, search_tensor)
            max_ind:int = int(torch.argmax(cosines).item())
            cosine:float = cosines[max_ind].item()
            if best_min_cosine is None or cosine > best_min_cosine:
                search_results.append(SearchResultEntry({'cosine': cosine, 'hash': hash, 'chunk_index': max_ind, 'entry': text_library[hash], 'text': None, 'significance': None}))
                search_results = sorted(search_results, key=lambda res: res['cosine'])
                search_results = search_results[-max_results:]
                best_min_cosine = search_results[0]['cosine']
        search_time = time.time() - start_time
        key = f"{self.config['embeddings_model_name']}-{str(self.device)}"
        self.perf[key] = search_time
        highlight_start = time.time()
        for index, result in enumerate(search_results):
            result_text = self.get_chunk_context_aware(result['entry']['text'], result['chunk_index'], self.config['chunk_size'], self.config['chunk_overlap'])
            replacers = [("\n", " "), ("\r", " "), ("\b", " "), ("\t", " "), ("  ", " ")]
            zero_width = [("\u200b", ""), ("\u200c", ""), ("\u200d", ""), ("\u00ad", ""), ("\ufeff", "")] 
            replacers += zero_width
            old_text = ""
            while old_text != result_text:
                old_text = result_text
                for rep in replacers:
                    result_text = result_text.replace(rep[0], rep[1])

            search_results[index]['text'] = result_text

            if highlight is True:
                significance: list[float] = [0.0] * len(result_text)
                stepped_significance: list[float] = self.get_significance(result_text, search_tensor, context_length, context_steps, highlight_cutoff)
                if highlight_dampening == 0.0:
                    self.log.error("Dampending must not be zero!")
                    highlight_dampening = 1.0
                for ind in range(len(result_text)):
                    significance[ind] = stepped_significance[ind // context_steps] * result['cosine'] / highlight_dampening
                search_results[index]['significance'] = significance
        
        if len(search_results) > 0:
            highlight_time = (time.time() - highlight_start) / len(search_results)
            key = f"{self.config['embeddings_model_name']}-{str(self.device)} highlight time per record"
            self.perf[key] = highlight_time
        return search_results
    
    def prepare_visualization_data(self, text_library:dict[str, TextLibraryEntry], max_points: int | None = None) -> dict[str, Any]:  # pyright:ignore[reportExplicitAny]
        matrix, hashes = self.get_embeddings_matrix()
        if umap is None:
            return {"error": "UMAP module is not available"}
        
        self.log.info(f"Loaded matrix {matrix.shape}, hash-count: {len(hashes)}")
        num_available_points = cast(int, matrix.shape[0])
        if max_points is not None:
            matrix = matrix[:max_points, :]
            hashes = hashes[:max_points]
            self.log.info(f"Reduced by max_points {max_points}, matrix {matrix.shape}, hash-count: {len(hashes)}")
        # total_mem:int = psutil.virtual_memory().total
        # if total_mem > 134809341952 // 2:
        #     low_mem = False
        #     self.log.info("High memory usage ok")
        # else:
        low_mem = True
        n_neighbors_val = 5
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors_val, 
                                min_dist=0.1, metric='cosine', low_memory=low_mem)
                                # min_dist=0.1, random_state=42, metric='cosine')
        try:
            self.log.info("Starting UMAP reducer")
            reduced_embeddings_np = cast(np.typing.NDArray[np.float32], reducer.fit_transform(matrix))  # pyright:ignore[reportUnknownMemberType]
            self.log.info("Reducer finished, compiling results")
        except Exception as e:
            self.log.error(f"UMAP reduction failed: {e} (shape: {matrix.shape}, n_neighbors: {n_neighbors_val})")
            return {"error": f"UMAP reduction failed: {str(e)}"}

        points:list[list[float]] = cast(list[list[float]], reduced_embeddings_np.tolist())
        texts = [text_library[hash]['descriptor']+f"[{chunk_id}]" for hash, chunk_id in hashes]
        doc_ids = [text_library[hash]['descriptor'] for hash, _chunk_id in hashes]

        unique_doc_ids = list(set(doc_ids))

        num_unique_docs = len(unique_doc_ids)
        color_map: dict[str, list[int]] = {}
        for i, doc_id_val in enumerate(unique_doc_ids):
            hue = i / num_unique_docs if num_unique_docs > 0 else 0
            rgb_float = colorsys.hls_to_rgb(hue, 0.5, 0.8) 
            color_map[doc_id_val] = [int(c * 255) for c in rgb_float]

        colors: list[list[int]] = [color_map.get(text_library[hash]['descriptor'], [128,128,128]) for hash, _chunk_id in hashes]
        sizes = [5.0] * len(points)
        self.log.info("UMAP finished")

        return {
            "points": points,
            "texts": texts,
            "colors": colors,
            "sizes": sizes,
            "doc_ids": doc_ids,
            "model_name": self.config['embeddings_model_name'],
            "reduction_method": "UMAP",
            "num_points_visualized": len(points),
            "num_points_available_before_sampling": num_available_points
        }


    def index3d(self, text_library:dict[str, TextLibraryEntry], max_points:int|None=None):
        point_cloud = self.prepare_visualization_data(text_library, max_points)
        filename = os.path.join(self.visualization_3d, self.config['embeddings_model_name']+'.json')
        with open(filename, "w") as f:
            json.dump(point_cloud, f, indent=2)
    
    def index3d_all(self, text_library:dict[str, TextLibraryEntry], max_points:int|None=None):
        current_old = self.config['embeddings_model_name']
        current_index = self.get_model_index(current_old)
        for ind in range(len(self.model_list)):
            if self.model_list[ind]['enabled'] is False:
                continue
            name = self.select(ind+1)
            self.log.info(f"{ind+1}., 3D-indexing {name}")
            self.index3d(text_library, max_points)
        if current_index is not None:
            name = self.select(current_index)
            self.log.info(f"Reactivated {name} after indexing all.")
        
    
class DocumentStore:
    def __init__(self):
        self.current_version: int = 5
        self.log: logging.Logger = logging.getLogger("DocumentStore")
        self.md_tools:dict[str, MarkdownTools] = {}
        self.org_tools:dict[str, OrgmodeTools] = {}
        self.cb_tools:dict[str, CalibreTools] = {}
        self.config_changed:bool = False
        self.config_path: str = os.path.expanduser("~/.config/local_research")
        if os.path.isdir(self.config_path) is False:
            os.makedirs(self.config_path)            
        self.sha256_cache: dict[str, Sha256CacheEntry] = {}
        self.sha256_cache_filename: str = os.path.join(self.config_path, 'sha256_cache.json')
        self.sha256_cache_changed: bool = False
        self.config_file:str = os.path.join(self.config_path, "document_store.json")
        self.valid_source_types: list[str] = ['calibre', 'md_notes', 'orgmode']
        self.config: DocumentConfig = self.get_config()
        self.text_library: dict[str, TextLibraryEntry] = {}
        self.metadata_library: dict[str, MetadataEntry]
        self.pdf_index:dict[str, PDFIndex] = {}
        self.perf: dict[str, float] = {}

        self.publish_path: str = os.path.expanduser(self.config['publish_path'])
        if os.path.isdir(self.publish_path) is False:
            os.makedirs(self.publish_path)

        self.storage_path: str = os.path.join(os.path.expanduser("~/.local/share"), "local_research")
        if os.path.isdir(self.storage_path) is False:
            os.makedirs(self.storage_path)
        self.text_document_library_file:str = os.path.join(self.storage_path, "document_library.json")
        self.metadata_library_file:str = os.path.join(self.storage_path, "metadata_library.json")
        self.sequence_file:str = os.path.join(self.storage_path, "version_seq.json")
        self.remote_sequence_file:str = os.path.join(self.publish_path, "version_seq.json")
        self.pdf_cache_path: str = os.path.join(self.storage_path, "pdf_cache")
        if os.path.isdir(self.pdf_cache_path) is False:
            os.makedirs(self.pdf_cache_path, exist_ok=True)
        self.pdf_index_file: str = os.path.join(self.pdf_cache_path, "pdf_index.json")

        self.load_metadata_library()
        self.load_text_library()
        remote, local = self.load_sequence_versions()
        self.log.info(f"DocumentStore initialized: remote data version: {remote}, local version: {local}")
        if self.local_update_required() is True:
            self.log.info("Please use 'import' to acquire the latest data version")

    def get_config(self) -> DocumentConfig:
        valid = False
        config: DocumentConfig | None = None
        if os.path.exists(self.sha256_cache_filename):
            with open(self.sha256_cache_filename, 'r') as f:
                self.sha256_cache = json.load(f)
                self.sha256_cache_changed = False
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config = cast(DocumentConfig, json.load(f))
                    valid = True
                    if config['version'] < self.current_version:
                        self.log.error(f"Config file {self.config_file} is outdated version, upgrading to new defaults!")
                        valid = False
                    else:
                        for source_name in config['document_sources']:
                            source = config['document_sources'][source_name]
                            if source['type'] not in self.valid_source_types:
                                self.log.error(f"{source_name} has invalid type {source['type']}, use one of {self.valid_source_types}")
                                valid = False
                                break                        
            except Exception as e:
                self.log.error(f"Failed to read config-file {self.config_file}: {e}, resetting to default configuration!")
                valid = False

        if valid is False or config is None:
            config = DocumentConfig({
                'version': self.current_version,
                'document_sources': {
                    'Calibre': DocumentSource({
                        'type': 'calibre',
                        'path': '~/ReferenceLibrary/Calibre Library',
                        'file_types': ['txt', 'pdf']
                    }),
                    'Notes': DocumentSource({
                        'type': 'md_notes',
                        'path': '~/Notes',
                        'file_types': ['md']
                    }),
                    'Orgmode': DocumentSource({
                        'type': 'orgmode',
                        'path': '~/OrgNotes',
                        'file_types': ['org']
                        })
                },
                'publish_path': '~/LocalResearch',
                'vars': {
                    'search_results': ("3", "int"),
                    'highlight': ("true", "bool"),
                    'highlight_cutoff': ("0.3", "float"),
                    'highlight_dampening': ("1.2", "float"),
                    'context_length': ("16", "int"),
                    'context_steps': ("4", "int"),
                    }
                })
            self.save_config(config)
            self.log.warning(f"Default configuration created at {self.config_file}, please review!")
        self.config_changed = False
        for source_name in config['document_sources']:
            if config['document_sources'][source_name]['type'] == 'md_notes':
                self.md_tools[source_name] = MarkdownTools(config['document_sources'][source_name]['path'])
            elif config['document_sources'][source_name]['type'] == 'orgmode':
                self.org_tools[source_name] = OrgmodeTools(config['document_sources'][source_name]['path'])
            elif config['document_sources'][source_name]['type'] == 'calibre':
                self.cb_tools[source_name] = CalibreTools(config['document_sources'][source_name]['path'])
            else:
                self.log.warning(f"Unexpected type for document source {source_name}")
        return config

    def get_sha256(self, filename:str, cache:bool = True):
        if cache is True:
            if filename in self.sha256_cache:
                cached = self.sha256_cache[filename]
                stat = os.stat(filename)
                dt = stat.st_mtime
                size = stat.st_size
                if cached['size'] == size and cached['modified'] == dt:
                    return cached['sha256']
                    
        with open(filename, 'rb', buffering=0) as f:
            hash = hashlib.file_digest(f, 'sha256').hexdigest()
            if cache is True:
                stat = os.stat(filename)
                dt = stat.st_mtime
                size = stat.st_size
                cached: Sha256CacheEntry = Sha256CacheEntry({'size': size, 'modified': dt, 'sha256':hash})
                self.sha256_cache[filename] = cached
                self.sha256_cache_changed = True
            return hash

    def save_sha256_cache(self):
        if self.sha256_cache_changed is False:
            return
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.sha256_cache_filename))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(self.sha256_cache, temp_file)
            os.replace(temp_path, self.sha256_cache_filename)  # atomic update
            self.sha256_cache_changed = False
        except Exception as e:
            self.log.error("Sha256Cache-file update interrupted, not updated.")
            os.remove(temp_path)
            raise e
            
    def save_config(self, config: DocumentConfig):
        self.save_sha256_cache()
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.config_file))
        config['version'] = self.current_version
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(config, temp_file)
            os.replace(temp_path, self.config_file)  # atomic update
        except Exception as e:
            self.log.error("Config-file update interrupted, not updated.")
            os.remove(temp_path)
            raise e

    def set_var(self, name:str, value:str) -> bool:
        if name not in self.config['vars']:
            self.log.error(f"Unknown config variable '{name}', use 'list vars' for possible names")
            return False
        val, type = self.config['vars'][name]
        if type == 'int':
            try:
                _v = int(value)
            except ValueError:
                self.log.error(f"{name} must be of type 'int'")
                return False
            except Exception as e:
                self.log.error(f"{name} of type int caused: {e}")
                return False
        elif type == 'float':
            try:
                _v = float(value)
            except ValueError:
                self.log.error(f"{name} must be of type 'float'")
                return False
            except Exception as e:
                self.log.error(f"{name} of type float caused: {e}")
                return False
        elif type == 'bool':
            value = value.lower()
            if value not in ['true', 'false']:
                self.log.error(f"{name} must be of type 'bool'")
                return False
        else:
            self.log.error(f"{name} has type {type} which is not implemented")
            return False
        self.config['vars'][name] = (value, type)
        if val != value:
            self.save_config(self.config)
        return True

    def get_var(self, name:str, local_vars:dict[str,str]|None=None) -> bool|int|float|str|None:
        if name not in self.config['vars']:
            self.log.error(f"Unknown config variable '{name}', use 'list vars' for possible names")
            return False
        val, type = self.config['vars'][name]
        if local_vars is not None and name in local_vars:
            val = local_vars[name]
        if type == 'int':
            try:
                v = int(val)
                return v
            except ValueError:
                self.log.error(f"{name} must be of type 'int'")
                return None
            except Exception as e:
                self.log.error(f"{name} of type int caused: {e}")
                return None
        elif type == 'float':
            try:
                v = float(val)
                return v
            except ValueError:
                self.log.error(f"{name} must be of type 'float'")
                return None
            except Exception as e:
                self.log.error(f"{name} of type float caused: {e}")
                return None            
        elif type == 'bool':
            val = val.lower()
            if val not in ['true', 'false']:
                self.log.error(f"{name} must be of type 'bool'")
                return None
            if val == 'true':
                return True
            else:
                return False
        else:
            self.log.error(f"{name} has type {type} which is not implemented")
            return None
        
    def load_sequence_versions(self) -> tuple[int,int]:
        try:
            with open(self.remote_sequence_file, 'r') as f:
                remote_sequence: SequenceVersion = cast(SequenceVersion, json.load(f))
        except Exception as e:
            self.log.warning(f"Failed to get remote data version: {e}")
            remote_sequence = SequenceVersion({'sequence': 0})
        try:
            with open(self.sequence_file, 'r') as f:
                local_sequence: SequenceVersion = cast(SequenceVersion, json.load(f))
        except:
            local_sequence = SequenceVersion({'sequence': 1})
        return (remote_sequence['sequence'], local_sequence['sequence'])

    def write_local_sequence_version(self, seq_no: int):
        local_sequence = SequenceVersion({'sequence': seq_no})
        with open(self.sequence_file, 'w') as f:
            json.dump(local_sequence, f)

    def local_update_required(self) -> bool:
        remote, local = self.load_sequence_versions()
        if remote > local:
            return True
        else:
            return False

    def get_source_name_from_path(self, path:str) -> str|None:
        full_path = os.path.expanduser(path)
        for source in self.config['document_sources']:
            if full_path.startswith(os.path.expanduser(self.config['document_sources'][source]['path'])):
                return source
        return None

    def get_descriptor_from_path(self, path:str, source_name: str|None=None) ->str:
        full_path = os.path.expanduser(path)
        if source_name is None:
            source_name = self.get_source_name_from_path(full_path)
            if source_name is None:
                self.log.error(f"Path {full_path} is not within a defined source!")
                return full_path
        if source_name not in self.config['document_sources']:
            self.log.error(f"Invalid source_name {source_name} referenced for {full_path}")
            return full_path
        source_path = os.path.expanduser(self.config['document_sources'][source_name]['path'])
        if full_path.startswith(source_path):
            descriptor = "{" + source_name + "}" + full_path[len(source_path):]
        else:
            self.log.error(f"Path {full_path} is not within source {source_name}")
            return full_path
        return descriptor

    def get_source_name_and_path_from_descriptor(self, descriptor: str) -> tuple[str,str]:
        if len(descriptor)<3:
            self.log.error(f"Expression >{descriptor}< is not a valid descriptor")
            return ("", descriptor)
        if descriptor[0] != '{':
            self.log.error(f"Descriptor >{descriptor}< must start with " + "{")
            return ("", descriptor)
        ind = descriptor.find('}')
        if ind == -1:
            self.log.error(f"Descriptor >{descriptor}< name must end with " +"}")
            return ("", descriptor)
        source_name = descriptor[1:ind]
        if source_name not in self.config['document_sources']:
            self.log.error(f">{source_name}< is not a valid source in {descriptor}")
            return "", descriptor
        full_path = os.path.join(os.path.expanduser(self.config['document_sources'][source_name]['path']), descriptor[ind+1:])
        return source_name, full_path
     
    def get_path_from_descriptor(self, descriptor:str) -> str:
        source, full_path = self.get_source_name_and_path_from_descriptor(descriptor)
        if source == "":
            return descriptor
        return full_path
    
    def load_text_library(self):
        self.log.info("Loading text_library data...")
        if os.path.exists(self.text_document_library_file):
            start_time = time.time()
            with open(self.text_document_library_file, "r") as f:
                self.text_library = json.load(f)
            if len(self.text_library.keys()) > 0:
                delta = (time.time() - start_time) / len(self.text_library.keys()) * 1000.0
                self.perf['load text library (1000 recs)'] = delta
                
        else:
            self.text_library = {}
        if os.path.exists(self.pdf_index_file):
            start_time = time.time()
            with open(self.pdf_index_file, "r") as f:
                self.pdf_index = json.load(f)
            if len(self.pdf_index.keys()) > 0:
                delta = (time.time() - start_time) / len(self.pdf_index.keys()) * 1000000.0
                self.perf['load pdf cache (10^6 recs)'] = delta
        else:
            self.pdf_index = {}
            
#        upgraded = False
#        for entry in self.text_library:
#            if self.text_library[entry]['descriptor'].startswith('{') is False:
#                descriptor = self.get_descriptor_from_path(self.text_library[entry]['descriptor'], self.text_library[entry]['source_name'])
#                self.text_library[entry]['descriptor'] = descriptor
#                upgraded = True
#        if upgraded is True:
#            self.save_text_library()
#        else:

    def load_metadata_library(self):
        self.log.info("Loading metadata_library data...")
        if os.path.exists(self.metadata_library_file):
            with open(self.metadata_library_file, "r") as f:
                self.metadata_library = json.load(f)
        else:
            self.metadata_library = {}

    def save_text_library(self):
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.text_document_library_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(self.text_library, temp_file)
            os.replace(temp_path, self.text_document_library_file)               
        except Exception as e:
            self.log.error("Library update was interrupted, atomic file update cancelled.")
            os.remove(temp_path)
            raise e

        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.pdf_index_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(self.pdf_index, temp_file)
            os.replace(temp_path, self.pdf_index_file)               
        except Exception as e:
            self.log.error("PDF Index update was interrupted, atomic file update cancelled.")
            os.remove(temp_path)
            raise e
        self.log.info("Library data saved")

    def save_metadata_library(self):
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.metadata_library_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(self.metadata_library, temp_file)
            os.replace(temp_path, self.metadata_library_file)               
        except Exception as e:
            self.log.error("Metadata library update was interrupted, atomic file update cancelled.")
            os.remove(temp_path)
            raise e
        
    def get_pdf_cache_filename(self, sha256_hash:str) -> str:
        basename = sha256_hash+".txt"
        return os.path.join(self.pdf_cache_path, basename)
    
    def get_pdf_text(self, full_path:str, sha256_hash: str, retry:bool=False) -> tuple[str | None, bool]:
        pdf_text: str | None = None
        index_changed: bool = False
        current_file_size: int = -1

        if os.path.exists(full_path) is False:
            self.log.error(f"Cannot process PDF file {full_path}, file does not exist!")
            return None, False
        current_file_size = os.path.getsize(full_path)
        
        if sha256_hash in self.pdf_index:
            cached_info = self.pdf_index[sha256_hash]
            if cached_info['previous_failure'] is True and retry is False:
                self.log.debug(f"Skipping PDF {full_path}: previously failed extraction.")
                return None, False
            else:
                pdf_cache_filename = self.get_pdf_cache_filename(sha256_hash)
                if os.path.exists(pdf_cache_filename):
                    try:
                        with open(pdf_cache_filename, 'r', encoding='utf-8') as f:
                            pdf_text = f.read()
                        return pdf_text, True # Return cached text, index not changed
                    except Exception as e:
                        self.log.warning(f"Failed to read PDF cache file {pdf_cache_filename} for {full_path}: {e}. Re-extracting.")
                else:
                    self.log.warning(f"PDF cache index points to non-existent file {pdf_cache_filename} for {full_path}. Re-extracting.")

        if pdf_text is None:
            extracted_text: str | None = None
            try:
                doc = pymupdf.open(full_path)
                extracted_pages = []
                for page_num, page in enumerate(doc):  # pyright:ignore[reportArgumentType, reportUnknownVariableType]
                    try:
                        page_text = page.get_text()  # pyright:ignore[reportUnknownMemberType, reportUnknownVariableType]
                        if isinstance(page_text, str) and len(page_text)>7 and page_text.strip()!='-----': 
                            extracted_pages.append(page_text)  # pyright: ignore[reportUnknownMemberType]
                    except Exception as page_e: 
                        self.log.warning(f"Failed extraction on page {page_num+1} of {full_path}: {page_e}")
                doc.close()
                combined_text = "\n".join(extracted_pages)  # pyright:ignore[reportUnknownArgumentType]
                combined_text = combined_text.replace('-----\n', '')
                if combined_text.strip() != "" and len(combined_text.strip()) > 7:
                    extracted_text = combined_text
            except Exception as e:
                self.log.info(f"Failed pymupdf extraction for {full_path}: {e}", exc_info=True)
                
            if extracted_text is None:
                try:
                    md_text = pymupdf4llm.to_markdown(full_path, ignore_images=True)  # pyright:ignore[reportUnknownMemberType]
                    md_text= md_text.replace('-----\n', '')
                    if md_text.strip() != "":
                        extracted_text = md_text # Use the markdown text
                    else:
                        self.log.info(f"pymupdf4llm text empty for {full_path}")
                except Exception as e:
                    self.log.error(f"Failed pymupdf4llm fallback extraction for {full_path}: {e}", exc_info=True)

            if extracted_text is None:
                if sha256_hash in self.pdf_index:
                    self.pdf_index[sha256_hash]["previous_failure"] = True
                    index_changed = True
                else:
                    cache_entry = PDFIndex({'file_size': current_file_size, 'previous_failure': True})
                    self.pdf_index[sha256_hash] = cache_entry
                    index_changed = True
                return None, index_changed
            else:
                pdf_text = extracted_text

            cache_entry = PDFIndex({
                    'file_size': current_file_size,
                    'previous_failure': False
                })
            self.pdf_index[sha256_hash] = cache_entry

            pdf_cache_filename = self.get_pdf_cache_filename(sha256_hash)
            with open(pdf_cache_filename, 'w') as f:
                _ = f.write(pdf_text)
            index_changed = True

        return pdf_text, index_changed

    def skip_file_in_sync(self, root_path:str, filename:str, valid_types:list[str]) -> bool:
        _base, ext_with_dot = os.path.splitext(filename)
        ext = ext_with_dot[1:].lower() if ext_with_dot else ""
        if '.caltrash' in root_path:
            return True
        if ext not in valid_types: 
            return True
        return False
    
    def skip_file_in_sync_alternate(self, root_path:str, filename:str, valid_types:list[str]) -> bool:
        base, ext_with_dot = os.path.splitext(filename)
        ext = ext_with_dot[1:].lower() if ext_with_dot else ""
        if '.caltrash' in root_path:
            return True
        if ext not in valid_types: 
            return True
        preferred_order: list[str] = ['txt', 'md', 'org']
        preferred_ext_exists:bool = False
        if ext == 'pdf':
             for pref_ext in preferred_order:
                 if os.path.exists(os.path.join(root_path, base + '.' + pref_ext)):
                     preferred_ext_exists = True
                     break
        if preferred_ext_exists:
            return True
        return False

    def get_source_type_from_name(self, source_name:str) -> str|None:
        if source_name not in self.config['document_sources']:
            return None
        return self.config['document_sources'][source_name]['type']
    
    def sync_texts(self, force:bool, retry:bool=False, progress_callback:Callable[[ProgressState], None ]|None=None, abort_check_callback:Callable[[], bool]|None=None) -> list[str]:
        errors:list[str] = []
        tables:list[DocumentTable] = []
        if self.local_update_required() is True:
            if force is False:
                self.log.warning("Please first update local data using 'import', since remote has newer data! (use 'force' to override)")
                errors.append("Please first update local data using 'import', since remote has newer data! (use 'force' to override)")
                return errors
            else:
                self.log.warning("Override active, syncing even so remote has newer data!")
        text_library_changed = False
        metadata_library_changed = False
        pdf_index_changed = False
        old_text_library_size = len(list(self.text_library.keys()))
        last_saved = time.time()

        existing_hashes: list[str] = list(self.text_library.keys())
        pdf_cache_hits = 0
        duplicate_count = 0
        last_status = time.time()

        doc_count = 0
        current_doc_count = 0
        source_file_count:dict[str,int]={}

        hash_cache:dict[str, str] = {}
        
        for source_name in self.config['document_sources']:
            source = self.config['document_sources'][source_name]
            if abort_check_callback is not None and abort_check_callback() is True: 
                return errors
            source_path = os.path.expanduser(source['path'])
            self.log.info(f"Scanning source '{source_name}' at '{source_path}'...")
            source_file_count[source_name] = 0
            if source['type'] in ['md_notes', 'orgmode']:
                for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: errors.append(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                    for filename in files:
                        if self.skip_file_in_sync(root, filename, source['file_types']) is True:
                            continue
                        source_file_count[source_name] += 1
                        doc_count += 1
                        doc_path = os.path.join(root, filename)
                        try:
                            with open(doc_path, 'r') as f:
                                text = f.read()
                        except Exception as e:
                            self.log.error(f"Failed to read {doc_path}, {e}")
                            text = ""
                        descriptor = self.get_descriptor_from_path(doc_path)
                        if descriptor in hash_cache:
                            sha256_hash = hash_cache[descriptor]
                        else:
                            sha256_hash = self.get_sha256(doc_path)
                            hash_cache[descriptor] = sha256_hash
                        md_content: str|None = None
                        metadata: MetadataEntry | None = None
                        if descriptor not in self.metadata_library:
                            if source['type'] == 'md_notes':
                                metadata, md_content, _meta_changed, mandatory_changed = self.md_tools[source_name].parse_markdown(doc_path, sha256_hash, descriptor, text)
                            elif source['type'] == 'orgmode':
                                metadata, _prefix, _content, _meta_changed, mandatory_changed = self.org_tools[source_name].parse_orgmode(doc_path, sha256_hash, descriptor, text)
                            else:
                                self.log.error(f"Invalid type {source['type']} at {descriptor}")
                                continue                                    
                            if mandatory_changed:
                                self.log.info(f"Markdown doc {doc_path} requires frontmatter update")
                                metadata_library_changed = True
                            self.metadata_library[descriptor] = metadata
                        else:
                            for doc_repr in self.metadata_library[descriptor]['representations']:
                                if doc_repr['doc_descriptor'] == descriptor and doc_repr['hash'] != sha256_hash:
                                    self.log.warning(f"{descriptor} has changed!")
                                    if source['type'] == 'md_notes':
                                        metadata, md_content, _meta_changed, mandatory_changed = self.md_tools[source_name].parse_markdown(doc_path, sha256_hash, descriptor, text)
                                    elif source['type'] == 'orgmode':
                                        metadata, _prefix, _content, _meta_changed, mandatory_changed = self.org_tools[source_name].parse_orgmode(doc_path, sha256_hash, descriptor, text)
                                    else:
                                        self.log.error(f"Invalid type {source['type']} at {descriptor}")
                                        continue                                    
                                    metadata_library_changed = True
                                    self.metadata_library[descriptor] = metadata
                        if md_content is not None and metadata is not None:
                            uuid:str = metadata['uuid']
                            tables += self.md_tools[source_name].get_tables(md_content, doc_path, uuid)
            elif source['type'] == 'calibre':
                for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: errors.append(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                    for filename in files:
                        if filename == 'metadata.opf':
                            filepath = os.path.join(root, filename)
                            metadata = self.cb_tools[source_name].parse_calibre_metadata(filepath)
                            if metadata is not None:
                                main_descriptor = self.get_descriptor_from_path(root)
                                if main_descriptor not in self.metadata_library:
                                    self.metadata_library[main_descriptor] = metadata
                                else:
                                    ### XXX Changed?
                                    pass
                                source_file_count[source_name] += 1
                                doc_count += 1
            else:
                self.log.error(f"Ignoring unknown source_type {source['type']}")
                continue
        
        for source_name in self.config['document_sources']:
            source = self.config['document_sources'][source_name]
            if abort_check_callback is not None and abort_check_callback() is True: 
                break
            source_path = os.path.expanduser(source['path'])
            self.log.info(f"Scanning source '{source_name}' at '{source_path}'...")
            file_count = 0
            for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: self.log.warning(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                if abort_check_callback is not None and abort_check_callback() is True: 
                    break
                for filename in files:
                    if abort_check_callback is not None and abort_check_callback() is True: 
                        break
                    if self.skip_file_in_sync_alternate(root, filename, source['file_types']) is True:
                        continue  ### XXX change, once metadata is primary ref

                    file_count += 1
                    current_doc_count += 1
                    full_path = os.path.join(root, filename)
                    descriptor = self.get_descriptor_from_path(full_path, source_name)
                    if descriptor in hash_cache:
                        sha256_hash = hash_cache[descriptor]
                    else:
                        sha256_hash = self.get_sha256(full_path)
                        hash_cache[descriptor] = sha256_hash
                    ext_with_dot = os.path.splitext(filename)[1]
                    ext = ext_with_dot[1:].lower() if ext_with_dot else ""

                    if sha256_hash in self.text_library and descriptor != self.text_library[sha256_hash]['descriptor']:
                        self.log.warning(f"File {full_path} is a duplicate of {self.text_library[sha256_hash]['descriptor']}, ignoring this copy.")
                        errors.append(f"File {full_path} is a duplicate of {self.text_library[sha256_hash]['descriptor']}, ignoring this copy.")
                        duplicate_count += 1
                        continue

                    if sha256_hash in existing_hashes:
                        existing_hashes.remove(sha256_hash)
                        if time.time() - last_status > 1 or file_count==source_file_count[source_name]:
                            perc = 0.0
                            if doc_count > 0.0:
                                perc = current_doc_count / doc_count
                            state = f"Checking source: {source_name}"
                            if progress_callback is not None:
                                progress_state = ProgressState({'issues': len(errors), 'state': state, 'percent_completion': perc, 'vars': {}, 'finished': False})
                                progress_callback(progress_state)
                            last_status = time.time()
                        continue

                    current_text: str | None = None
                    
                    if ext in ['md', 'txt', 'org']:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            current_text = f.read()
                        if time.time() - last_status > 1 or file_count==source_file_count[source_name]:
                            perc = 0.0
                            if doc_count > 0.0:
                                perc = current_doc_count / doc_count
                            state = f"{current_doc_count}/{doc_count} | {full_path[-80:]:80s} Text"
                            if progress_callback is not None:
                                progress_state = ProgressState({'issues': len(errors), 'state': state, 'percent_completion': perc, 'vars': {}, 'finished': False})
                                progress_callback(progress_state)
                            last_status = time.time()
                    elif ext == 'pdf':
                        current_text, pdf_index_changed_during_get = self.get_pdf_text(full_path, sha256_hash, retry)
                        if pdf_index_changed_during_get is True:
                            pdf_index_changed = True
                        else:
                            pdf_cache_hits += 1
                        if time.time() - last_status > 1 or file_count==source_file_count[source_name]:
                            perc = 0.0
                            if doc_count > 0.0:
                                perc = current_doc_count / doc_count * 100.0
                            state = f"{current_doc_count}/{doc_count} | {full_path[-80:]:80s} PDF "
                            if progress_callback is not None:
                                progress_state = ProgressState({'issues': len(errors), 'state': state, 'percent_completion': perc, 'vars': {}, 'finished': False})
                                progress_callback(progress_state)
                            last_status = time.time()
                    else:
                        self.log.error(f"Handling for file-type {ext} not implemented!")
                        errors.append(f"Handling for file-type {ext} not implemented!")

                    if current_text is None:
                        if sha256_hash in self.pdf_index and self.pdf_index[sha256_hash]['previous_failure'] is True and retry is False:
                            if sha256_hash in existing_hashes:
                                existing_hashes.remove(sha256_hash)
                            continue
                        self.log.error(f"{full_path} has no content or doesnt exist!")
                        errors.append(f"{full_path} has no content or doesnt exist!")
                        if sha256_hash in existing_hashes:
                            existing_hashes.remove(sha256_hash)
                        continue
                                            
                    self.text_library[sha256_hash] = TextLibraryEntry({'source_name': source_name, 'descriptor': descriptor, 'text': current_text})
                    text_library_changed = True

                    if time.time() - last_saved > 180:
                        self.save_text_library()
                        text_library_changed = False
                        last_saved =  time.time()
            if progress_callback is not None:
                progress_state = ProgressState({'issues': len(errors), 'state': "", 'percent_completion': 1.0, 'vars': {}, 'finished': True})
                progress_callback(progress_state)
        new_text_library_size = len(list(self.text_library.keys()))
        if duplicate_count > 0:
            self.log.warning(f"{duplicate_count} duplicates were ignored during import, please re-run sync.")
            errors.append(f"{duplicate_count} duplicates were ignored during import, please re-run sync.")
            
        if len(existing_hashes) > 0:
            self.log.warning(f"{len(existing_hashes)} debris entries")
            for debris in existing_hashes:
                if debris in self.text_library:
                    self.log.info(f"Deleting text_library entry {self.text_library[debris]['descriptor']}")
                    errors.append(f"Deleting debris text_library entry {self.text_library[debris]['descriptor']}")
                    del self.text_library[debris]
                    text_library_changed = True
                if debris in self.pdf_index:
                    del self.pdf_index[debris]
                    pdf_index_changed = True
            self.log.warning("Please use 'check clean' to remove superflucious indices")
            errors.append("Please use 'check clean' to remove superflucious indices")

        self.save_sha256_cache()
        if metadata_library_changed is True:
            self.save_metadata_library()
        if text_library_changed is True or pdf_index_changed is True:
            self.save_text_library()
            self.log.info(f"Library size {old_text_library_size} -> {new_text_library_size}")
        else:
            self.log.info(f"No changes")
        return errors

    def check_pdf_cache(self, clean:bool) -> tuple[int, int, int, int, int, int , int, bool]:
        failure_count = 0
        entry_count = 0
        orphan_count = 0
        orphan2_count = 0
        missing_count = 0
        deleted_count = 0
        deleted2_count = 0
        cache_changed = False

        for cache_entry_hash in list(self.pdf_index.keys()):
            cache_entry = self.pdf_index[cache_entry_hash]
            entry_count += 1
            pdf_cache_filename = self.get_pdf_cache_filename(cache_entry_hash)
            if cache_entry['previous_failure'] is True:
                if os.path.exists(pdf_cache_filename) is False and cache_entry_hash not in self.text_library:
                    failure_count += 1
            if cache_entry_hash not in self.text_library:
                orphan_count += 1
                if clean is True:
                    if os.path.exists(pdf_cache_filename):
                        os.remove(pdf_cache_filename)
                        deleted_count += 1
                    del self.pdf_index[cache_entry_hash]
                    deleted2_count += 1
                    cache_changed = True
                    continue
            if os.path.exists(pdf_cache_filename) is False:
                if clean is True:
                    del self.pdf_index[cache_entry_hash]
                    deleted2_count += 1
                    cache_changed = True
                missing_count += 1
        cache_files = get_files_of_extensions(self.pdf_cache_path, ['pdf'])
        for cache_file in cache_files:
            hash = os.path.splitext(cache_file)[0]
            if hash not in self.pdf_index:
                orphan2_count += 1
                if clean is True:
                    pdf_cache_filename = self.get_pdf_cache_filename(hash)
                    if os.path.exists(pdf_cache_filename):
                        os.remove(pdf_cache_filename)
                        deleted_count += 1
        if cache_changed is True:
            self.save_text_library()
        return (entry_count, failure_count, orphan_count, orphan2_count, deleted_count, deleted2_count, missing_count, cache_changed)
                    
    def get_sources_ext_cnts(self) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        exts: list[str] = []
        sum_ext_cnts: dict[str,int] = {}
        source_ext_cnts: dict[str, dict[str, int]] = {}
        sum_cnt = 0
        for source_name in self.config['document_sources']:
            for ext in self.config['document_sources'][source_name]['file_types']:
                if ext not in exts:
                    exts.append(ext)
        for source_name in self.config['document_sources']:
            # source = self.config['document_sources'][source_name]
            cnt = 0
            source_ext_cnts[source_name] = {}
            for hsh in self.text_library:
                if self.text_library[hsh]['source_name'] == source_name:
                    cnt += 1
                    sum_cnt += 1
                    ext = os.path.splitext(self.text_library[hsh]['descriptor'].lower())[1]
                    if ext and len(ext) > 0:
                        ext = ext[1:]
                    if ext and ext in exts:
                        if ext in source_ext_cnts[source_name]:
                            source_ext_cnts[source_name][ext] += 1
                        else:
                            source_ext_cnts[source_name][ext] = 1
                        if ext in sum_ext_cnts:
                            sum_ext_cnts[ext] += 1
                        else:
                            sum_ext_cnts[ext] = 1
        return sum_ext_cnts, source_ext_cnts
        
    def publish(self, parameters:list[str]|None) -> bool:
        if parameters is None:
            parameters = []
        if self.local_update_required() is True:
            if 'force' not in parameters:
                self.log.warning("Remote version is newer than local version, publish aborted. Use 'force' to override!")
                return False
            else:
                self.log.warning("Override active, publishing older local version over newer remote version!")
                remote, local = self.load_sequence_versions()
                self.write_local_sequence_version(remote)

        remote, local = self.load_sequence_versions()
        self.write_local_sequence_version(local+1)
                
        src = self.storage_path
        if src.endswith('/') is False:
            src += '/'
        dest = self.publish_path
        if dest.endswith('/') is False:
            dest += '/'
        cmd = ['rsync','-avxh', '--exclude', '.*', src, dest, '--delete']
        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.returncode != 0:
            self.log.error(f"Failure: {result.stderr}")
            return False
        remote, local = self.load_sequence_versions()
        self.log.info(f"Published successful remote version: {remote}, local version: {local}")
        return True

    def import_local(self, parameters:list[str]|None) -> bool:
        if parameters is None:
            parameters = []
        remote, local = self.load_sequence_versions()
        if local>remote:
            if 'force' not in parameters:
                self.log.warning("Local data has newer version than remote data, not importing. Use 'force' to override!")
                return False
            else:
                self.log.warning("Override active, importing older version from remote!")
        src = self.publish_path
        if src.endswith('/') is False:
            src += '/'
        dest = self.storage_path
        if dest.endswith('/') is False:
            dest += '/'
        cmd = ['rsync','-avxh', '--exclude', '.*', src, dest, '--delete']
        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.returncode != 0:
            self.log.error(f"Failure: {result.stderr}")
            return False
        remote, local = self.load_sequence_versions()
        self.log.info(f"Import successful remote version: {remote}, local version: {local}")
        return True
