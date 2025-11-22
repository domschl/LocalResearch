import os
import logging
import json
import time
import datetime
import tempfile
from typing import TypedDict, cast, Any, Callable
import colorsys
import math
import numpy as np

import torch
import transformers
from sentence_transformers import SentenceTransformer

from research_defs import TextLibraryEntry, ProgressState, get_files_of_extensions, SearchResultEntry

support_dim3d = False
try:
    import umap  # pyright: ignore[reportMissingTypeStubs]
    support_dim3d = True
except ImportError:
    umap = None
    pass

# INTEL XPU incantation:
# uv pip install -U --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu


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





class VectorStore:
    def __init__(self, storage_path:str, config_path:str):
        self.current_version: int = 6
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
            name = text_library[hash].descriptor
            if os.path.exists(filename):
                continue
            new_chunks += VectorStore.get_chunk_count(text_library[hash].text, self.config['chunk_size'], self.config['chunk_overlap'])

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
            name = text_library[hash].descriptor
            if os.path.exists(filename):
                continue
            # self.save_embeddings_tensor(text_library[hash].text, filename)
            batch_size = self.config['batch_base_multiplier'] * self.model['batch_multiplier']
            chunks: list[str] = VectorStore.get_chunks(text_library[hash].text, self.config['chunk_size'], self.config['chunk_overlap'])
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
            progress_state = ProgressState(issues=len(errors), state="Index complete", percent_completion=1.0, vars={}, finished=True)
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
                search_results.append(SearchResultEntry(
                    cosine=cosine, 
                    hash=hash, 
                    chunk_index=max_ind, 
                    entry=text_library[hash], 
                    text=None, 
                    significance=None
                ))
                search_results = sorted(search_results, key=lambda res: res.cosine)
                search_results = search_results[-max_results:]
                best_min_cosine = search_results[0].cosine
        search_time = time.time() - start_time
        key = f"{self.config['embeddings_model_name']}-{str(self.device)}"
        self.perf[key] = search_time
        highlight_start = time.time()
        for index, result in enumerate(search_results):
            result_text = self.get_chunk_context_aware(result.entry.text, result.chunk_index, self.config['chunk_size'], self.config['chunk_overlap'])
            replacers = [("\n", " "), ("\r", " "), ("\b", " "), ("\t", " "), ("  ", " ")]
            zero_width = [("\u200b", ""), ("\u200c", ""), ("\u200d", ""), ("\u00ad", ""), ("\ufeff", "")] 
            replacers += zero_width
            old_text = ""
            while old_text != result_text:
                old_text = result_text
                for rep in replacers:
                    result_text = result_text.replace(rep[0], rep[1])

            search_results[index].text = result_text

            if highlight is True:
                significance: list[float] = [0.0] * len(result_text)
                stepped_significance: list[float] = self.get_significance(result_text, search_tensor, context_length, context_steps, highlight_cutoff)
                if highlight_dampening == 0.0:
                    self.log.error("Dampending must not be zero!")
                    highlight_dampening = 1.0
                for ind in range(len(result_text)):
                    significance[ind] = stepped_significance[ind // context_steps] * result.cosine / highlight_dampening
                search_results[index].significance = significance
        
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
        texts = [text_library[hash].descriptor+f"[{chunk_id}]" for hash, chunk_id in hashes]
        doc_ids = [text_library[hash].descriptor for hash, _chunk_id in hashes]

        unique_doc_ids = list(set(doc_ids))

        num_unique_docs = len(unique_doc_ids)
        color_map: dict[str, list[int]] = {}
        for i, doc_id_val in enumerate(unique_doc_ids):
            hue = i / num_unique_docs if num_unique_docs > 0 else 0
            rgb_float = colorsys.hls_to_rgb(hue, 0.5, 0.8) 
            color_map[doc_id_val] = [int(c * 255) for c in rgb_float]

        colors: list[list[int]] = [color_map.get(text_library[hash].descriptor, [128,128,128]) for hash, _chunk_id in hashes]
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
