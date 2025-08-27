import os
import logging
import json
import time
import hashlib
import tempfile
from typing import TypedDict, cast
import io
import base64
import subprocess
        
from PIL import Image
import pymupdf  # pyright: ignore[reportMissingTypeStubs]
import pymupdf4llm  # pyright: ignore[reportMissingTypeStubs]  # XXX currently locked to 0.19, otherwise export returns empty docs, requires investigation!
import torch
from sentence_transformers import SentenceTransformer


class DocumentSource(TypedDict):
    name: str
    type: str
    path: str
    file_types: list[str]


class DocumentConfig(TypedDict):
    document_sources: list[DocumentSource]
    publish_path: str


class VectorConfig(TypedDict):
    embeddings_model_name: str
    embeddings_device: str
    embeddings_model_trust_code: bool
    batch_base_multiplier: int


class LibraryEntry(TypedDict):
    source_name: str
    source_path: str
    text: str


class PDFIndex(TypedDict):
    previous_failure: bool
    file_size: int


class EmbeddingModel(TypedDict):
    model_hf_name: str
    model_name: str
    emb_dim: int
    max_input_token: int
    chunk_size: int
    chunk_overlap: int
    batch_multiplier: int

    
class SequenceVersion(TypedDict):
    sequence: int


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
        self.log: logging.Logger = logging.getLogger("VectorStore")
        self.storage_path:str = storage_path
        self.config_path:str = config_path
        if os.path.isdir(self.config_path) is False:
            os.makedirs(self.config_path)
        self.config_changed: bool = False
        self.config_file:str = os.path.join(self.config_path, "vector_store.json")
        self.config: VectorConfig = self.get_config()
        self.embeddings_path:str = os.path.join(self.storage_path, "embeddings")
        if os.path.isdir(self.embeddings_path) is False:
            os.makedirs(self.embeddings_path, exist_ok=True)
        self.model_list: list[EmbeddingModel]
        self.get_model_list()
        for model in self.model_list:
            model_path = self.model_embedding_path(model['model_name'])
            if os.path.isdir(model_path) is False:
                os.makedirs(model_path, exist_ok=True)
        self.model: EmbeddingModel | None = None
        self.engine: SentenceTransformer | None = None
        self.device: torch.device = torch.device(self.resolve_device())
            
    def model_embedding_path(self, model_name: str) -> str:
        return os.path.join(self.embeddings_path, model_name)

    def get_model_list(self):
        self.model_list = [
            { # granite-107m
                'model_hf_name': 'ibm-granite/granite-embedding-107m-multilingual',
                'model_name': 'granite-embedding-107m-multilingual',
                'emb_dim': 384, 
                'max_input_token': 512,
                'chunk_size': 2048, 
                'chunk_overlap': 2048 // 3,
                'batch_multiplier': 64,
            },
            {
                'model_hf_name': 'ibm-granite/granite-embedding-278m-multilingual',
                'model_name': 'granite-embedding-278m-multilingual',
                'emb_dim': 768,
                'max_input_token': 512,
                'chunk_size': 2048,
                'chunk_overlap': 2048 // 3,
                'batch_multiplier': 32,
            },
            {
                'model_hf_name': 'nomic-ai/nomic-embed-text-v2-moe',
                'model_name': 'nomic-embed-text-v2-moe',
                'emb_dim': 768,  #  Matryoshka Embeddings
                'max_input_token': 512,
                'chunk_size': 2048,
                'chunk_overlap': 2048 // 3,
                'batch_multiplier': 32,
            },
            {
                'model_hf_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'model_name': 'all-MiniLM-L6-v2',
                'emb_dim': 384,
                'max_input_token': 512, # Use underlying model's limit or common practice
                'chunk_size': 1024,     # Adjust based on token limit and desired context
                'chunk_overlap': 1024 // 3,
                'batch_multiplier': 128,
            },
            {
                'model_hf_name': 'Qwen/Qwen3-Embedding-0.6B',
                'model_name': 'Qwen3-Embedding-0.6B',
                'emb_dim': 1024,
                'max_input_token': 512,
                'chunk_size': 1024,     # Adjust based on token limit and desired context
                'chunk_overlap': 1024 // 3,
                'batch_multiplier': 1,
            },
        ]

    def get_config(self) -> VectorConfig:
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config: VectorConfig = cast(VectorConfig, json.load(f))
        else:
            config = VectorConfig({
                'embeddings_model_name': 'granite-embedding-107m-multilingual',
                'embeddings_device': 'auto',
                'embeddings_model_trust_code': True,
                'batch_base_multiplier': 2,
            })
            self.save_config(config)
            self.log.warning(f"Default configuration created at {self.config_file}, please review!")
        self.config_changed = False
        return config

    def save_config(self, config: VectorConfig):
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.config_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(config, temp_file)
            os.replace(temp_path, self.config_file)  # atomic update
        except Exception as e:
            self.log.error("Config-file update interrupted, not updated.")
            os.remove(temp_path)
            raise e

    def list_models(self, current_model: str|None = None):
        print()
        print("Index | Embbedings | Model")
        print("------+------------+--------------------------------------")
        for ind, model in enumerate(self.model_list):
            path = self.model_embedding_path(model['model_name'])
            cnt = len(get_files_of_extensions(path, ['pt']))
            if model['model_name'] == current_model:
                print(f" >{ind+1}<  | {cnt:10d} | {model['model_name']}")
            else:
                print(f"  {ind+1}   | {cnt:10d} | {model['model_name']}")
        print("  Use 'select <index>' to change the active model")

    def list(self, mode: str):
        if mode == "" or 'models' in mode:
            self.list_models(self.config['embeddings_model_name'])

    def select(self, ind: int) -> str | None:
        if ind<1 or ind>len(self.model_list):
            self.log.error(f"Invalid model index {ind}, use 'list models' to get valid indices")
            return None
        new_model = self.model_list[ind-1]['model_name']
        if new_model == self.config['embeddings_model_name']:
            self.log.info("Model {new_model} was already active")
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

    def get_embedding_filename(self, hash:str) -> str:
        current_embeddings_path = os.path.join(self.embeddings_path, self.config['embeddings_model_name'])
        return os.path.join(current_embeddings_path, hash+".pt")

    def resolve_device(self) -> str:
        dev = self.config.get('embeddings_device', 'auto')
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
            self.log.warning(f"Undefined device {dev}, using 'cpu' fallback")
            return 'cpu'

    def load_model(self):
        if self.model is None:
            for model in self.model_list:
                if model['model_name'] == self.config['embeddings_model_name']:
                    self.model = model
        if self.model is None:
            self.log.error(f"Invalid model {self.config['embeddings_model_name']} could not be identified, load_model failed!")
            return
        if self.engine is None:
            self.engine = SentenceTransformer(self.model['model_hf_name'], trust_remote_code=self.config['embeddings_model_trust_code']).to(self.device)          

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
        
    def save_embeddings_tensor(self, text:str, filename:str):
        if self.model is None or self.engine is None:
            return
        batch_size = self.config['batch_base_multiplier'] * self.model['batch_multiplier']
        chunks: list[str] = VectorStore.get_chunks(text, self.model['chunk_size'], self.model['chunk_overlap'])
        embeddings_tensor = self.engine.encode(chunks, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)  # pyright:ignore[reportUnknownMemberType]
        self.save_tensor(embeddings_tensor, filename)
        del embeddings_tensor
    
    def index(self, library:dict[str,LibraryEntry]):
        self.load_model()
        if self.model is None or self.engine is None:
            self.log.error("Failed to load model, cannot index!")
            return
        
        for hash in library:
            filename = self.get_embedding_filename(hash)
            name = library[hash]['source_path']
            if os.path.exists(filename):
                continue
            print(f"\rIndexing: {name[-80:]:80s}", end="")
            self.save_embeddings_tensor(library[hash]['text'], filename)
        print(" "*80)
        self.log.info("Index completed")

    def search(self, search_text:str, library:dict[str,LibraryEntry]):
        self.load_model()
        if self.model is None or self.engine is None:
            self.log.error("Failed to load model, cannot index!")
            return
        device = torch.device(self.resolve_device())
        search_tensor = self.engine.encode([search_text], convert_to_tensor=True, show_progress_bar=False).to(device)  # pyright:ignore[reportUnknownMemberType]
        path = self.model_embedding_path(self.model['model_name'])
        tensor_file_list = get_files_of_extensions(path, ['pt'])
        best_cosine: float | None = None
        best_chunk: int | None = None
        best_doc: LibraryEntry | None = None
        for tensor_file in tensor_file_list:
            tensor_path = os.path.join(path, tensor_file)
            tensor:torch.Tensor = cast(torch.Tensor, torch.load(tensor_path, map_location=device))
            cosines = torch.matmul(search_tensor, tensor.T).T
            max_ind:int = int(torch.argmax(cosines).item())
            cosine:float = cosines[max_ind].item()
            if best_cosine is None or cosine > best_cosine:
                best_cosine = cosine
                hash = os.path.splitext(tensor_file)[0]
                best_chunk = max_ind
                best_doc = library[hash]
                print(f"\r{best_cosine:.4f} {best_doc['source_path'][-80:]:80s}, best_chunk: {best_chunk}")
        print()
        if best_doc is not None and best_chunk is not None:
            result_text = self.get_chunk(best_doc['text'], best_chunk, self.model['chunk_size'], self.model['chunk_overlap'])
            replacers = [("\n", " "), ("\t", " "), ("  ", " ")]
            old_text = ""
            while old_text != result_text:
                old_text = result_text
                for rep in replacers:
                    result_text = result_text.replace(rep[0], rep[1])            
            print(result_text)
            
    
class DocumentStore:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("DocumentStore")
        self.config_changed:bool = False
        self.config_path: str = os.path.expanduser("~/.config/local_research")
        if os.path.isdir(self.config_path) is False:
            os.makedirs(self.config_path)            
        self.config_file:str = os.path.join(self.config_path, "document_store.json")
        self.config: DocumentConfig = self.get_config()
        self.library: dict[str, LibraryEntry] = {}
        self.pdf_index:dict[str, PDFIndex] = {}

        self.publish_path: str = os.path.expanduser(self.config['publish_path'])
        if os.path.isdir(self.publish_path) is False:
            os.makedirs(self.publish_path)

        self.storage_path: str = os.path.join(os.path.expanduser("~/.local/share"), "local_research")
        if os.path.isdir(self.storage_path) is False:
            os.makedirs(self.storage_path)
        self.library_file:str = os.path.join(self.storage_path, "document_library.json")
        self.sequence_file:str = os.path.join(self.storage_path, "version_seq.json")
        self.remote_sequence_file:str = os.path.join(self.publish_path, "version_seq.json")
        self.pdf_cache_path: str = os.path.join(self.storage_path, "pdf_cache")
        if os.path.isdir(self.pdf_cache_path) is False:
            os.makedirs(self.pdf_cache_path, exist_ok=True)
        self.pdf_index_file: str = os.path.join(self.pdf_cache_path, "pdf_index.json")

        self.load_library()
        remote, local = self.load_sequence_versions()
        self.log.info(f"DocumentStore initialized: remote data version: {remote}, local version: {local}")
        if self.local_update_required() is True:
            self.log.info("Please use 'import' to acquire the latest data version")

    def get_config(self) -> DocumentConfig:
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config: DocumentConfig = cast(DocumentConfig, json.load(f))
        else:
            config = DocumentConfig({
                'document_sources': [
                    DocumentSource({
                        'name': 'Calibre',
                        'type': 'calibre_library',
                        'path': '~/ReferenceLibrary/Calibre Library',
                        'file_types': ['txt', 'pdf']
                    }),
                    DocumentSource({
                        'name': 'Notes',
                        'type': 'folder',
                        'path': '~/Notes',
                        'file_types': ['md']
                    })
                ],
                'publish_path': '~/LocalResearch'
                })
            self.save_config(config)
            self.log.warning(f"Default configuration created at {self.config_file}, please review!")
        self.config_changed = False
        return config

    @staticmethod
    def _get_sha256(filename:str):
        with open(filename, 'rb', buffering=0) as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()

    def save_config(self, config: DocumentConfig):
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.config_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(config, temp_file)
            os.replace(temp_path, self.config_file)  # atomic update
        except Exception as e:
            self.log.error("Config-file update interrupted, not updated.")
            os.remove(temp_path)
            raise e

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
        
    def load_library(self):
        print("Loading library data...", end="", flush=True)
        if os.path.exists(self.library_file):
            with open(self.library_file, "r") as f:
                self.library = json.load(f)
        else:
            self.library = {}
        if os.path.exists(self.pdf_index_file):
            with open(self.pdf_index_file, "r") as f:
                self.pdf_index = json.load(f)
        else:
            self.pdf_index = {}
            
        # home = os.path.expanduser("~")
        # if home.startswith('/Users'):
        #     foreign_home = home.replace('/Users', '/home')
        # else:
        #     foreign_home = home.replace('/home', '/Users')
        # upgraded = False
        # for entry in self.library:
        #     if 'icon' in self.library[entry]:
        #         del self.library[entry]['icon']
        #         upgraded = True
        #     if self.library[entry]['source_path'].startswith(home):
        #         self.library[entry]['source_path'] = '~' + self.library[entry]['source_path'][len(home):]
        #         upgraded = True
        #     elif self.library[entry]['source_path'].startswith(foreign_home):
        #         self.library[entry]['source_path'] = '~' + self.library[entry]['source_path'][len(foreign_home):]
        #         upgraded = True
        # if upgraded is True:
        #     print("upgraded... ", end="")
        #     self.save_library()
        # else:
        #     print("not upgraded. ", end="")
            
        print(" Done.")

    def save_library(self):
        print("Saving library data...", end="", flush=True)
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.library_file))
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:            
                json.dump(self.library, temp_file)
            os.replace(temp_path, self.library_file)               
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
        print(" Done.")

    def get_pdf_cache_filename(self, sha256_hash:str) -> str:
        basename = sha256_hash+".txt"
        return os.path.join(self.pdf_cache_path, basename)
    
    def get_pdf_text(self, full_path:str, sha256_hash: str) -> tuple[str | None, bool]:
        pdf_text: str | None = None
        index_changed: bool = False
        current_file_size: int = -1

        if os.path.exists(full_path) is False:
            self.log.error(f"Cannot process PDF file {full_path}, file does not exist!")
            return None, False
        current_file_size = os.path.getsize(full_path)
        
        if sha256_hash in self.pdf_index:
            cached_info = self.pdf_index[sha256_hash]
            if cached_info['previous_failure']:
                print()
                self.log.debug(f"Skipping PDF {full_path}: previously failed extraction.")
                return None, False
            else:                
                pdf_cache_filename = self.get_pdf_cache_filename(sha256_hash)
                if os.path.exists(pdf_cache_filename):
                    try:
                        with open(pdf_cache_filename, 'r', encoding='utf-8') as f:
                            pdf_text = f.read()
                        return pdf_text, False # Return cached text, index not changed
                    except Exception as e:
                        print()
                        self.log.warning(f"Failed to read PDF cache file {pdf_cache_filename} for {full_path}: {e}. Re-extracting.")
                else:
                    print()
                    self.log.warning(f"PDF cache index points to non-existent file {pdf_cache_filename} for {full_path}. Re-extracting.")

        if pdf_text is None:
            extracted_text: str | None = None
            try:
                doc = pymupdf.open(full_path)
                extracted_pages = []
                for page_num, page in enumerate(doc):  # pyright:ignore[reportArgumentType, reportUnknownVariableType]
                    try:
                        page_text = page.get_text()  # pyright:ignore[reportUnknownMemberType, reportUnknownVariableType]
                        if isinstance(page_text, str) and len(page_text)>7: 
                            extracted_pages.append(page_text)  # pyright: ignore[reportUnknownMemberType]
                    except Exception as page_e: 
                        self.log.warning(f"Failed extraction on page {page_num+1} of {full_path}: {page_e}")
                doc.close()
                combined_text = "\n".join(extracted_pages)  # pyright:ignore[reportUnknownArgumentType]
                if combined_text.strip() != "" and len(combined_text.strip()) > 7:
                    extracted_text = combined_text
            except Exception as e:
                self.log.info(f"Failed pymupdf extraction for {full_path}: {e}", exc_info=True)
                
            if extracted_text is None:
                try:
                    md_text = pymupdf4llm.to_markdown(full_path)  # pyright:ignore[reportUnknownMemberType]
                    if md_text and md_text.strip():
                        extracted_text = md_text # Use the markdown text
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
        base, ext_with_dot = os.path.splitext(filename)
        ext = ext_with_dot[1:].lower() if ext_with_dot else ""
        if '.caltrash' in root_path:
            return True
        if ext not in valid_types: 
            return True
        preferred_ext_exists = False
        preferred_order = ['txt', 'md']
        if ext == 'pdf':
             for pref_ext in preferred_order:
                 if os.path.exists(os.path.join(root_path, base + '.' + pref_ext)):
                     preferred_ext_exists = True
                     break
        if preferred_ext_exists:
            return True
        return False
    
    def sync_texts(self, parameters:str):
        if self.local_update_required() is True:
            if 'force' not in parameters.split(' '):
                self.log.warning("Please first update local data using 'import', since remote has newer data! (use 'force' to override)")
                return
            else:
                self.log.warning("Override active, syncing even so remote has newer data!")
        library_changed = False
        pdf_index_changed = False
        old_library_size = len(list(self.library.keys()))
        last_saved = time.time()

        existing_hashes: list[str] = list(self.library.keys())
        abort_scan = False
        pdf_cache_hits = 0
        duplicate_count = 0
        nl_required = False
        last_status = time.time()
        for source in self.config['document_sources']:
            if abort_scan: 
                break
            source_path = os.path.expanduser(source['path'])
            self.log.info(f"Scanning source '{source['name']}' at '{source_path}'...")
            is_calibre: bool = False
            if source['type'] == 'calibre_library':
                is_calibre = True
            source_file_count = 0
            for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: self.log.warning(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                for filename in files:
                    if self.skip_file_in_sync(root, filename, source['file_types']) is True:
                        continue
                    source_file_count += 1
            file_count = 0
            for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: self.log.warning(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                if abort_scan: 
                    break
                for filename in files:
                    if abort_scan: 
                        break
                    if self.skip_file_in_sync(root, filename, source['file_types']) is True:
                        continue

                    file_count += 1
                    full_path = os.path.join(root, filename)
                    if full_path.startswith(os.path.expanduser('~')):
                        rel_path = '~'+full_path[len(os.path.expanduser('~')):]
                    else:
                        rel_path = full_path
                    sha256_hash = DocumentStore._get_sha256(full_path)
                    ext_with_dot = os.path.splitext(filename)[1]
                    ext = ext_with_dot[1:].lower() if ext_with_dot else ""

                    if sha256_hash in self.library and rel_path != self.library[sha256_hash]['source_path']:
                        if nl_required:
                            print()
                            nl_required = False
                        self.log.warning(f"File {full_path} is a duplicate of {self.library[sha256_hash]['source_path']}, ignoring this copy.")
                        duplicate_count += 1
                        continue

                    if sha256_hash in existing_hashes:
                        existing_hashes.remove(sha256_hash)
                        if time.time() - last_status > 1 or file_count==source_file_count:
                            if file_count == source_file_count:
                                print(f"\rChecked  {file_count}/{source_file_count}   ", end="", flush=True)
                            else:
                                print(f"\rChecking {file_count}/{source_file_count}...", end="", flush=True)
                            last_status = time.time()
                            nl_required = True
                        continue

                    current_text: str | None = None
                    
                    if ext in ['md', 'txt']:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            current_text = f.read()
                        print(f"\r {file_count}/{source_file_count} | {full_path[-80:]:80s} Text       ", end="")
                        nl_required = True
                    elif ext == 'pdf':
                        print(f"\r {file_count}/{source_file_count} | {full_path[-80:]:80s} Scanning...", end="")
                        current_text, pdf_index_changed_during_get = self.get_pdf_text(full_path, sha256_hash)
                        if pdf_index_changed_during_get is True:
                            pdf_index_changed = True
                        else:
                            pdf_cache_hits += 1
                        print(f"\r {file_count}/{source_file_count} | {full_path[-80:]:80s} PDF        ", end="")
                        nl_required = True
                    else:
                        self.log.error(f"Handling for file-type {ext} not implemented!")

                    if current_text is None:
                        self.log.error(f"{full_path} has no content or doesnt exist!")
                        existing_hashes.remove(sha256_hash)
                        continue
                                            
                    self.library[sha256_hash] = LibraryEntry({'source_name': source['name'], 'source_path': rel_path, 'text': current_text})
                    library_changed = True

                    if time.time() - last_saved > 180:
                        self.save_library()
                        library_changed = False
                        last_saved =  time.time()
            if nl_required is True:
                print()
                nl_required = False
        new_library_size = len(list(self.library.keys()))
        if duplicate_count > 0:
            self.log.warning(f"{duplicate_count} duplicates were ignored during import, please re-run sync.")
            
        if len(existing_hashes) > 0:
            self.log.warning(f"{len(existing_hashes)} debris entries")
            for debris in existing_hashes:
                if debris in self.library:
                    self.log.info(f"Deleting library entry {self.library[debris]['source_path']}")
                    del self.library[debris]
                    library_changed = True
                if debris in self.pdf_index:
                    del self.pdf_index[debris]
                    pdf_index_changed = True
                
        if library_changed is True or pdf_index_changed is True:
            self.save_library()
            self.log.info(f"Library size {old_library_size} -> {new_library_size}")
        else:
            self.log.info(f"No changes")

    def check_pdf_cache(self):
        failure_count = 0
        entry_count = 0
        orphan_count = 0
        missing_count = 0
        for cache_entry_hash in self.pdf_index:
            cache_entry = self.pdf_index[cache_entry_hash]
            entry_count += 1
            if cache_entry['previous_failure'] is True:
                failure_count += 1
            if cache_entry_hash not in self.library:
                orphan_count += 1
            pdf_cache_filename = self.get_pdf_cache_filename(cache_entry_hash)
            if os.path.exists(pdf_cache_filename) is False:
                missing_count += 1
        print(f"PDF cache entries:   {entry_count}")
        print(f"PDF failures:        {failure_count}")
        print(f"PDF cache orphans:   {orphan_count}")
        print(f"Missing cache files: {missing_count}")
        
    def check(self, mode: str | None = None):
        if mode is None or mode == "" or 'pdf' in mode.lower():
            self.log.info("Checking PDF cache consistency")
            self.check_pdf_cache()
        else:
            self.log.error(f"Use 'check <mode>', valid modes are: 'pdf'")
        
    def list(self, mode: str):
        if mode == "" or 'sources' in mode:
            exts = ['pdf', 'txt', 'md']
            print()
            print("Source   | Docs  |", end="")
            for ext in exts:
                print(f" {ext:5s} |", end="")
            print(" Path                             |")
            print("---------+-------+", end="")
            for ext in exts:
                print("-------+", end="")
            print("----------------------------------+")
            sum_ext_cnts: dict[str,int] = {}
            sum_cnt = 0
            for source in self.config['document_sources']:
                source_name = source['name']
                cnt = 0
                ext_cnts: dict[str,int] = {}
                for hsh in self.library:
                    if self.library[hsh]['source_name'] == source_name:
                        cnt += 1
                        sum_cnt += 1
                        ext = os.path.splitext(self.library[hsh]['source_path'].lower())[1]
                        if ext and len(ext) > 0:
                            ext = ext[1:]
                        if ext and ext in exts:
                            if ext in ext_cnts:
                                ext_cnts[ext] += 1
                            else:
                                ext_cnts[ext] = 1
                            if ext in sum_ext_cnts:
                                sum_ext_cnts[ext] += 1
                            else:
                                sum_ext_cnts[ext] = 1
                print(f"{source_name[:8]:8s} | {cnt:5d} |", end="")
                for ext in exts:
                    if ext in ext_cnts:
                        print(f" {ext_cnts[ext]:5d} |", end="")
                    else:
                        print(f" {0:5d} |", end="")
                print(f" {source['path'][-32:]:32s} |")
            print("---------+-------+", end="")
            for ext in exts:
                print("-------+", end="")
            print("----------------------------------+")
            print(f"Total    | {sum_cnt:5d} |", end="")
            for ext in exts:
                if ext in sum_ext_cnts:
                    print(f" {sum_ext_cnts[ext]:5d} |", end="")
                else:
                    print(f" {0:5d} |", end="")
            print(f"                                  |")

    def publish(self, parameters:str) -> bool:
        if self.local_update_required() is True:
            if 'force' not in parameters.split(' '):
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

    def import_local(self, parameters:str) -> bool:
        remote, local = self.load_sequence_versions()
        if local>remote:
            if 'force' not in parameters.split(' '):
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
    
