

import os
import logging
import json
import time
import hashlib
import tempfile
import zlib
from typing import TypedDict, cast
import io
import base64
        
from PIL import Image
import pymupdf  # pyright: ignore[reportMissingTypeStubs]
import pymupdf4llm  # pyright: ignore[reportMissingTypeStubs]  # XXX currently locked to 0.19, otherwise export returns empty docs, requires investigation!

class VectorSource(TypedDict):
    name: str
    type: str
    path: str
    file_types: list[str]

    
class VectorConfig(TypedDict):
    vector_sources: list[VectorSource]
    embeddings_model_name: str
    embeddings_device: str
    embeddings_model_trust_code: bool


class LibraryEntry(TypedDict):
    source_name: str
    source_path: str
    icon: str
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


class VectorStore:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("VectorStore")
        self.config_changed:bool = False
        self.config_dir: str = os.path.expanduser("~/.config/vector_store")
        if os.path.isdir(self.config_dir) is False:
            os.makedirs(self.config_dir)            
        self.config_file:str = os.path.join(self.config_dir, "vector_store.json")
        self.config: VectorConfig = self.get_config()
        self.library: dict[str, LibraryEntry] = {}
        self.pdf_index:dict[str, PDFIndex] = {}

        self.storage_path: str = os.path.join(os.path.expanduser("~/.local/share"), "vector_store")
        if os.path.isdir(self.storage_path) is False:
            os.makedirs(self.storage_path)
        self.library_file:str = os.path.join(self.storage_path, "vector_library.json")
        self.pdf_cache_path: str = os.path.join(self.storage_path, "pdf_cache")
        if os.path.isdir(self.pdf_cache_path) is False:
            os.makedirs(self.pdf_cache_path, exist_ok=True)
        self.pdf_index_file: str = os.path.join(self.pdf_cache_path, "pdf_index.json")

        self.embeddings_path:str = os.path.join(self.storage_path, "embeddings")
        if os.path.isdir(self.embeddings_path) is False:
            os.makedirs(self.embeddings_path, exist_ok=True)

        self.icon_width:int = 240
        self.icon_height:int = 320
        
        self.model_list: list[EmbeddingModel]
        self.get_model_list()

        for model in self.model_list:
            model_path = self.model_embedding_path(model['model_name'])
            if os.path.isdir(model_path) is False:
                os.makedirs(model_path, exist_ok=True)
            
        self.load_library()
        self.log.info("VectorStore initialized")

    def model_embedding_path(self, model_name: str) -> str:
        return os.path.join(self.embeddings_path, model_name)
        
    def get_config(self) -> VectorConfig:
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config: VectorConfig = cast(VectorConfig, json.load(f))
        else:
            config = VectorConfig({
                'vector_sources': [
                    VectorSource({
                        'name': 'Calibre',
                        'type': 'calibre_library',
                        'path': '~/ReferenceLibrary/Calibre Library',
                        'file_types': ['txt', 'pdf']
                    }),
                    VectorSource({
                        'name': 'Notes',
                        'type': 'folder',
                        'path': '~/Notes',
                        'file_types': ['md']
                    })
                ],
                'embeddings_model_name': 'granite-embedding-107m-multilingual',
                'embeddings_device': 'auto',
                'embeddings_model_trust_code': True,
            }
            )
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

    @staticmethod
    def _get_sha256(file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _get_crc32(file_path: str) -> int:
        crc32 = 0
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                crc32 = crc32 ^ zlib.crc32(chunk)
        return crc32

    @staticmethod
    def _encode_image_to_base64(image_path: str, width: int | None = None, height: int | None = None) -> str:
        with Image.open(image_path) as img:
            if width is not None or height is not None:
                if width is None:
                    aspect_ratio = img.width / img.height
                    width = int(cast(int, height) * aspect_ratio)
                elif height is None:
                    aspect_ratio = img.height / img.width
                    height = int(width * aspect_ratio)
                img = img.resize((width, height), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            buffer = io.BytesIO()
            img_format = img.format if img.format else 'PNG'
            img.save(buffer, format=img_format)
            _ = buffer.seek(0)
            encoded_string = base64.standard_b64encode(buffer.getvalue()).decode('ascii')
        return encoded_string

    @staticmethod
    def decode_base64_to_image(base64_string: str, output_format: str = "PNG", output_path: str | None = None) -> bytes | None:
        image_data = base64.standard_b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        # Output format JPEG or PNG
        if output_path:
            image.save(output_path, format=output_format)
            return None
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=output_format)
        return img_byte_arr.getvalue()

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
    
    def sync_texts(self):
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
        for source in self.config['vector_sources']:
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
                    sha256_hash = VectorStore._get_sha256(full_path)
                    ext_with_dot = os.path.splitext(filename)[1]
                    ext = ext_with_dot[1:].lower() if ext_with_dot else ""

                    if sha256_hash in self.library and full_path != self.library[sha256_hash]['source_path']:
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

                    icon:str = ""
                    if is_calibre is True:
                        calibre_icon_path = os.path.join(root, 'cover.jpg')
                        if os.path.exists(calibre_icon_path):
                            icon = VectorStore._encode_image_to_base64(calibre_icon_path, self.icon_width, self.icon_height)
                        else:
                            self.log.warning(f"Calibre icon file {calibre_icon_path} not found.")

                    if current_text is None:
                        self.log.error(f"{full_path} has no content or doesnt exist!")
                        existing_hashes.remove(sha256_hash)
                        continue

                    self.library[sha256_hash] = LibraryEntry({'source_name': source['name'], 'source_path': full_path, 'icon': icon, 'text': current_text})
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
        if mode == "" or 'models' in mode:
            print()
            print("Index | Model")
            print("------+--------------------------------------")
            for ind, model in enumerate(self.model_list):
                if model['model_name'] == self.config['embeddings_model_name']:
                    print(f" >{ind+1}<  | {model['model_name']}")
                else:
                    print(f"  {ind+1}   | {model['model_name']}")
            print("  Use 'select <index>' to change the active model")
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
            for source in self.config['vector_sources']:
                source_name = source['name']
                cnt = 0
                ext_cnts: dict[str,int] = {}
                for hsh in self.library:
                    if self.library[hsh]['source_name'] == source_name:
                        cnt += 1
                        ext = os.path.splitext(self.library[hsh]['source_path'].lower())[1]
                        if ext and len(ext) > 0:
                            ext = ext[1:]
                        if ext and ext in exts:
                            if ext in ext_cnts:
                                ext_cnts[ext] += 1
                            else:
                                ext_cnts[ext] = 1
                print(f"{source_name[:8]:8s} | {cnt:5d} |", end="")
                for ext in exts:
                    if ext in ext_cnts:
                        print(f" {ext_cnts[ext]:5d} |", end="")
                    else:
                        print(f" {0:5d} |", end="")
                print(f" {source['path'][-32:]:32s} |")

    def select(self, ind: int):
        if ind<1 or ind>len(self.model_list):
            self.log.error(f"Invalid model index {ind}, use 'list models' to get valid indices")
            return
        new_model = self.model_list[ind-1]['model_name']
        if new_model == self.config['embeddings_model_name']:
            self.log.info("Model {new_model} was already active")
            return
        self.log.info(f"Model {new_model} active")
        self.config['embeddings_model_name'] = new_model
        self.config_changed = True
        self.save_config(self.config)
        
        
