
import os
import logging
import json
import uuid
import hashlib
import zlib
from typing import TypedDict, cast
import io
import base64
        
from PIL import Image
import pymupdf  # pyright: ignore[reportMissingTypeStubs]
import pymupdf4llm  # pyright: ignore[reportMissingTypeStubs]  # XXX currently locked to 0.19, otherwise export returns empty docs, requires investigation!

class VectorSource(TypedDict):
    name: str
    vectype: str
    path: str
    file_types: list[str]

    
class VectorConfig(TypedDict):
    vector_sources: list[VectorSource]
    embeddings_model_name: str
    embeddings_device: str
    embeddings_model_trust_code: bool


class LibraryEntry(TypedDict):
    source_name: str
    filename: str
    sha256_hash: str
    icon: str
    text: str


class PDFIndex(TypedDict):
    previous_failure: bool
    filename: str
    file_size: int
    sha256_hash: str


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

        self.icon_width:int = 240
        self.icon_height:int = 320
        
        self.model_list: list[EmbeddingModel]
        self.get_model_list()
        self.load_library()
        self.log.info("VectorStore initialized")

    def get_config(self) -> VectorConfig:
        config_dir = os.path.expanduser("~/.config/vector_store")
        if os.path.isdir(config_dir) is False:
            os.makedirs(config_dir)            
        config_file = os.path.join(config_dir, "vector_store.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config: VectorConfig = cast(VectorConfig, json.load(f))
        else:
            config = VectorConfig({
                'vector_sources': [
                    VectorSource({
                        'name': 'Calibre',
                        'vectype': 'calibre_library',
                        'path': '~/ReferenceLibrary/Calibre Library',
                        'file_types': ['txt', 'pdf']
                    }),
                    VectorSource({
                        'name': 'Notes',
                        'vectype': 'folder',
                        'path': '~/Notes',
                        'file_types': ['md']
                    })
                ],
                'embeddings_model_name': 'ibm-granite/granite-embedding-107m-multilingual',
                'embeddings_device': 'auto',
                'embeddings_model_trust_code': True,
            }
            )
            with open(config_file, "w") as f:
                json.dump(config, f)
            self.log.warning(f"Default configuration created at {config_file}, please review!")
        return config

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

    def save_library(self):
        with open(self.library_file, "w") as f:
            json.dump(self.library, f)
        with open(self.pdf_index_file, "w") as f:
            json.dump(self.pdf_index, f)

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
                img = img.resize((width, height), Image.LANCZOS)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
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

    def get_pdf_text(self, desc:str, full_path:str, sha256_hash: str) -> tuple[str | None, bool]:
        pdf_text: str | None = None
        index_changed: bool = False
        current_file_size: int = -1

        if os.path.exists(full_path) is False:
            self.log.error(f"Cannot process PDF file {full_path}, file does not exist!")
            return None, False
        current_file_size = os.path.getsize(full_path)
        
        if desc in self.pdf_index:
            cached_info = self.pdf_index[desc]
            if ((sha256_hash != cached_info['sha256_hash'] or current_file_size == cached_info['file_size']) and cached_info['previous_failure']):
                self.log.debug(f"Skipping PDF {desc}: Size matches and previously failed extraction.")
                return None, False
            elif (current_file_size == cached_info['file_size'] and
                not cached_info['previous_failure'] and
                cached_info.get('filename')):
                cache_filename = os.path.join(self.pdf_cache_path, cached_info['filename'])
                if os.path.exists(cache_filename):
                    try:
                        with open(cache_filename, 'r', encoding='utf-8') as f:
                            pdf_text = f.read()
                        return pdf_text, False # Return cached text, index not changed
                    except Exception as e:
                        self.log.warning(f"Failed to read PDF cache file {cache_filename} for {desc}: {e}. Re-extracting.")
                else:
                     self.log.warning(f"PDF cache index points to non-existent file {cache_filename} for {desc}. Re-extracting.")
            elif current_file_size != cached_info['file_size']:
                self.log.info(f"PDF file size changed for {desc} at {full_path}, re-importing text.")
            else: # Size matches, not failed, but filename missing? Inconsistent.
                self.log.warning(f"Inconsistent PDF cache state for {desc}. Re-extracting.")

        if pdf_text is None:
            extracted_text: str | None = None
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
                combined_text = "\n".join(extracted_pages)  # pyright:ignore[reportUnknownArgumentType]
                if combined_text.strip() != "":
                    extracted_text = combined_text
            except Exception as e:
                self.log.info(f"Failed pymupdf extraction for {desc}: {e}", exc_info=True)
                
            if extracted_text is None:
                try:
                    md_text = pymupdf4llm.to_markdown(full_path)  # pyright:ignore[reportUnknownMemberType]
                    if md_text and md_text.strip():
                        extracted_text = md_text # Use the markdown text
                except Exception as e:
                    self.log.error(f"Failed pymupdf4llm fallback extraction for {desc}: {e}", exc_info=True)

            if extracted_text is None:
                if desc in self.pdf_index:
                    self.pdf_index[desc]["previous_failure"] = True
                else:
                    cache_entry = PDFIndex({'filename': "", 'file_size': current_file_size, 'sha256_hash': sha256_hash, 'previous_failure': True})
                    self.pdf_index[desc] = cache_entry
                    index_changed = True
                return None, index_changed
            else:
                pdf_text = extracted_text

            cache_entry: PDFIndex
            if desc in self.pdf_index:
                cache_entry = self.pdf_index[desc]
                self.pdf_index[desc]['file_size'] = len(pdf_text)
                self.pdf_index[desc]['sha256_hash'] = sha256_hash
                self.pdf_index[desc]['previous_failure'] = False                
            else:
                cache_entry = PDFIndex({
                    'filename': str(uuid.uuid4()) + ".txt",
                    'file_size': len(pdf_text),
                    'sha256_hash': sha256_hash,
                    'previous_failure': False
                })
                self.pdf_index[desc] = cache_entry

            cache_filename = os.path.join(self.pdf_cache_path, cache_entry['filename'])
            with open(cache_filename, 'w') as f:
                _ = f.write(pdf_text)
            self.pdf_index[desc] = cache_entry
            index_changed = True

        return pdf_text, index_changed

    def sync_texts(self):
            library_changed = False
            pdf_index_changed = False
            old_library_size = len(list(self.library.keys()))

            existing_descriptors: list[str] = list(self.library.keys())
            abort_scan = False
            for source in self.config['vector_sources']:
                if abort_scan: 
                    break
                source_path = os.path.expanduser(source['path'])
                self.log.info(f"Scanning source '{source['name']}' at '{source_path}'...")
                is_calibre: bool = False
                if source['vectype'] == 'calibre_library':
                    is_calibre = True

                for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: self.log.warning(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                    if abort_scan: 
                        break
                    for filename in files:
                        if abort_scan: 
                            break
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
                        if preferred_ext_exists:
                            continue

                        full_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(full_path, source_path)
                        descriptor = "{" + source['name'] + "}" + relative_path                        
                        sha256_hash = VectorStore._get_sha256(full_path)

                        print("\r{descriptor:80s}", end="")
                        
                        if descriptor in self.library:
                            existing_entry: LibraryEntry | None = self.library[descriptor]
                            if sha256_hash == existing_entry['sha256_hash']:
                                existing_descriptors.remove(descriptor)
                                continue
                        else:
                            existing_entry = None

                        current_text: str | None = None
                        if ext in ['md', 'txt']:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                current_text = f.read()
                        elif ext == 'pdf':
                            # PDF handling might add complexity, ensure it works if used
                            current_text, pdf_index_changed_during_get = self.get_pdf_text(descriptor, full_path, sha256_hash)
                            if pdf_index_changed_during_get:
                                pdf_index_changed = True

                        icon:str = ""
                        if is_calibre is True:
                            calibre_icon_path = os.path.join(root, 'cover.jpg')
                            if os.path.exists(calibre_icon_path):
                                icon = VectorStore._encode_image_to_base64(calibre_icon_path, self.icon_width, self.icon_height)
                            else:
                                self.log.warning(f"Calibre icon file {calibre_icon_path} not found.")

                        if current_text is None:
                            self.log.error(f"{full_path} has no content or doesnt exist!")
                            existing_descriptors.remove(descriptor)
                            continue
                        
                        self.library[descriptor] = LibraryEntry({'source_name': source['name'], 'filename': filename, 'sha256_hash': sha256_hash, 'icon': icon, 'text': current_text})
                        library_changed = True
            print()
            new_library_size = len(list(self.library.keys()))
            if library_changed is True or pdf_index_changed is True:
                self.save_library()
                self.log.info(f"Library size {old_library_size} -> {new_library_size}")
            else:
                self.log.info("No changes")
