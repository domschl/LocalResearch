
import os
import logging
import json
import uuid
import hashlib
import zlib
from typing import TypedDict, cast, Literal
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
    emb_ptrs: dict[str, tuple[int, int]]


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
                basename = os.path.basename(cached_info['filename'])
                local_path = os.path.join(self.pdf_cache_path, basename)
                if os.path.exists(local_path):
                    try:
                        with open(local_path, 'r', encoding='utf-8') as f:
                            pdf_text = f.read()
                        return pdf_text, False # Return cached text, index not changed
                    except Exception as e:
                        self.log.warning(f"Failed to read PDF cache file {local_path} for {desc}: {e}. Re-extracting.")
                else:
                     self.log.warning(f"PDF cache index points to non-existent file {local_path} for {desc}. Re-extracting.")
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

            if desc in self.pdf_index:
                cache_entry = self.pdf_index[desc]
                self.pdf_index[desc]['file_size'] = len(pdf_text)
                self.pdf_index[desc]['sha256_hash'] = sha256_hash
                self.pdf_index[desc]['previous_failure'] = False                
            else:
                cache_entry: PDFIndex = {
                    'filename': str(uuid.uuid4()) + ".txt",
                    'file_size': len(pdf_text),
                    'sha256_hash': sha256_hash,
                    'previous_failure': False
                }
                self.pdf_index[desc] = cache_entry

            with open(cache_entry['filename'], 'w') as f:
                _ = f.write(pdf_text)
            self.pdf_index[desc] = cache_entry
            index_changed = True

        return pdf_text, index_changed

    def sync_texts(self, max_imports: int | None = None):
            library_changed = False
            pdf_index_changed = False

            existing_descriptors: list[str] = list(self.library.keys())
            abort_scan = False
            for source in self.config['vector_sources']:
                if abort_scan: 
                    break
                source_path = source['path']
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

                        if descriptor in self.library:
                            existing_entry: LibraryEntry | None = self.library[descriptor]
                            if sha256_hash == existing_entry['sha256_hash']:
                                existing_descriptors.remove(descriptor)
                                continue
                        else:
                            existing_entry = None
                                                            
                        if ext in ['md', 'txt']:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                current_text = f.read()
                        elif ext == 'pdf':
                            # PDF handling might add complexity, ensure it works if used
                            current_text, pdf_index_changed_during_get = self.get_pdf_text(descriptor, full_path, sha256_hash)
                            if pdf_index_changed_during_get: pdf_index_changed = True

                        icon:str = ""
                        if is_calibre is True:
                            calibre_icon_path = os.path.join(root, 'cover.jpg')
                            if os.path.exists(calibre_icon_path):
                                icon = VectorStore._encode_image_to_base64(calibre_icon_path, self.icon_width, self.icon_height)
                            else:
                                self.log.warning(f"Calibre icon file {calibre_icon_path} not found.")

                        needs_update = False
                        
                        if existing_entry:
                            # Update handling
                            old_pointers_to_collect: dict[str, tuple[int, int]] = {}

                            if icon != existing_entry.get('icon'):
                                self.log.info(f"Updating icon for {desc_path}")
                                # needs_update = True
                                existing_entry['icon'] = icon
                                lib_changed = True
                                # old_pointers_to_collect = existing_entry.get('emb_ptrs', {}).copy()
                            if current_text is not None and existing_entry.get('text') != current_text:
                                self.log.info(f"Updating text for {desc_path}")
                                needs_update = True
                                if old_pointers_to_collect == {}:
                                    old_pointers_to_collect = existing_entry.get('emb_ptrs', {}).copy()
                                existing_entry['text'] = current_text
                            elif current_text is None: #  and existing_entry.get('text') is not None:
                                self.log.warning(f"Text unreadable for {desc_path}. Clearing.")
                                needs_update = True
                                if old_pointers_to_collect == {}:
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
                            entry: LibEntry = LibEntry({'source_name': source['name'], 'desc_filename': desc_path, 'filename': full_path, 'text': current_text, 'emb_ptrs': {}, 'icon': icon})
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

                        # temp_device = self.resolve_device(self.config['embeddings_device'])
                        original_tensor: torch.Tensor = cast(torch.Tensor, torch.load(temp_tensor_path, map_location="cpu")) 
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
                            raise VectorCriticalError(f"Tensor shape mismatch after indexing for {model_name}. Aborting cleanup.")
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
                                    raise VectorCriticalError(f"Pointer adjustment failed critically for {model_name}.")
                                current_model_pointers[old_start] = (new_start, old_length)
                        new_pointer_maps[model_name] = current_model_pointers

                    except VectorCriticalError: 
                        all_models_processed_ok = False
                        raise
                    except Exception as e: 
                        all_models_processed_ok = False
                        raise VectorCriticalError(f"Failure during tensor processing for {model_name}.") from e

                # --- Commit Phase ---
                if all_models_processed_ok:
                    # Save modified tensors
                    for model_name, tensor_data in modified_tensors.items():
                        try:
                            self._atomic_save_tensor(tensor_data, self._get_tensor_path(model_name))
                            self.log.info(f"Atomically saved cleaned tensor for '{model_name}'. Shape: {tensor_data.shape}")
                        except VectorCriticalError as e:
                            raise VectorCriticalError(f"Failed to commit cleaned tensor for {model_name}.") from e

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
                                raise VectorCriticalError(consistency_msg)
                            else:
                                self.log.info(f"In-memory tensor for '{current_model_name}' updated and passed consistency check ({actual_rows} rows).")
                        except Exception as check_e:
                             # Handle potential errors during the check itself
                             self.log.critical(f"CRITICAL Error during post-assignment consistency check for {current_model_name}: {check_e}")
                             self.embeddings_matrix = None # Safety
                             # Re-raise as critical? Or just log and clear matrix? Re-raise seems safer.
                             raise VectorCriticalError(f"Post-assignment check failed for {current_model_name}") from check_e

                    # Reload current tensor if modified
                    # if self.current_model and self.current_model['model_name'] in modified_tensors:
                    #      self.log.info(f"Reloading current model's ({self.current_model['model_name']}) tensor after cleanup.")
                    #      try:
                    #          # Check consistency *after* reload
                    #          _ = self._load_tensor_internal(self.current_model['model_name'], check_consistency=True)
                    #      except (VectorConsistencyError, VectorCriticalError) as e:
                    #          # This is where the previous crash happened - need to ensure it passes now
                    #          self.log.critical(f"CRITICAL: Consistency error *after* cleanup reload for {self.current_model['model_name']}: {e}")
                    #          self.embeddings_matrix = None # Safety
                    #          raise VectorCriticalError("Consistency check failed immediately after cleanup. Logic error suspected.") from e

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

