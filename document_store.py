import os
import tempfile
import time
import logging
import json
import hashlib
import subprocess

# import pymupdf
pymupdf = None
import pymupdf.layout  # pyright: ignore[reportUnusedImport, reportMissingTypeStubs]
import pymupdf4llm  # pyright: ignore[reportMissingTypeStubs]  # XXX currently locked to 0.19, otherwise export returns empty docs, requires investigation!

from typing import TypedDict, cast, Callable

from research_defs import MetadataEntry, TextLibraryEntry, ProgressState, SearchResultEntry, get_files_of_extensions
from research_tools import DocumentTable
from markdown_handler import MarkdownTools
from orgmode_handler import OrgmodeTools
from calibre_handler import CalibreTools
from time_lines import TimeLines
from sync_tools import SyncTools, SyncTarget
from search_tools import SearchTools
from perf_stats import PerfStats


class DocumentSource(TypedDict):
    type: str
    path: str
    file_types: list[str]


class DocumentConfig(TypedDict):
    version: int
    document_sources: dict[str, DocumentSource]
    sync_targets: dict[str, SyncTarget]
    vars: dict[str, tuple[str,str]]
    publish_path: str

    
class Sha256CacheEntry(TypedDict):
    size: int
    modified: float
    sha256: str


class PDFIndex(TypedDict):
    previous_failure: bool
    file_size: int

    
class SequenceVersion(TypedDict):
    sequence: int


class DocumentStore:
    def __init__(self, load_libraries:bool = True):
        self.current_version: int = 5
        self.log: logging.Logger = logging.getLogger("DocumentStore")
        self.md_tools:dict[str, MarkdownTools] = {}
        self.org_tools:dict[str, OrgmodeTools] = {}
        self.cb_tools:dict[str, CalibreTools] = {}
        self.sync_tools:dict[str, SyncTools] = {}
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
        self.metadata_library: dict[str, MetadataEntry] = {}
        self.pdf_index:dict[str, PDFIndex] = {}
        self.tables:list[DocumentTable] = []
        self.tl:TimeLines = TimeLines()

        self.publish_path: str = os.path.expanduser(self.config['publish_path'])
        if os.path.isdir(self.publish_path) is False:
            os.makedirs(self.publish_path)

        self.storage_path: str = os.path.join(os.path.expanduser("~/.local/share"), "local_research")
        if os.path.isdir(self.storage_path) is False:
            os.makedirs(self.storage_path)

        state_dir = os.path.join(self.storage_path, "state")
        if os.path.isdir(state_dir) is False:
            os.makedirs(state_dir)
        state_glob_dir = os.path.join(self.publish_path, "state")
        if os.path.isdir(state_glob_dir) is False:
            os.makedirs(state_glob_dir)
        self.perf_stats: PerfStats = PerfStats(state_dir, state_glob_dir)
        self.text_document_library_file:str = os.path.join(self.storage_path, "document_library.json")
        self.metadata_library_file:str = os.path.join(self.storage_path, "metadata_library.json")
        self.sequence_file:str = os.path.join(self.storage_path, "version_seq.json")
        self.remote_sequence_file:str = os.path.join(self.publish_path, "version_seq.json")
        self.pdf_cache_path: str = os.path.join(self.storage_path, "pdf_cache")
        if os.path.isdir(self.pdf_cache_path) is False:
            os.makedirs(self.pdf_cache_path, exist_ok=True)
        self.pdf_index_file: str = os.path.join(self.pdf_cache_path, "pdf_index.json")

        if self.local_update_required() is False and load_libraries is True:
            self.load_document_data()
        
        remote, local = self.load_sequence_versions()
        self.log.info(f"DocumentStore initialized: remote data version: {remote}, local version: {local}")
        if self.local_update_required() is True and load_libraries is True:
            self.log.warning("No document data loaded, since local is outdated. Use 'force_load_docs' to override")
            self.log.info("Please use 'import' to acquire the latest data version")

    def load_document_data(self):
        self.load_metadata_library()
        self.load_text_library()            
        errors:list[str] = []
        metadata_library_changed, doc_count, source_file_count  = self.parse_documents(errors)
        if metadata_library_changed is True:
            self.save_metadata_library()
        else:
            self.log.info(f"Metadata unchanged, docs: {doc_count}, sources: {source_file_count}")

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
                    required_sections: list[str] = ['version', 'document_sources', 'sync_targets', 'vars']
                    for section_name in required_sections:
                        if section_name not in config:
                            self.log.error(f'Required section {section_name} not in config {self.config_file}, resetting to defaults!')
                            valid = False
                    if valid is True: 
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
                        }),
                },
                'sync_targets': {
                    'MetaLibrary': SyncTarget({
                        'types': ["calibre"],
                        'path': "~/MetaLibrary",
                        'file_types': ['pdf', 'epub'],
                        'metadata': None,
                        }),
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
        for sync_target_name in config['sync_targets']:
            sync_target = config['sync_targets'][sync_target_name]
            valid = True
            for type in sync_target['types']:
                if type not in ['md_notes', 'orgmode', 'calibre']:
                    self.log.error(f"Sync target {sync_target_name} has invalid type {type}, ignoring this")
                    valid = False
            if os.path.exists(os.path.expanduser(sync_target['path'])) is False:
                valid = False
                self.log.error("Sync target {sync_target_name} destination {sync_target['path']} doesn not exist, ignoring this")
            if valid is False:
                del config['sync_targets'][sync_target_name]
            else:
                self.sync_tools[sync_target_name] = SyncTools(sync_target)
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
        
    def get_vars(self) -> dict[str, tuple[str,str]]:
        return self.config['vars']

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
        if remote > local and remote != 0:
            return True
        else:
            if remote == 0:
                self.log.warning("Failed to get remote version on update-required check! Update may be required after fixing remote data being unavailable!")
                return True
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
        
        relative_path = descriptor[ind+1:]
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
            
        full_path = os.path.join(os.path.expanduser(self.config['document_sources'][source_name]['path']), relative_path)
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
                data = json.load(f)  # pyright:ignore[reportAny]
                self.text_library = {}
                for key, value in data.items():  # pyright:ignore[reportAny]
                    try:
                        self.text_library[key] = TextLibraryEntry(**value)  # pyright:ignore[reportAny]
                    except Exception as e:
                        self.log.error(f"Failed to load text library entry for {key}: {e}")
            if len(self.text_library.keys()) > 0:
                delta = (time.time() - start_time) / len(self.text_library.keys()) * 1000.0
                self.perf_stats.add_perf(f'load text library (1000 recs)', delta)
                
        else:
            self.text_library = {}
        if os.path.exists(self.pdf_index_file):
            start_time = time.time()
            with open(self.pdf_index_file, "r") as f:
                self.pdf_index = json.load(f)
            if len(self.pdf_index.keys()) > 0:
                delta = (time.time() - start_time) / len(self.pdf_index.keys()) * 1000000.0
                self.perf_stats.add_perf(f'load pdf cache (10^6 recs)', delta)
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

    def get_metadata(self, descriptor: str) -> MetadataEntry | None:
        if descriptor in self.metadata_library:
            return self.metadata_library[descriptor]
        
        # Handle Calibre (directory-based metadata)
        source_name, _ = self.get_source_name_and_path_from_descriptor(descriptor)
        if source_name:
            source_type = self.get_source_type_from_name(source_name)
            if source_type == 'calibre':
                # Try parent directory
                last_slash = descriptor.rfind('/')
                if last_slash != -1:
                    parent_descriptor = descriptor[:last_slash]
                    if parent_descriptor in self.metadata_library:
                        return self.metadata_library[parent_descriptor]
        return None

    def load_metadata_library(self):
        self.log.info("Loading metadata_library data...")
        if os.path.exists(self.metadata_library_file):
            with open(self.metadata_library_file, "r") as f:
                data = json.load(f)  # pyright:ignore[reportAny]
                self.metadata_library = {}
                for key, value in data.items():  # pyright:ignore[reportAny]
                    try:
                        self.metadata_library[key] = MetadataEntry(**value)  # pyright:ignore[reportAny]
                    except Exception as e:
                        self.log.error(f"Failed to load metadata for {key}: {e}")
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
                # Convert Pydantic models to dicts for JSON serialization
                json.dump(self.metadata_library, temp_file)
            os.replace(temp_path, self.metadata_library_file)
            self.log.info(f"Metadata library saved to {self.metadata_library_file}")
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
            if pymupdf is not None:
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
        base, ext_with_dot = os.path.splitext(filename)
        ext = ext_with_dot[1:].lower() if ext_with_dot else ""
        if '.caltrash' in root_path:
            return True
        if ext not in valid_types: 
            return True
        if base.startswith('.#'):  # Emacs tmp
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


    def parse_documents(self, errors:list[str], abort_check_callback:Callable[[], bool]|None=None, force:bool=False) ->tuple[bool, int, dict[str, int]]:
        self.tables = []
        self.tl.tl_events = []
        metadata_library_changed = False
        doc_count = 0
        source_file_count:dict[str,int]={}
        old_metadata_keys:list[str] = list(self.metadata_library.keys())
        
        for source_name in self.config['document_sources']:
            source = self.config['document_sources'][source_name]
            if abort_check_callback is not None and abort_check_callback() is True: 
                return metadata_library_changed, doc_count, source_file_count
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
                        sha256_hash = self.get_sha256(doc_path)
                        md_content: str|None = None
                        metadata: MetadataEntry | None = None
                        if descriptor not in self.metadata_library:
                            if source['type'] == 'md_notes':
                                metadata, md_content, _meta_changed, _mandatory_changed = self.md_tools[source_name].parse_markdown(doc_path, sha256_hash, descriptor, text)
                            elif source['type'] == 'orgmode':
                                metadata, _prefix, _content, _meta_changed, _mandatory_changed = self.org_tools[source_name].parse_orgmode(doc_path, sha256_hash, descriptor, text)
                            else:
                                self.log.error(f"Invalid type {source['type']} at {descriptor}")
                                continue                                    
                            metadata_library_changed = True
                            self.metadata_library[descriptor] = metadata
                            self.log.info(f"New metadata entry {descriptor}")
                        else:
                            for doc_repr in self.metadata_library[descriptor]['representations']:
                                if doc_repr['doc_descriptor'] == descriptor and doc_repr['hash'] != sha256_hash:
                                    self.log.warning(f"{descriptor} has changed!")
                                    if source['type'] == 'md_notes':
                                        metadata, md_content, meta_changed, mandatory_changed = self.md_tools[source_name].parse_markdown(doc_path, sha256_hash, descriptor, text)
                                    elif source['type'] == 'orgmode':
                                        metadata, _prefix, _content, meta_changed, mandatory_changed = self.org_tools[source_name].parse_orgmode(doc_path, sha256_hash, descriptor, text)
                                    else:
                                        self.log.error(f"Invalid type {source['type']} at {descriptor}")
                                        continue
                                    if meta_changed or mandatory_changed:
                                        metadata_library_changed = True
                                        self.metadata_library[descriptor] = metadata
                                        self.log.info(f"Metadata changed for {descriptor}")
                                else:
                                    if source['type'] == 'md_notes':
                                        metadata = self.metadata_library[descriptor]
                                        _header, md_content = self.md_tools[source_name].split_header_content(text)
                        if md_content is not None and metadata is not None:
                            uuid:str = metadata['uuid']
                            self.tables += self.md_tools[source_name].get_tables(md_content, doc_path, uuid)
                        else:
                            if source['type'] == 'md_notes':
                                if md_content is None:
                                    self.log.warning(f"{doc_path} has no content")
                                if metadata is None:
                                    self.log.warning(f"{doc_path} has no metadata")
                        if descriptor in self.metadata_library:
                            if descriptor in old_metadata_keys:
                                old_metadata_keys.remove(descriptor)
                        
            elif source['type'] == 'calibre':
                for root, _dirs, files in os.walk(source_path, topdown=True, onerror=lambda e: errors.append(f"Cannot access directory {e.filename}: {e.strerror}")): # pyright:ignore[reportAny]
                    for filename in files:
                        if self.skip_file_in_sync(root, filename, ['opf']) is True:
                            continue
                        if filename == 'metadata.opf':
                            filepath = os.path.join(root, filename)
                            main_descriptor = self.get_descriptor_from_path(root)
                            existing_metadata = None
                            if not force:
                                existing_metadata = self.metadata_library.get(main_descriptor)
                            metadata = self.cb_tools[source_name].parse_calibre_metadata(filepath, main_descriptor, existing_metadata)
                            if metadata is not None:
                                # Calculate hashes for representations
                                for f in files:
                                    ext = os.path.splitext(f)[1].lower().lstrip('.')
                                    if ext in ['txt', 'pdf', 'md']:
                                        f_path = os.path.join(root, f)
                                        f_hash = self.get_sha256(f_path)
                                        for rep in metadata['representations']:
                                            if rep['format'] == ext and rep['hash'] == "":
                                                rep['hash'] = f_hash

                                if main_descriptor not in self.metadata_library:
                                    self.metadata_library[main_descriptor] = metadata
                                    metadata_library_changed = True
                                    self.log.info(f"New metadata for {main_descriptor}")
                                else:
                                    if self.metadata_library[main_descriptor] != metadata:
                                        self.log.info(f"Metadata update for {main_descriptor}")
                                        self.metadata_library[main_descriptor] = metadata
                                        metadata_library_changed = True                                        
                                source_file_count[source_name] += 1
                                doc_count += 1
                                if main_descriptor in self.metadata_library:
                                    if main_descriptor in old_metadata_keys:
                                        old_metadata_keys.remove(main_descriptor)
            else:
                self.log.error(f"Ignoring unknown source_type {source['type']}")
                continue

        if len(old_metadata_keys) > 0:
            for key in old_metadata_keys:
                self.log.info(f"Removing metadata for orphan {key}")
                del self.metadata_library[key]
                metadata_library_changed = True
                
        self.tl.add_notes_events(self.tables)
        return metadata_library_changed, doc_count, source_file_count
        
    def keyword_search(self, query: str, source: str | None = None) -> list[SearchResultEntry]:
        results: list[SearchResultEntry] = []
        keywords = query.split()
        if not keywords:
            return results

        if len(self.metadata_library) == 0 or len(self.text_library) == 0:
            self.load_document_data()
        for descriptor, metadata in self.metadata_library.items():
            if source is not None:
                doc_source, _ = self.get_source_name_and_path_from_descriptor(descriptor)
                if doc_source.lower() != source.lower():
                    continue

            # Construct a searchable text blob from metadata
            searchable_text_parts:list[str] = []
            searchable_text_parts.append(metadata['title'])
            searchable_text_parts.extend([str(a) for a in metadata['authors']])
            searchable_text_parts.extend([str(t) for t in metadata['tags']])
            searchable_text_parts.append(metadata['description'])
            searchable_text_parts.append(metadata['context'])
            
            searchable_text = " ".join(searchable_text_parts)
            
            if SearchTools.match(searchable_text, keywords):
                doc_hash = ""
                if metadata['representations']:
                    for rep in metadata['representations']:
                        if rep['hash']:
                            doc_hash = rep['hash']
                            break
                
                text_entry = None
                if doc_hash and doc_hash in self.text_library:
                    text_entry = self.text_library[doc_hash]

                if text_entry:
                     # Format a text summary for display
                    display_text = f"Title: {metadata['title']}\n"
                    if metadata['authors']:
                        display_text += f"Authors: {', '.join([str(a) for a in metadata['authors']])}\n"
                    if metadata['tags']:
                        display_text += f"Tags: {', '.join([str(t) for t in metadata['tags']])}\n"
                    if metadata['description']:
                        display_text += f"Description: {metadata['description'][:200]}...\n"
                    results.append(SearchResultEntry(
                         cosine=1.0, 
                         hash=doc_hash, 
                         chunk_index=0, 
                         entry=text_entry, 
                         text=display_text, 
                         significance=None
                     ))

        return results

    def sync_texts(self, force:bool, retry:bool=False, progress_callback:Callable[[ProgressState], None ]|None=None, abort_check_callback:Callable[[], bool]|None=None) -> list[str]:
        if len(self.metadata_library) == 0 or len(self.text_library) == 0:
            self.load_document_data()
        errors:list[str] = []
        if self.local_update_required() is True:
            if force is False:
                self.log.warning("Please first update local data using 'import', since remote has newer data! (use 'force' to override)")
                errors.append("Please first update local data using 'import', since remote has newer data! (use 'force' to override)")
                return errors
            else:
                self.log.warning("Override active, syncing even so remote has newer data!")
        text_library_changed = False
        pdf_index_changed = False
        old_text_library_size = len(list(self.text_library.keys()))
        last_saved = time.time()

        existing_hashes: list[str] = list(self.text_library.keys())
        pdf_cache_hits = 0
        duplicate_count = 0
        last_status = time.time()

        current_doc_count = 0

        metadata_library_changed, doc_count, source_file_count = self.parse_documents(errors, abort_check_callback, force)
        
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
                    sha256_hash = self.get_sha256(full_path)
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
                                progress_state = ProgressState(issues=len(errors), state=state, percent_completion=perc, vars={}, finished=False)
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
                                progress_state = ProgressState(issues=len(errors), state=state, percent_completion=perc, vars={}, finished=False)
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
                                progress_state = ProgressState(issues=len(errors), state=state, percent_completion=perc, vars={}, finished=False)
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
                                            
                    self.text_library[sha256_hash] = TextLibraryEntry(source_name=source_name, descriptor=descriptor, text=current_text)
                    text_library_changed = True

                    if time.time() - last_saved > 180:
                        self.save_text_library()
                        text_library_changed = False
                        last_saved =  time.time()
            if progress_callback is not None:
                progress_state = ProgressState(issues=len(errors), state="", percent_completion=1.0, vars={}, finished=True)
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
        else:
            self.log.info("Metadata unchanged")
        if text_library_changed is True or pdf_index_changed is True:
            self.save_text_library()
            self.log.info(f"Library size {old_text_library_size} -> {new_text_library_size}")
        else:
            self.log.info(f"No text library changes")
        return errors

    def check_sha256_cache(self, clean:bool) -> tuple[int, int, int]:
        debris = 0
        deleted = 0
        entries = 0
        for filename in list(self.sha256_cache):
            entries += 1
            if os.path.exists(filename) is False:
                debris += 1
                if clean is True:
                    del self.sha256_cache[filename]
                    deleted += 1
                    self.sha256_cache_changed = True
        return entries, debris, deleted
        
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
        self.perf_stats.sync_perf()
        if len(self.metadata_library) == 0 or len(self.text_library) == 0:
            self.log.error("No metadata or text library found, cannot publish!")
            return False
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
