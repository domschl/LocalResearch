import logging
import os
import json
import uuid
import time
import numpy as np

import torch
from sentence_transformers import SentenceTransformer

from typing import TypedDict, cast, override
import pymupdf  # pyright: ignore[reportMissingTypeStubs]


class TqSource(TypedDict):
    name: str
    tqtype: str
    path: str
    file_types: list[str]

class IcotqConfig(TypedDict):
    icotq_path: str
    tq_sources: list[TqSource]
    ebook_mirror: str
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
                'ebook_mirror': '~/MetaLibrary',
                'embeddings_model_name': 'nomic-embed-text-v2-moe',
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
                    'emb_dim': 0,
                    'max_input_token': 0,
                    'chunk_size': 2048,
                    'chunk_overlap': 2048 // 3
                }
            ]
            with open(model_list_path, 'w') as f:
                json.dump(self.model_list, f)
            self.log.warning(f"Initialized {model_list_path} with default embeddings model list. Please verify.")
        self.current_model: EmbeddingsModel | None = None
        self.engine: SentenceTransformer | None = None
        self.device: str | None = None
        self.embeddings_matrix: torch.Tensor | None = None
        if self.config['embeddings_model_name'] != "":
            _ = self.load_model(self.config['embeddings_model_name'], self.config['embeddings_device'], self.config['embeddings_model_trust_code'])
        config_subdirs = ['Texts', 'Embeddings', 'PDFTextCache', 'EpubTextCache']
        for cdir in config_subdirs:
            full_path = os.path.join(self.root_path, cdir)
            if os.path.isdir(full_path) is False:
                os.makedirs(full_path)
        self.embeddings_path: str = os.path.join(self.root_path, "Embeddings")
        for source in self.config['tq_sources']:
            valid:bool = True
            known_types: list[str] = ['txt', 'epub', 'md', 'pdf']
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
            json.dump(self.config,f)
        self.log.info(f"Configuration changes saved to {self.config_file}")

    def save_pdf_cache_state(self):
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        pdf_cache_index = os.path.join(pdf_cache, "pdf_index.json")
        with open(pdf_cache_index, 'w') as f:
            json.dump(self.pdf_index, f)

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
            self.log.error("No embeddings available to save")
            return False

    def load_tensor(self) -> bool:
        if self.current_model is None or self.device is None:
            self.log.error("Can't save embeddings tensor: no current model information available!")
            return False
        embeddings_tensor_file = os.path.join(self.embeddings_path, f"embeddings_{self.current_model['model_name']}.pt")
        map_location = torch.device(self.device)
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
            self.log.info(f"Matrix: {self.embeddings_matrix.shape}, chunks: {sum}, texts: {len(self.lib)}")
            if sum != self.embeddings_matrix.shape[0]:
                self.log.warning(f"Embeddings-matrix incompatible with text library! Sum: {sum}, EmbMat: {self.embeddings_matrix.shape}")
            return True
        else:
            self.log.warning("No embeddings available!")
            return False

    def write_library(self):
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        with open(lib_path, 'w') as f:
            json.dump(self.lib, f)
        self.save_pdf_cache_state()

    def get_pdf_text(self, desc:str, full_path:str) -> tuple[str | None, bool]:
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        text: str | None = None
        if desc in self.pdf_index:
            cur_file_size = os.path.getsize(full_path)
            if cur_file_size == self.pdf_index[desc]['file_size'] and self.pdf_index[desc]['previous_failure'] is False:
                try:
                    with open(self.pdf_index[desc]['filename'], 'r') as f:
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
                cache_filename = os.path.join(pdf_cache, str(uuid.uuid4()))
                failure = False
                self.log.info(f"Importing and caching PDF {full_path}")
            pdf_ind: PDFIndex = {
                'filename': cache_filename,
                'file_size': os.path.getsize(full_path),
                'previous_failure': failure
            }
            if failure is False and text is not None:
                with open(pdf_ind['filename'], 'w') as f:
                    _ = f.write(text)
            self.pdf_index[desc] = pdf_ind
            # self.save_pdf_cache_state()
            self.log.info(f"Added {desc} to PDF cache, size: {len(self.pdf_index.keys())}, failure: {failure}")
            changed = True
        return text, changed

    def import_texts(self, max_imports: int|None = None):
        if len(self.config['tq_sources']) == 0:
            self.log.error(f"No valid sources defined in config, can't import")
            return
        lib_changed = False
        for source in self.config['tq_sources']:
            source_path = os.path.expanduser(source['path'])
            for root, _dir, files in os.walk(source_path):
                for filename in files:
                    parts = os.path.splitext(filename)
                    file_base = parts[0]
                    if len(parts[1]) > 0:
                        ext = parts[1][1:].lower()  # remove leading '.'
                    else:
                        ext = ""
                    if ext not in source['file_types']:
                        continue
                    alt_exists = False
                    if ext in ['epub', 'pdf']:
                        for alt in ['txt', 'epub']:
                            if alt == ext:
                                continue
                            alt_file = os.path.join(root, file_base + '.' + alt)
                            if os.path.exists(alt_file):
                                # self.log.info(f"Better format {alt} exists for {filename}")
                                alt_exists = True
                        if alt_exists is True:  # better format of same file exist, so skip this one
                            continue                    
                    full_path = os.path.join(root, filename)
                    desc_path = "{"+ source['name'] + "}" + full_path[len(source_path):]
                    in_lib = False
                    for entry in self.lib:
                        if entry['desc_filename'] == desc_path:
                            # Check if changed!
                            in_lib = True
                            break
                    if in_lib is False:
                        text = None
                        if ext in ['md', 'py', 'txt']:
                            with open(full_path, 'r') as f:
                                text = f.read()
                        elif ext == 'pdf':
                            text, changed = self.get_pdf_text(desc_path, full_path)
                            if changed is True:
                                lib_changed = True
                        else:
                            self.log.error(f"Unsupported conversion {ext} to text at {desc_path}")
                            continue
                        if text is not None:
                            entry: LibEntry = LibEntry({
                                'source_name': source['name'],
                                'desc_filename': desc_path,
                                'filename': full_path,
                                'text': text,
                                'emb_ptrs': {}
                            })
                            self.lib.append(entry)
                            lib_changed = True
                            if max_imports is not None and len(self.lib) >= max_imports:
                                self.write_library()
                                self.log.warning(f"Import reached max {max_imports}, library: {len(self.lib)} entries")
                                return
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
                    if device=='auto':
                        if torch.cuda.is_available():
                            self.engine = self.engine.to(torch.device('cuda'))
                            self.device = 'cuda'
                        elif torch.backends.mps.is_available():
                            self.engine = self.engine.to(torch.device('mps'))
                            self.device = 'mps'
                        else:
                            self.engine = self.engine.to(torch.device('cpu'))
                            self.device = 'cpu'
                    else:
                        self.engine = self.engine.to(torch.device(device))
                        self.device = device
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

    def generate_embeddings(self, save_every_sec:int = 180):
        if self.current_model is None or self.engine is None:
            self.log.error("No current embeddings model loaded!")
            return
        start_time: float = time.time()
        for ind, entry in enumerate(self.lib):
            self.log.info(f"Embedding: {ind+1}/{len(self.lib)}")
            if self.current_model['model_name'] in entry['emb_ptrs']:
                continue
            text_chunks = self.get_chunks(entry['text'], self.current_model['chunk_size'], self.current_model['chunk_overlap'])
            if len(text_chunks) == 0:
                self.log.error(f"Cannot encode empty text list at {entry['desc_filename']}")
                continue
            self.log.info(f"Encoding {len(text_chunks)} chunks...")
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

    def yellow_line_it(self, text: str, search_embeddings: torch.Tensor, context:int=16, context_steps:int=1) -> np.typing.NDArray[np.float32]:
        if self.embeddings_matrix is None or self.engine is None:
            self.log.error("No embeddings available at yellow-lining!")
            return np.array([], dtype=np.float32)
        clr: list[str] = []
        for i in range(0, len(text), context_steps):
            i0 = i - context // 2
            i1 = i + context // 2
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

    def search(self, search_text:str, max_results:int=2, yellow_liner:bool=False, context:int=16, context_steps:int=4, compress:str="none"):
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
            if compress == "light":
                new_chunk = chunk
                old_chunk = None
                while new_chunk != old_chunk:
                    old_chunk = new_chunk
                    new_chunk = old_chunk.replace("  ", " ").replace("\n\n", "\n")
                chunk = new_chunk
            elif compress == "full":
                new_chunk = chunk
                old_chunk = None
                while new_chunk != chunk:
                    old_chunk = new_chunk
                    new_chunk = old_chunk.replace("\n", " ").replace("\r"," ").replace("\t", " ").replace("  ", " ")
                chunk = new_chunk
            if yellow_liner is True:
                yellow_liner_weights = self.yellow_line_it(chunk, search_embeddings, context=context, context_steps=context_steps)
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
        return search_results
 