import logging
import os
import uuid
from datetime import datetime, date
from typing import TypedDict, cast, TypeVar, Any

from research_tools import ResearchTools



class TextLibraryEntry(TypedDict):
    source_name: str
    descriptor: str
    text: str


class SearchResultEntry(TypedDict):
    cosine: float
    hash: str
    chunk_index: int
    entry: TextLibraryEntry
    text: str|None
    significance: list[float]|None



class DocumentRepresentationEntry(TypedDict):
    doc_descriptor: str
    hash: str
    format: str
    doc_date: str

    
class MetadataEntry(TypedDict):
    uuid: str
    representations: list[DocumentRepresentationEntry]
    authors: list[str]
    identifiers: list[str]
    languages: list[str]
    context: str
    creation_date: str
    publication_date: str
    publisher: str
    series: str
    tags: list[str]
    title: str
    title_sort: str
    normalized_filename: str
    description: str
    icon: str


class ProgressState(TypedDict):
    issues: int
    state: str
    percent_completion: float
    vars: dict[str, str]
    finished: bool


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


class ResearchMetadata:
    def __init__(self):
        self.log:logging.Logger = logging.getLogger("ResearchMetadata")
        self.rt:ResearchTools = ResearchTools()

    def normalize_metadata(self, root_folder:str|None, filepath:str, hash:str, descriptor:str, meta_dict:dict[str, Any]) -> tuple[MetadataEntry, bool, bool]:  # pyright:ignore[reportExplicitAny]
        changed:bool = False
        mandatory_changed:bool = False
        mandatory_fields: list[str] = ['creation', 'context', 'uuid']

        T = TypeVar('T')
        
        def get_meta_field(source:dict[str, Any], source_field:str, default:T, alt_source_field:str|None=None) -> T:  #pyright:ignore[reportExplicitAny]
            result: T
            nonlocal changed
            nonlocal mandatory_fields
            nonlocal mandatory_changed
            some_result:date|datetime|T
            if source_field not in source:
                if alt_source_field is not None:
                    if alt_source_field not in source:
                        result = default
                        changed = True
                        if source_field in mandatory_fields:
                            mandatory_changed = True
                    else:
                        some_result = source[alt_source_field]  #pyright:ignore[reportAny]
                        if type(some_result) == date or type(some_result) == datetime:
                            result = cast(T, some_result.isoformat())
                        else:
                            result = cast(T, some_result)
                        changed = True
                        if source_field in mandatory_fields:
                            mandatory_changed = True
                else:
                    result = default
                    changed = True
                    if source_field in mandatory_fields:
                        mandatory_changed = True
            else:
                some_result = cast(T, source[source_field])
                if type(some_result) == date or type(some_result) == datetime:
                    result = cast(T, some_result.isoformat())
                else:
                    result = cast(T, some_result)
                changed = False
            if result is None:
                result = cast(T, "")
            return result

        def get_meta(source:dict[str, Any], source_field:str, default:str, alt_source_field:str|None=None) -> str:  #pyright:ignore[reportExplicitAny]
            if type(default) not in [str, list]:
                self.log.error(f"Invalid type for default for {source}: {type(default)}")
                raise ValueError
            return get_meta_field(source, source_field, default, alt_source_field)
        
        def get_meta_list(source:dict[str, Any], source_field:str, default:list[str], alt_source_field:str|None=None) -> list[str]:  #pyright:ignore[reportExplicitAny]
            if type(default) not in [str, list]:
                self.log.error(f"Invalid type for default for {source}: {type(default)}")
                raise ValueError
            return get_meta_field(source, source_field, default, alt_source_field)

        doc_uuid = get_meta(meta_dict, 'uuid', str(uuid.uuid4()))
        authors = get_meta_list(meta_dict, 'authors', [])
        cd: datetime = self.rt.get_note_creation_date(root_folder, filepath)
        default_creation_date: str = cd.isoformat()
        creation_date = get_meta(meta_dict, 'creation', default_creation_date)
        identifiers = get_meta_list(meta_dict, 'identifiers', [])
        languages = get_meta_list(meta_dict, 'languages', [])
        ind1 = descriptor.find('/')
        ind2 = descriptor.rfind('/')
        if ind1 != -1 and ind2 != -1:
            ind1 += 1
            default_context = descriptor[ind1:ind2]
        else:
            default_context = ""
        context = get_meta(meta_dict, 'context', default_context)
        publication_date = get_meta(meta_dict, 'pubdate', "")
        publisher = get_meta(meta_dict, 'publisher', default="")
        series = get_meta(meta_dict, 'series', "")
        tags = get_meta_list(meta_dict, 'tags', [])
        title = get_meta(meta_dict, 'title', "")
        title_sort = get_meta(meta_dict, 'title_sort', "")
        default_normalized_filename = os.path.basename(filepath)  ## XXX!
        normalized_filename = get_meta(meta_dict, 'normalized_filename', default_normalized_filename)
        description = get_meta(meta_dict, 'description', '')
        icon = get_meta(meta_dict, 'icon', '')

        doc_representation = DocumentRepresentationEntry({'doc_descriptor': descriptor,
                                                          'hash': hash,
                                                          'format': 'md',
                                                          'doc_date': creation_date})
        metadata:MetadataEntry = MetadataEntry({'uuid': doc_uuid,
                                  'representations': [doc_representation],
                                  'authors': authors,
                                  'identifiers': identifiers,
                                  'languages': languages,
                                  'context': context,
                                  'creation_date': creation_date,
                                  'publication_date': publication_date,
                                  'publisher': publisher,
                                  'series': series,
                                  'tags': tags,
                                  'title': title,
                                  'title_sort': title_sort,
                                  'normalized_filename': normalized_filename,
                                  'description': description,
                                  'icon': icon,
                                  })

        for key in metadata:
            if type(metadata[key]) not in [str, list]:  # pyright:ignore[reportUnknownArgumentType]
                self.log.error(f"Field {key} has unexpected type {type(metadata[key])} in {metadata}")  # pyright:ignore[reportUnknownArgumentType]
                raise ValueError
        return metadata, changed, mandatory_changed
