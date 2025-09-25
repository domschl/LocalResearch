import logging
import datetime
import os
import subprocess
import uuid
import yaml
from typing import Any, cast, TypeVar

from vector_store import MetadataEntry, DocumentRepresentationEntry


class MarkdownTools:
    def __init__(self, notes_folder:str|None = None):
        self.log: logging.Logger = logging.getLogger("MarkdownTools")
        self.notes_folder:str|None = notes_folder

    def assemble_markdown(self, metadata: dict[str, Any], content: str) -> str:  # pyright: ignore[reportExplicitAny]
        if metadata  == {}:
            return content
        header = yaml.dump(metadata, default_flow_style=False, indent=2)
        return f"---\n{header}---\n{content}"


    def _note_get_file_creation_date_from_git(self, filepath: str) -> datetime.datetime | None:
        if self.notes_folder is None:
            return None
        try:
            creation_date = subprocess.check_output(
                args = [
                    "git",
                    "-C",
                    self.notes_folder,
                    "--no-pager",
                    "log",
                    "--follow",
                    "--format=%aI",
                    "--reverse",
                    filepath,
                ]
            )
            cr_date = creation_date.decode("utf-8").split("\n")[0]
            # datetime parse:
            try:
                dt = datetime.datetime.strptime(cr_date, "%Y-%m-%dT%H:%M:%S%z")
                return dt
            except Exception as e:
                self.log.debug(
                    f"Error file {filepath}, failed parsing date {cr_date}: {e}"
                )
                return None
        except Exception as _:
            return None

    def _note_get_file_modification_date(self, filepath: str) -> datetime.datetime | None:
        try:
            stat = os.stat(filepath)
            dt = datetime.datetime.fromtimestamp(stat.st_mtime)
            return dt
        except Exception as e:
            self.log.warning(f"Error getting file modification date {filepath}: {e}")
            return None

    def get_note_creation_date(self, filepath: str) -> datetime.datetime:
        dt_git: datetime.datetime | None = self._note_get_file_creation_date_from_git(filepath)
        dt_stat = self._note_get_file_modification_date(filepath)
        # self.log.info(f"Creation date for {filepath} from git: {dt_git}, from stat: {dt_stat}")
        if dt_git is not None:
            return dt_git
        if dt_stat is not None:
            return dt_stat
        return datetime.datetime.now()
    
    def parse_markdown(self, filepath:str, hash:str, descriptor:str, md_text:str) -> tuple[MetadataEntry, str, bool]:
        state:int = 0
        lines: list[str] = md_text.split("\n")
        frontmatter_lines: list[str] = []
        frontmatter: str = ""
        content_lines: list[str] = []
        content: str = ""
        start = True
        for line in lines:
            if state == 0:
                if line == "---" and start is True:
                    state = 1
                else:
                    content_lines.append(line)
                start = False
            elif state == 1:
                if line == "---":
                    state = 2
                else:
                    frontmatter_lines.append(line)
            elif state == 2:
                if len(content_lines) == 0:
                    if line == "":
                        continue
                content_lines.append(line)
        frontmatter = "\n".join(frontmatter_lines)
        content = "\n".join(content_lines) + "\n"
        try:
            yaml_metadata: dict[str, Any] = yaml.safe_load(frontmatter)  # pyright: ignore[reportAny, reportExplicitAny]
        except Exception as e:
            self.log.error(f"Error parsing frontmatter: {e}")
            yaml_metadata = {}


        changed:bool = False

        T = TypeVar('T')
        
        def get_meta_field(source:dict[str, Any], source_field:str, default:T, alt_source_field:str|None=None) -> T:  #pyright:ignore[reportExplicitAny]
            result: T
            nonlocal changed
            if source_field not in source:
                if alt_source_field is not None:
                    if alt_source_field not in source:
                        result = default
                        changed = True
                    else:
                        result = cast(T, source[alt_source_field])
                        changed = True
                else:
                    result = default
                    changed = True
            else:
                result = cast(T, source[source_field])
                changed = False
            return result

        def get_meta(source:dict[str, Any], source_field:str, default:str, alt_source_field:str|None=None) -> str:  #pyright:ignore[reportExplicitAny]
            return get_meta_field(source, source_field, default, alt_source_field)
        
        def get_meta_list(source:dict[str, Any], source_field:str, default:list[str], alt_source_field:str|None=None) -> list[str]:  #pyright:ignore[reportExplicitAny]
            return get_meta_field(source, source_field, default, alt_source_field)

        doc_uuid = get_meta(yaml_metadata, 'uuid', str(uuid.uuid4()))
        authors = get_meta_list(yaml_metadata, 'authors', [])
        cd: datetime.datetime = self.get_note_creation_date(filepath)
        default_creation_date: str = cd.isoformat()
        creation_date = get_meta(yaml_metadata, 'creation_date', default_creation_date, 'creation')
        identifiers = get_meta_list(yaml_metadata, 'identifiers', [])
        languages = get_meta_list(yaml_metadata, 'languages', [])
        ind1 = descriptor.find('/')
        ind2 = descriptor.rfind('/')
        if ind1 != -1 and ind2 != -1:
            ind1 += 1
            default_context = descriptor[ind1:ind2]
        else:
            default_context = ""
        context = get_meta(yaml_metadata, 'context', default_context)
        publication_date = get_meta(yaml_metadata, 'publication_date', "", 'pubdate')
        series = get_meta(yaml_metadata, 'series', "")
        tags = get_meta_list(yaml_metadata, 'tags', [])
        title = get_meta(yaml_metadata, 'title', "")
        title_sort = get_meta(yaml_metadata, 'title_sort', "")
        default_normalized_filename = os.path.basename(filepath)  ## XXX!
        normalized_filename = get_meta(yaml_metadata, 'normalized_filename', default_normalized_filename)
        description = get_meta(yaml_metadata, 'description', '')
        icon = get_meta(yaml_metadata, 'icon', '')

        doc_representation = DocumentRepresentationEntry({'doc_descriptor': descriptor,
                                                          'hash': hash,
                                                          'format': 'md',
                                                          'creation_date': creation_date})
        metadata:MetadataEntry = MetadataEntry({'uuid': doc_uuid,
                                  'representations': [doc_representation],
                                  'authors': authors,
                                  'identifiers': identifiers,
                                  'languages': languages,
                                  'context': context,
                                  'creation_date': creation_date,
                                  'publication_date': publication_date,
                                  'series': series,
                                  'tags': tags,
                                  'title': title,
                                  'title_sort': title_sort,
                                  'normalized_filename': normalized_filename,
                                  'description': description,
                                  'icon': icon,
                                  })
        
        return metadata, content, changed
