import logging
import os
import yaml
from typing import Any

from research_defs import MetadataEntry, ResearchMetadata
from research_tools import ResearchTools, DocumentTable


class MarkdownTools:
    def __init__(self, notes_folder:str):
        self.log: logging.Logger = logging.getLogger("MarkdownTools")
        self.notes_folder:str|None = os.path.expanduser(notes_folder)
        self.rt:ResearchTools = ResearchTools()
        self.rm:ResearchMetadata = ResearchMetadata()

    def parse_markdown(self, filepath:str, hash:str, descriptor:str, md_text:str) -> tuple[MetadataEntry, str, bool, bool]:
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
            yaml_metadata_raw: dict[str, Any]|Any = yaml.safe_load(frontmatter)  # pyright: ignore[reportAny, reportExplicitAny]
            if yaml_metadata_raw is None:
                self.log.warning(f"{filepath} has no metadata")
                yaml_metadata = {}
            else:
                yaml_metadata = yaml_metadata_raw
        except Exception as e:
            self.log.error(f"Error parsing frontmatter: {e}")
            yaml_metadata = {}
        # print("YAML: ", yaml_metadata)
        
        metadata, changed, mandatory_changed = self.rm.normalize_metadata(self.notes_folder, filepath, hash, descriptor, yaml_metadata)
        return metadata, content, changed, mandatory_changed

    def assemble_markdown(self, metadata: dict[str, Any]|MetadataEntry, content: str) -> str:  # pyright: ignore[reportExplicitAny]
        if metadata  == {}:
            return content
        header = yaml.dump(metadata, default_flow_style=False, indent=2)
        return f"---\n{header}---\n{content}"

    def assemble_markdown_ext(self, metadata:MetadataEntry, content: str) -> str:
        # XXX expand possible optional entries in MetadataEntry (not yet defined)
        return self.assemble_markdown(metadata, content)
    
    def get_links(self, content:str) -> list[str]:
        links: list[str] = []
        for line in content.split("\n"):
            # Find links of format [[link]] or [[link|alias]]
            ind = line.find("[[")
            while ind >= 0:
                ind2a = line.find("]]", ind)
                ind2b = line.find("|", ind)
                if ind2a <= 0:
                    ind = line.find("[[", ind + 2)
                    continue
                if ind2b > 0 and ind2b < ind2a:
                    ind2 = ind2b
                else:
                    ind2 = ind2a
                link = line[ind + 2 : ind2].strip().lower()
                links.append(link)
                ind = line.find("[[", ind2)
        return links

    def get_tables(self, content: str, filepath:str, note_uuid:str | None=None) -> list[DocumentTable]:
        tables: list[DocumentTable] = []
        lines: list[str] = content.split("\n")
        table_state = 0
        rows: list[list[str]] = []
        columns: list[str] = []
        metadata: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]
        for line in lines:
            line = line.strip()
            if "<!--" in line:
                metadata = {}
                try:
                    meta = line.split("<!--")[1].split("-->")[0]
                except Exception as _:
                    meta = ""
                if len(meta) > 0:
                    key = None
                    value = None
                    if ":" in meta:
                        try:
                            key, value = meta.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            metadata[key] = value
                        except Exception as _:
                            pass

            if table_state == 0:
                if line.startswith("```"):
                    if line.lower().startswith("```indraevent"):
                        columns = []
                        rows = []
                        table_state = 6
                    else:
                        columns = []
                        rows = []
                        table_state = 4
                    continue
                if line.startswith("$$"):  # start of latex block
                    table_state = 5
                    columns = []
                    rows = []
                    continue
                if line.startswith("|"):
                    if not line.endswith("|"):
                        self.log.warning(
                            f"Table line does not end with '|', >{line}<, malformed tabled! Ignoring data."
                        )
                        return tables
                    table_state = 1
                    columns = [col.strip() for col in line[1:].split("|")][:-1]
                    rows = []
                    continue
            elif table_state == 1:
                if line.startswith("|"):
                    for c in line:
                        if c not in ["|", " ", "-", ":", ">", "<"]:
                            table_state = 0
                            columns = []
                            rows = []
                            metadata = {}
                            continue
                    table_state = 2
                    continue
                else:
                    columns = []
                    rows = []
                    metadata = {}
                    table_state = 0
                    continue
            elif table_state == 2:
                if line.startswith("|"):
                    row = [col.strip() for col in line[1:].split("|")][:-1]
                    if len(row)>0 and row[0] == "Date":
                        print(f"Bad line in middle of table: {line}")
                    rows.append(row)
                    continue
                else:
                    table_state = 0
                    subfolders = None
                    # Strip extension from filepath
                    filepath_noext = os.path.splitext(filepath)[0]
                    if self.notes_folder is not None and filepath_noext.startswith(self.notes_folder):
                        subfolders = filepath_noext[len(self.notes_folder) + 1 :]
                        if len(columns) > 1 and columns[0] == "Date":
                            subfolders = f"{subfolders}/{columns[1]}"
                            bad_chars = [" ", "#", "+", "$", ".", ",", ":"]
                            for bc in bad_chars:
                                subfolders = subfolders.replace(bc, "_")
                            elim_chars = ["*"]
                            for ec in elim_chars:
                                subfolders = subfolders.replace(ec, "")
                            subfolders = subfolders.replace("__", "_")
                            if subfolders.endswith("_"):
                                subfolders = subfolders[:-1]
                        else:
                            subfolders = None
                    else:
                        self.log.error(
                            f"Unexpected filepath {filepath} not in notes folder {self.notes_folder}"
                        )
                        subfolders = None
                    if len(rows) > 0:
                        if "domain" not in metadata and subfolders is not None:
                            metadata["domain"] = f"{subfolders}"
                        # if len(metadata.keys()) > 0:
                        #     print(f"Table metadata: {metadata}")
                        if note_uuid is None:
                            note_uuid = ""
                        table_entry: DocumentTable = {
                            "columns": columns,
                            "rows": rows,
                            "metadata": metadata,
                            "note_uuid": note_uuid
                        }
                        tables.append(table_entry)
                    rows = []
                    columns = []
                    metadata = {}
                    continue
            elif table_state == 4:
                if line.startswith("```"):
                    table_state = 0
                    rows = []
                    columns = []
                    metadata = {}
                    continue
            elif table_state == 5:
                if line.startswith("$$"):  # end of latex block
                    table_state = 0
                    rows = []
                    columns = []
                    metadata = {}
                    continue
            elif table_state == 6:
                if line.startswith("```"):
                    table_state = 0
                    rows = []
                    columns = []
                    metadata = {}
                else:
                    meta = line.split("=", 1)
                    if len(meta) == 2:
                        key = meta[0].strip()
                        val = meta[1].strip()
                        if val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]
                        metadata[key] = val
                continue
        return tables
