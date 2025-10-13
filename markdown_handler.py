import logging
import os
import yaml
from typing import Any

from research_defs import MetadataEntry, ResearchMetadata
from research_tools import ResearchTools


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
    
