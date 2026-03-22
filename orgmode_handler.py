import logging
import os
import json
from typing import Any, cast

from research_defs import MetadataEntry, ResearchMetadata
from research_tools import ResearchTools


class OrgmodeTools:
    def __init__(self, orgmode_folder:str):
        self.log: logging.Logger = logging.getLogger("OrgmodeTools")
        self.orgmode_folder:str|None = os.path.expanduser(orgmode_folder)
        self.rt:ResearchTools = ResearchTools()
        self.rm:ResearchMetadata = ResearchMetadata()

    def parse_org_prop_value(self, value:str) -> Any:
        vals:list[str] = []
        in_str:bool = False
        in_time:bool = False
        esc:bool = False
        cur:str = ""
        for c in value:
            if esc is True:
                esc = False
                if c == 'n':
                    cur += '\n'
                else:
                    cur += c
                continue
            if c == '\\':
                esc = True
                continue
            if in_str is True:
                if c == '"':
                    in_str = False
                    vals.append(cur)
                    cur = ""
                else:
                    cur += c
                continue
            if in_time is True:
                if c == '>':
                    in_time = False
                    vals.append(cur)
                    cur = ""
                else:
                    cur += c
                continue
            if c == '"':
                in_str = True
                continue                
            if c == '<' and len(cur) == 0:
                in_time = True
                continue
            if c == ' ':
                if len(cur) > 0:
                    vals.append(cur)
                    cur = ""
                continue
            cur += c
            continue
        if esc is True:
            self.log.error(f"Unterminated esc sequence in >{value}<")
            cur = ""
        if in_str is True:
            self.log.error(f"Unterminated string in >{value}<")
            cur = ""
        if in_time is True:
            self.log.error(f"Unterminated timestamp in [{value}]")
        if len(cur) > 1:
            vals.append(cur)
            
        parsed_vals: list[Any] = []
        for v in vals:
            if isinstance(v, str) and ((v.startswith('{') and v.endswith('}')) or (v.startswith('[') and v.endswith(']'))):
                try:
                    parsed_vals.append(json.loads(v))
                except Exception:
                    parsed_vals.append(v)
            else:
                parsed_vals.append(v)
                
        if len(parsed_vals) == 0:
            return ""
        elif len(parsed_vals) == 1:
            return parsed_vals[0]
        else:
            return parsed_vals
            
    def parse_org_properties(self, frontmatter_lines:list[str]) -> dict[str, Any]:  # pyright:ignore[reportExplicitAny]
        metadata:dict[str, Any] = {}  # pyright:ignore[reportExplicitAny]
        timedate_fields = ['pubdate', 'publication_date', 'creation', 'creation_date']
        for line in frontmatter_lines:
            line = line.strip()
            if len(line) < 1:
                continue
            if line[0] != ':':
                continue
            key_end = line[1:].find(':')
            if key_end == -1:
                continue
            key_end += 1
            prop_name = line[1:key_end]
            value_raw = line[key_end+1:]
            if prop_name in timedate_fields:
                if value_raw.startswith('>') and value_raw.endswith('<'):
                    value_raw = value_raw[1:-1]
            metadata[prop_name] = self.parse_org_prop_value(value_raw)
        return metadata
    
    def parse_orgmode(self, filepath:str, hash:str, descriptor:str, md_text:str) -> tuple[MetadataEntry, str, str, bool, bool]:
        state:int = 0
        lines: list[str] = md_text.split("\n")
        frontmatter_lines: list[str] = []
        content_lines: list[str] = []
        prefix_lines: list[str] = []
        content: str = ""
        for line in lines:
            if state == 0:
                if line.strip() == ":PROPERTIES:":
                    if len(content_lines) > 0:
                        for line in content_lines:
                            prefix_lines.append(line)
                        content_lines = []
                    state = 1
                else:
                    content_lines.append(line)                
            elif state == 1:
                if line.strip() == ":END:":
                    state = 2
                else:
                    frontmatter_lines.append(line)
            elif state == 2:
                if len(content_lines) == 0:
                    if line == "":
                        continue
                content_lines.append(line)
        content = "\n".join(content_lines)
        prefix = "\n".join(prefix_lines)
        if len(prefix) > 0:
            prefix += '\n'
        metadata_raw: dict[str, Any] = self.parse_org_properties(frontmatter_lines)  # pyright:ignore[reportExplicitAny]

        print(f"metadata_raw: {metadata_raw}")

        metadata, changed, mandatory_changed = self.rm.normalize_metadata(self.orgmode_folder, filepath, hash, descriptor, metadata_raw)
        return metadata, prefix, content, changed, mandatory_changed
        
    def assemble_orgmode(self, metadata: dict[str, Any]|MetadataEntry, prefix:str, content: str, timedate_fields:list[str]|None=None) -> str:  # pyright: ignore[reportExplicitAny]
        if metadata  == {}:
            return prefix + content
        if timedate_fields is None:
            timedate_fields = ['pubdate', 'publication_date', 'creation', 'creation_date']
        meta_header:list[str] = [":PROPERTIES:"]
        for props in metadata.keys():
            line: str = f":{props}:"
            val: str|list[str] = metadata[props]  # pyright:ignore[reportUnknownVariableType]
            if isinstance(val, list) is True:
                if len(val) == 0:
                    continue
                for vali in val:  # pyright:ignore[reportUnknownVariableType]
                    if isinstance(vali, str) is True:
                        if ' ' in vali:
                            line += ' "' + cast(str, vali) + '"'
                        else:
                            try:
                                line += ' ' + cast(str, vali)
                            except Exception as e:
                                self.log.error(f"Error assembling orgmode for {vali}: {e}")
                    else:
                        try:
                            if hasattr(vali, '__len__') and len(vali) == 0:
                                continue
                        except Exception:
                            pass
                        json_vali = json.dumps(vali)
                        escaped_json = json_vali.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        line += f' "{escaped_json}"'
            else:
                if val == "":
                    continue
                if props in timedate_fields:
                    line += f">{cast(str, val)}<"
                else:
                    if ' ' in val:
                        line += ' "' + cast(str, val) + '"'
                    else:
                        line += ' ' + cast(str, val)            
            meta_header.append(line)
        meta_header.append(":END:")
        org_doc:str = prefix + '\n'.join(meta_header) + "\n" + content
        return org_doc

    def assemble_orgmode_ext(self, metadata:MetadataEntry, prefix:str, content: str) -> str:
        # XXX expand possible optional entries in MetadataEntry (not yet defined)
        return self.assemble_orgmode(metadata, prefix, content)

