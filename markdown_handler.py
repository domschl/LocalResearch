import logging
import os
import yaml
from typing import Any

from research_defs import MetadataEntry, ResearchMetadata
from research_tools import ResearchTools, DocumentTable

from orgmode_handler import OrgmodeTools

class MarkdownTools:
    def __init__(self, notes_folder:str):
        self.log: logging.Logger = logging.getLogger("MarkdownTools")
        self.notes_folder:str|None = os.path.expanduser(notes_folder)
        self.rt:ResearchTools = ResearchTools()
        self.rm:ResearchMetadata = ResearchMetadata()

    def split_header_content(self, text:str) -> tuple[str, str]:
        separator1 = "---\n"
        separator2 = "\n---\n"
        if text.startswith(separator1):
            d1 = 0
        else:
            d1 = text.find(separator2)
            if d1 > 10 or d1 == -1:
                return ("", text)
            d1 += 1
         
        d2 = text[d1+len(separator1):].find(separator2)
        if d2 == -1:
            return ("", text)
        d2 += d1+len(separator1)
        header = text[d1+len(separator1):d2+1]
        content = text[d2+len(separator2):]
        return (header, content)
         
    def parse_markdown(self, filepath:str, hash:str, descriptor:str, md_text:str) -> tuple[MetadataEntry, str, bool, bool]:
        frontmatter, content = self.split_header_content(md_text)
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
            
        filtered_metadata: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, list) and len(v) == 0:
                continue
            if isinstance(v, str) and v == "":
                continue
            filtered_metadata[k] = v
            
        header = yaml.dump(filtered_metadata, default_flow_style=False, indent=2)
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
                        if "context" not in metadata and subfolders is not None:
                            metadata["context"] = f"{subfolders}"
                            for column in columns:
                                metadata["context"] += f"_{column}"
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

    def convert_markdown_to_org(self, md_text:str) -> str|None:
        import re
        lines = md_text.split('\n')
        out_lines: list[str] = []
        in_code_block = False
        in_math_block = False
        in_quote_block = False
        
        # convert the following markdown syntactic elements:
        # Headers [#]+ (underlining headers not supported!), lists and enumerations, quoting with '>', tables, codeblocks, bold and italic, links, images, code blocks, 
        # and latex math blocks with '$' and '$$' delimiters. 
        for line in lines:
            # code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    out_lines.append("#+END_SRC")
                    in_code_block = False
                else:
                    lang = line.strip()[3:].strip()
                    if lang:
                        out_lines.append(f"#+BEGIN_SRC {lang}")
                    else:
                        out_lines.append("#+BEGIN_SRC")
                    in_code_block = True
                continue
                
            if in_code_block:
                out_lines.append(line)
                continue
                
            # block math $$
            if line.strip() == "$$":
                if in_math_block:
                    out_lines.append("\\]")
                    in_math_block = False
                else:
                    out_lines.append("\\[")
                    in_math_block = True
                continue
                
            if in_math_block:
                out_lines.append(line)
                continue
                
            # blockquotes
            if line.lstrip().startswith(">"):
                if not in_quote_block:
                    out_lines.append("#+BEGIN_QUOTE")
                    in_quote_block = True
                line = line.lstrip()[1:].lstrip()
            else:
                if in_quote_block:
                    out_lines.append("#+END_QUOTE")
                    in_quote_block = False
                    
            # headers
            header_match = re.match(r'^(#+)\s+(.*)', line)
            if header_match:
                level = len(header_match.group(1))
                line = '*' * level + ' ' + header_match.group(2)
                
            # lists
            list_match = re.match(r'^(\s*)\* +(.*)', line)
            if list_match:
                line = f"{list_match.group(1)}- {list_match.group(2)}"
                
            # table separator
            if line.lstrip().startswith('|') and re.match(r'^[\s\|:\-]+$', line) and '-' in line:
                line = line.replace(':', '-')
                line = line.replace('|', '+')
                if len(line.strip()) > 1:
                    line = '|' + line.strip()[1:-1] + '|'
                    
            # images: ![text](url) -> [[url]]
            line = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'[[\2]]', line)
            
            # links: [text](url) -> [[url][text]]
            line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[[\2][\1]]', line)
            
            # bold and italic
            line = re.sub(r'\*\*(.+?)\*\*', r'*\1*', line)
            line = re.sub(r'__(.+?)__', r'*\1*', line)
            line = re.sub(r'(?<![\w\*])\*(?!\s)(.+?)(?<!\s)\*(?![\w\*])', r'/\1/', line)
            line = re.sub(r'\b_(.+?)_\b', r'/\1/', line)
            
            # inline code
            line = re.sub(r'`([^`]+)`', r'~\1~', line)
            
            # inline math
            line = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', r'\(\1\)', line)
            
            # block math inline form $$math$$
            line = re.sub(r'\$\$(.+?)\$\$', r'\[\1\]', line)

            out_lines.append(line)
            
        if in_quote_block:
            out_lines.append("#+END_QUOTE")
            
        org_text = '\n'.join(out_lines)
        return org_text


    def export_as_orgmode(self, filepath:str, hash:str, descriptor:str, md_text:str, org_directory:str, prefix:str) -> bool:
        if not os.path.exists(org_directory):
            self.log.warning(f"Org directory {org_directory} does not exist")
            return False
        ot = OrgmodeTools(org_directory)
        metadata, content, changed, mandatory_changed = self.parse_markdown(filepath, hash, descriptor, md_text)
        if changed or mandatory_changed:
            self.log.warning(f"Ignoring export of {filepath} as it has changed or has mandatory fields that are not set")
            return False
        
        # Create the directory structure
        source_folder = os.path.dirname(filepath)
        source_filename = os.path.basename(filepath)
        if source_filename.endswith(".md"):
            dest_filename = source_filename[:-3] + ".org"
        else:
            self.log.warning(f"Filepath {filepath} does not end with .md")
            return False
        if self.notes_folder is not None and source_folder.startswith(self.notes_folder):
            subfolders = source_folder[len(self.notes_folder) + 1 :]
        else:
            self.log.warning(f"Filepath {filepath} does not start with notes folder {self.notes_folder}")
            return False
        dest_folder = os.path.join(org_directory, subfolders)
        dest_filepath:str = os.path.join(dest_folder, dest_filename)
        if os.path.exists(dest_filepath):
            self.log.warning(f"Filepath {dest_filepath} already exists")
            return False

        org_text = self.convert_markdown_to_org(content)
        if org_text is None:
            self.log.warning(f"Failed to convert markdown to org for {filepath}")
            return False

        org_doc = ot.assemble_orgmode(metadata, prefix, org_text)

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        with open(dest_filepath, "w") as f:
            _ = f.write(org_doc)
        return True
    
    def export_as_markdown(self, filepath:str, hash:str, descriptor:str, org_text:str, org_directory:str) -> bool:
        if self.notes_folder is None:
            self.log.warning("Notes folder is not defined")
            return False
            
        ot = OrgmodeTools(org_directory)
        metadata, prefix, content, changed, mandatory_changed = ot.parse_orgmode(filepath, hash, descriptor, org_text)
        if changed or mandatory_changed:
            self.log.warning(f"Ignoring export of {filepath} as it has changed or has mandatory fields that are not set")
            return False
            
        source_folder = os.path.dirname(filepath)
        source_filename = os.path.basename(filepath)
        if source_filename.endswith(".org"):
            dest_filename = source_filename[:-4] + ".md"
        else:
            self.log.warning(f"Filepath {filepath} does not end with .org")
            return False
            
        if source_folder.startswith(org_directory):
            subfolders = source_folder[len(org_directory) + 1 :]
        else:
            self.log.warning(f"Filepath {filepath} does not start with org directory {org_directory}")
            return False
            
        dest_folder = os.path.join(self.notes_folder, subfolders)
        dest_filepath:str = os.path.join(dest_folder, dest_filename)
        if os.path.exists(dest_filepath):
            self.log.warning(f"Filepath {dest_filepath} already exists")
            return False

        md_content = self.convert_org_to_markdown(content)
        if md_content is None:
            self.log.warning(f"Failed to convert org to markdown for {filepath}")
            return False
            
        full_content = prefix + md_content
        md_doc = self.assemble_markdown_ext(metadata, full_content)

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        with open(dest_filepath, "w") as f:
            _ = f.write(md_doc)
        return True

    def convert_org_to_markdown(self, org_text: str) -> str|None:
        import re
        lines = org_text.split('\n')
        out_lines: list[str] = []
        in_code_block = False
        in_quote_block = False
        
        for line in lines:
            # Code blocks
            if re.match(r'^\s*#\+(BEGIN|begin)_(SRC|src)', line):
                parts = line.split()
                lang = parts[1] if len(parts) > 1 else ""
                out_lines.append(f"```{lang}")
                in_code_block = True
                continue
                
            if re.match(r'^\s*#\+(END|end)_(SRC|src)', line):
                out_lines.append("```")
                in_code_block = False
                continue
                
            if in_code_block:
                out_lines.append(line)
                continue
                
            # Quotes
            if re.match(r'^\s*#\+(BEGIN|begin)_(QUOTE|quote)', line):
                in_quote_block = True
                continue
                
            if re.match(r'^\s*#\+(END|end)_(QUOTE|quote)', line):
                in_quote_block = False
                continue
                
            if in_quote_block:
                out_lines.append(f"> {line}")
                continue
                
            # Block math \[ \]
            if line.strip() == "\\[":
                out_lines.append("$$")
                continue
            if line.strip() == "\\]":
                out_lines.append("$$")
                continue
                
            # Headers
            header_match = re.match(r'^(\*+)\s+(.*)', line)
            if header_match:
                level = len(header_match.group(1))
                line = '#' * level + ' ' + header_match.group(2)
                
            # Lists
            list_match = re.match(r'^(\s*)\+ +(.*)', line)
            if list_match:
                line = f"{list_match.group(1)}- {list_match.group(2)}"
                
            # Table separator
            if line.lstrip().startswith('|') and re.match(r'^[\s\|:\-\+]+$', line) and ('-' in line or '+' in line):
                line = line.replace('+', '|')
                
            # Links
            line = re.sub(r'\[\[([^\]]+)\]\[([^\]]+)\]\]', r'[\2](\1)', line)
            line = re.sub(r'\[\[([^\]]+)\]\]', r'<\1>', line)
            
            # Bold
            line = re.sub(r'(?<![\w\*])\*(?!\s)(.+?)(?<!\s)\*(?![\w\*])', r'**\1**', line)
            # Italics
            line = re.sub(r'(?<![\w:\/])\/(?!\s)(.+?)(?<!\s)\/(?![\w\/])', r'*\1*', line)
            # Inline code
            line = re.sub(r'(?<!\~)~(?!\s)(.+?)(?<!\s)~(?!\~)', r'`\1`', line)
            line = re.sub(r'(?<!=)=(?!\s)(.+?)(?<!\s)=(?!=)', r'`\1`', line)
            # Inline math
            line = re.sub(r'\\\((.+?)\\\)', r'$\1$', line)
            
            out_lines.append(line)
            
        md_text = '\n'.join(out_lines)
        return md_text

    def test_roundtrip(self, filepath: str, descriptor: str) -> bool:
        import difflib
        try:
            with open(filepath, "r") as f:
                md_text = f.read()
        except Exception as e:
            self.log.error(f"Failed to read file {filepath}: {e}")
            return False
            
        metadata, md_content, _, _ = self.parse_markdown(filepath, "test_hash", descriptor, md_text)
        
        org_content = self.convert_markdown_to_org(md_content)
        if org_content is None:
            self.log.error("Failed to convert markdown to org")
            return False
            
        # Assemble orgmode doc
        # OrgmodeTools is imported globally in this file, we can instantiate it
        test_org_dir = os.path.expanduser("~/OrgNotes/Scratch")
        ot = OrgmodeTools(test_org_dir) # dummy dir
        org_doc = ot.assemble_orgmode(metadata, "", org_content)
        
        print("--- Markdown input --------------------")
        print(md_text)
        print("--- Orgmode output --------------------")
        print(org_doc)
        
        # Parse orgmode doc
        org_metadata, org_prefix, org_parsed_content, _, _ = ot.parse_orgmode(filepath, "test_hash", descriptor, org_doc)
        
        unsupported_new_fields = [] # 'normalized_filename', 'representations']
        for field in unsupported_new_fields:
            if field in org_metadata:
                del org_metadata[field]
        rt_md_content = self.convert_org_to_markdown(org_parsed_content)
        if rt_md_content is None:
            self.log.error("Failed to convert org to markdown")
            return False

        print("--- Metadata ---")
        print(metadata)
        print("--- Org Metadata ---")
        print(org_metadata)

        original_assembled = self.assemble_markdown_ext(metadata, md_content)
        rt_assembled = self.assemble_markdown_ext(org_metadata, org_prefix + rt_md_content)
        
        print("--- Original assembled --------------------")
        print(original_assembled)
        print("--- Roundtrip assembled -------------------")
        print(rt_assembled)
        print("-------------------------------------------")

        if original_assembled != rt_assembled:
            self.log.error(f"Roundtrip failed for {filepath}! Differences:")
            diff = difflib.unified_diff(
                original_assembled.splitlines(),
                rt_assembled.splitlines(),
                fromfile='original.md',
                tofile='roundtrip.md',
                lineterm=''
            )
            for line in diff:
                print(line)
            return False
            
        self.log.info(f"Roundtrip successful for {filepath}!")
        return True