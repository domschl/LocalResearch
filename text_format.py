import logging
import os
from rich import print as rprint

class TextFormat:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("TextFormat")
        self.sep:str = "┇"

    def shorten(self, text:str, length:int, left_align:bool|None=None, ellipsis:str='⋯') -> str:
        if len(text) == length:
            return text
        if len(text) == 0:
            return ' ' * length
        elif len(text) < length:
            if left_align is None:
                return text + ' '*(length - len(text))
            elif left_align is True:
                return text + ' '*(length - len(text))
            else:
                return ' '*(length - len(text)) + text
        else:
            if length == 0:
                return ""
            if length == 1:
                return ellipsis
            if left_align is None:
                l = length // 3
                r = length - l - 1
                return text[:l] + ellipsis + text[-r:]
            elif left_align is True:
                w = length - 1
                return text[:w] + ellipsis
            else:
                w = length - 1
                return text[-w:] + ellipsis

    def valid_split(self, line:str, length:int) -> tuple[str, str]:
        if len(line) <= length:
            return line, ""
        ind = length
        mx = 10
        if mx > len(line) // 2:
            mx = len(line) // 2
        for mxi in range(mx):
            if line[ind - mxi] == ' ':
                return line[:ind - mxi], line[ind - mxi + 1:]
            if line[ind - mxi] == '-':
                return line[:ind - mxi + 1], line[ind - mxi + 1:]
        return line[:length], line[length:]
        
    
    def multi_liner(self, text:str, length:int) -> list[str]:
        lines: list[str]= []
        while len(text) > 0:
            if len(text) <= length:
                lines.append(text + ' ' * (length - len(text)))
                text = ""
            else:
                line, text = self.valid_split(text, length)
                lines.append(line + ' ' * (length - len(line)))
        if lines == []:
            lines.append(' ' * length)
        return lines

    def key_highlight(self, text:str, keys:list[str]|None) -> str:
        if keys is None or keys == []:
            return text
        for key in keys:
            text = text.replace(key, "[bold red]"+key+"[/bold red]")
        return text
                
    def print_table(self, header:list[str], rows:list[list[str]], alignments:list[bool|None]|None=None, multi_line:bool=False, max_width:int=0, keywords:list[str]|None=None) -> bool:
        if max_width == 0:
            width:int = os.get_terminal_size()[0]
        else:
            width = max_width
        header_cols = len(header)
        if alignments is not None and len(alignments) != len(header):
            self.log.error(f"If alignments are not None, dim must be equal to header dim")
            return False
        for row in rows:
            if len(row) > header_cols:
                self.log.error(f"Faulty row: {row}, too many columns!")
                return False
            elif len(row) < header_cols:
                self.log.error(f"Faulty row {row}, too few columns!")
                return False
        col_width: list[int] = []
        for column in header:
            col_width.append(len(column))
        for row in rows:
            for index, col in enumerate(row):
                if col_width[index] < len(col):
                    col_width[index] = len(col)
                    
        full_width = 1
        for cw in col_width:
            full_width += cw + 3

        if full_width > width:
            med = (width - 2) // len(header) - 3
            if med < 1:
                med = 1
            old_width = -1
            while full_width > width and old_width != full_width:
                old_width = full_width
                for ind, cw in enumerate(col_width):
                    if cw > med:
                        col_width[ind] -= 1
                        full_width -= 1
                        if full_width <= width:
                            break

        bg = 230
        line = f"[black on color({bg})]{self.sep}"
        for index, col in enumerate(header):
            al = None
            entry = self.shorten(col, col_width[index], al)
            line += " " + entry + " " + self.sep
        line += "[/]"
        rprint(line)
        bg = 231
        for row in rows:
            line = f"[black on color({bg})]"
            if multi_line is False:
                line += self.sep
                for index, col in enumerate(row):
                    if alignments is not None:
                        al = alignments[index]
                    else:
                        al = None
                    entry = self.shorten(col, col_width[index], al)
                    line += " " + entry + " " + self.sep
                line += "[/]"
                rprint(line)
            else:
                sub_lines:list[list[str]] = []
                max_sub_lines = 0
                for index, col in enumerate(row):
                    sls: list[str] = self.multi_liner(col, col_width[index])
                    if len(sls) > max_sub_lines:
                        max_sub_lines = len(sls)
                    sub_lines.append(sls)
                for sl in range(max_sub_lines):
                    line = f"[black on color({bg})]"
                    line += self.sep
                    for index in range(len(sub_lines)):
                        line += ' '
                        if len(sub_lines[index]) > sl:
                            sline = sub_lines[index][sl].replace('[', '\\[')  # RICH ESCAPE MESS
                            line += self.key_highlight(sline, keywords)
                        else:
                            line += ' ' * col_width[index]
                        line += ' ' + self.sep
                    line += "[/]"
                    rprint(line)
                
        return True
