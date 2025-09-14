import logging
import os
from dataclasses import dataclass
#from typing import TypedDict
# from rich import print as rprint

@dataclass

class Color:
    def __init__(self, h:str|None=None, r:int|None=None, g:int|None=None, b:int|None=None, a:int|None=None):
        self.r: int = 0
        self.g: int = 0
        self.b: int = 0
        self.a: int = 0xff
        if h is not None and h.startswith('#'):
            if len(h) == 7 or len(h) == 9:
                self.r=int(h[1:3], 16)
                self.g=int(h[3:5], 16)
                self.b=int(h[5:7], 16)
            if len(h) == 9:
                self.a = int(h[7:9], 16)
            else:
                self.a=0xff
        else:
            if a is not None:
                self.a = a
            if r is not None:
                self.r = r
            if g is not None:
                self.g = g
            if b is not None:
                self.b = b

class TextFormat:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("TextFormat")
        self.sep:str = "┇"
        self.paper_color: Color = Color("#fbfaf2")
        self.theme: dict[str, tuple[Color,Color]] = {
            'header': (Color("#e0d67f"), Color("#000000")),
            'text': (self.paper_color, Color("#000000")),
            'keyword': (Color("#ffffa0"), Color("#0c4dff")),
            }

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

    @staticmethod
    def fg(col:Color):
        print(f"\033[38;2;{col.r};{col.g};{col.b}m", end="")  # Set foreground color as RGB.

    @staticmethod
    def bg(col:Color):
        print(f"\033[48;2;{col.r};{col.g};{col.b}m", end="")  # Set background color as RGB.

    @staticmethod
    def defc():
        print("\033[m", end="", flush=True)

    def tkc(self, name:str):
        if name not in self.theme:
            self.log.error(f"Unknown theme component {name} referenced.")
            return
        bg, fg = self.theme[name]
        TextFormat.bg(bg)
        TextFormat.fg(fg)        
    
    def valid_split(self, line:str, length:int) -> tuple[str, str]:
        if len(line) <= length:
            return line, ""
        ind = length
        mx = 18
        if mx > len(line) // 2:
            mx = len(line) // 2
        for mxi in range(mx):
            if line[ind - mxi] == ' ':
                return line[:ind - mxi], line[ind - mxi + 1:]
            if line[ind - mxi] == '-' and mxi != 0:
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

    def key_highlight(self, text:str, keys:list[str]|None):
        if keys is None or keys == []:
            self.tkc("text")
            print(text, end="")
            return
        while len(text) > 0:
            min_ind = -1
            key_len = 0
            for key in keys:
                ind = text.lower().find(key.lower())
                if ind > -1 and ind > min_ind:
                    min_ind = ind
                    key_len = len(key)
            if min_ind == -1:
                self.tkc("text")
                print(text, end="")
                return
            else:
                self.tkc("text")
                print(text[:min_ind], end="")
                self.tkc("keyword")
                print(text[min_ind:min_ind+key_len], end="")
                self.tkc("text")
                text = text[min_ind+key_len:]
                
    def filter_keys(self, keywords: list[str]|None) -> list[str]|None:
        if keywords is None or keywords == []:
            return keywords
        trivials = ["and", "or", "to", "of", "the", "a", "in", "this", "these", "be", "it", "for"]
        f_keys:list[str] = []
        for key in keywords:
            if key not in trivials:
                f_keys.append(key)
        return f_keys
    
    def print_table(self, header:list[str], rows:list[list[str|tuple[str,str]]], alignments:list[bool|None]|None=None, multi_line:bool=False, max_width:int=0, keywords:list[str]|None=None) -> bool:
        if max_width == 0:
            width:int = os.get_terminal_size()[0]
        else:
            width = max_width
        if width < 30:
            self.log.error(f"Window width insufficient: {width}")
            return False
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

        self.tkc("header")
        print(self.sep, end="")
        for index, col in enumerate(header):
            al = None
            entry = self.shorten(col, col_width[index], al)
            print(" " + entry + " " + self.sep, end="")
        print()
        self.tkc("text")
        for row in rows:
            if multi_line is False:
                print(self.sep, end="")
                for index, col in enumerate(row):
                    if alignments is not None:
                        al = alignments[index]
                    else:
                        al = None
                    entry = self.shorten(col, col_width[index], al)
                    print(" " + entry + " " + self.sep, end="")
                print()
            else:
                sub_lines:list[list[str]] = []
                max_sub_lines = 0
                for index, col in enumerate(row):
                    sls: list[str] = self.multi_liner(col, col_width[index])
                    if len(sls) > max_sub_lines:
                        max_sub_lines = len(sls)
                    sub_lines.append(sls)
                for sl in range(max_sub_lines):
                    print(self.sep, end="")
                    for index in range(len(sub_lines)):
                        print(' ', end="")
                        if len(sub_lines[index]) > sl:
                            sline = sub_lines[index][sl]
                            self.key_highlight(sline, self.filter_keys(keywords))
                        else:
                            print(' ' * col_width[index], end="")
                        print(' ' + self.sep, end="")
                    print()
        TextFormat.defc()
        return True
