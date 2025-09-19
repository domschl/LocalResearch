import logging
import os
from dataclasses import dataclass
from copy import copy
from enum import Enum
#from typing import TypedDict

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


class AttrString:
    def __init__(self, string:str, bg:Color|list[Color], fg:Color|list[Color]):
        self.string:str = string
        if isinstance(bg, list):
            if len(self.string) != len(bg):
                raise ValueError("String length must be equal to bg color list")
            self.bg = copy(bg)
        else:
            self.bg:list[Color] = []
            for _ in range(len(self.string)):
                self.bg.append(copy(bg))
        if isinstance(fg, list):
            if len(self.string) != len(fg):
                raise ValueError("String length must be equal to fg color list")
            self.fg = copy(fg)
        else:
            self.fg:list[Color] = []
            for _ in range(len(self.string)):
                self.fg.append(copy(fg))
        
    def __len__(self):
        return len(self.string)

    def __add__(self, other:'AttrString'):  ## Type work-around
        return AttrString(self.string + other.string, self.bg+other.bg, self.fg+other.fg)

    def substring(self, start:int, end:int=-1) -> 'AttrString':  ## Type work-around
        if end == -1:
            end = len(self.string)
        return AttrString(self.string[start:end], self.bg[start:end], self.fg[start:end])
            

class TextFormat:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("TextFormat")
        self.sep:str = "┇"
        self.paper_color: Color = Color("#fbfaf9")
        self.theme: dict[str, tuple[Color,Color]] = {
            'header': (Color("#e0d67f"), Color("#000000")),
            'text': (self.paper_color, Color("#000000")),
            'keyword': (Color("#ffffa0"), Color("#ff0000")),
            'selected': (Color("#e0e0ff"), Color("#000000")),
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

    def theme_col(self, theme_name:str):
        if theme_name not in self.theme:
            self.log.error(f"Unknown theme color {theme_name} referenced")
            return
        TextFormat.bg(self.theme[theme_name][0])
        TextFormat.fg(self.theme[theme_name][1])
                      
    @staticmethod
    def print_attr(text:AttrString):
        # last_bg:Color|None = None
        # last_fg:Color|None = None
        for ind in range(len(text.string)):
            c = text.string[ind]
            # if last_bg != text.bg[ind]:
            last_bg = text.bg[ind]
            TextFormat.bg(last_bg)
            # if last_fg != text.fg[ind]:
            last_fg = text.fg[ind]
            TextFormat.fg(last_fg)
            print(c, end="")
        TextFormat.defc()
            
    def tkc(self, name:str):
        if name not in self.theme:
            self.log.error(f"Unknown theme component {name} referenced.")
            return
        bg, fg = self.theme[name]
        TextFormat.bg(bg)
        TextFormat.fg(fg)        
    
    def valid_split(self, line:AttrString, length:int) -> tuple[AttrString, AttrString]:
        if len(line) <= length:
            return line, AttrString("", *self.theme['text'])
        ind = length
        mx = 18
        if mx > len(line) // 2:
            mx = len(line) // 2
        for mxi in range(mx):
            if line.string[ind - mxi] == ' ':
                return line.substring(0,ind - mxi), line.substring(ind - mxi + 1)
            if line.string[ind - mxi] == '-' and mxi != 0:
                return line.substring(0,ind - mxi + 1), line.substring(ind - mxi + 1)
        return line.substring(0,length), line.substring(length)
            
    def markup(self, text:AttrString, keywords:list[str], significance:list[float]|None):
        for ind in range(len(text.string)):
            for keyword in keywords:
                if text.string[ind:ind+len(keyword)].lower() == keyword.lower():
                    for mark in range(ind, ind+len(keyword)):
                        text.bg[mark] = copy(self.theme['keyword'][0])
                        text.fg[mark] = copy(self.theme['keyword'][1])
            if significance is not None and significance[ind] != 0:
                cur_bg = text.bg[ind]
                yellow = int(cur_bg.b - significance[ind] * 255)
                if yellow > 255:
                    yellow = 255
                if yellow < 0:
                    yellow = 0
                text.bg[ind].b = yellow
        return text
                
    def multi_liner(self, text:AttrString, length:int) -> list[AttrString]:
        lines: list[AttrString]= []
        while len(text) > 0:
            if len(text) <= length:
                lines.append(text + AttrString(' ' * (length - len(text)), *self.theme['text']))
                text = AttrString("", *self.theme['text'])
            else:
                line, text = self.valid_split(text, length)
                lines.append(line + AttrString(' ' * (length - len(line)), *self.theme['text']))
        if lines == []:
            lines.append(AttrString(' ' * length, *self.theme['text']))
        return lines

    def filter_keys(self, keywords: list[str]) -> list[str]:
        trivials = ["and", "or", "to", "of", "the", "a", "in", "what", "which", "this", "these",
                            "be", "it", "for", "on", "he", "she", "is", "was"]
        f_keys:list[str] = []
        for key in keywords:
            if key.lower() not in trivials:
                f_keys.append(key.lower())
        return f_keys
    
    def print_table(self, header:list[str], rows:list[list[str]], alignments:list[bool|None]|None=None,
                    multi_line:bool=False, max_width:int=0, keywords:list[str]|None=None, selected:list[int]|int|None=None,
                    significance:list[list[list[float]|None]]|None=None) -> bool:
        if max_width == 0:
            width:int = os.get_terminal_size()[0]
        else:
            width = max_width
        if keywords is None:
            keywords = []
        if width < 30:
            self.log.error(f"Window width insufficient: {width}")
            return False
        header_cols = len(header)
        if alignments is not None and len(alignments) != len(header):
            self.log.error(f"If alignments are not None, dim must be equal to header dim")
            return False
        if significance is not None:
            if len(significance) != len(rows):
                self.log.error(f"If significance is not None, dim must be equal to dim rows")
                return False
        for index, row in enumerate(rows):
            if len(row) > header_cols:
                self.log.error(f"Faulty row: {row}, too many columns!")
                return False
            elif len(row) < header_cols:
                self.log.error(f"Faulty row {row}, too few columns!")
                return False
            if significance is not None:
                if len(significance[index]) != header_cols:
                    self.log.error(f"Significance row-index {index} has wrong dim!")
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
        self.defc()
        print()
        self.tkc("text")
        for line_index, row in enumerate(rows):
            if multi_line is False:
                if selected is not None and (selected == line_index or (isinstance(selected, list) and line_index in selected)):
                    self.theme_col('selected')
                else:
                    self.theme_col('text')
                print(self.sep, end="")
                for index, col in enumerate(row):
                    if alignments is not None:
                        al = alignments[index]
                    else:
                        al = None
                    entry = self.shorten(col, col_width[index], al)
                    print(" " + entry + " " + self.sep, end="")
                self.defc()
                print()
            else:
                sub_lines:list[list[AttrString]] = []
                max_sub_lines = 0
                for index, col in enumerate(row):
                    acol = AttrString(col, *self.theme['text'])
                    if significance is not None and significance[line_index][index] is not None:
                        acol = self.markup(acol, self.filter_keys(keywords), significance[line_index][index])
                    else:
                        acol = self.markup(acol, self.filter_keys(keywords), None)
                    sls: list[AttrString] = self.multi_liner(acol, col_width[index])
                    if len(sls) > max_sub_lines:
                        max_sub_lines = len(sls)
                    sub_lines.append(sls)
                for sl in range(max_sub_lines):
                    print(self.sep, end="")
                    for index in range(len(sub_lines)):
                        print(' ', end="")
                        if len(sub_lines[index]) > sl:
                            sline = sub_lines[index][sl]
                            TextFormat.print_attr(sline)
                        else:
                            self.theme_col('text')
                            print(' ' * col_width[index], end="")
                        print(' ' + self.sep, end="")
                    print()
        TextFormat.defc()
        return True


class TokType(Enum):
    arg = 1
    string = 2
    op = 3
    key = 4
    val = 5

class TextParse:
        
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("TextParser")

    def parse_full(self, text:str) -> tuple[list[str]|None, list[TokType]|None]:
        result: list[str] = []
        format: list[TokType] = []
        tok:str = ""
        in_str:bool = False
        esc:bool = False
        key_val = False
        str_chr = ""
        for ind, c in enumerate(text):
            if esc is True:
                esc = False
                if c == 'n':
                    tok += '\n'
                else:
                    tok += c
                if ind == len(text) - 1:
                    result.append(tok)
                    if key_val is True:
                        format.append(TokType['val'])
                        key_val = False
                    else:
                        format.append(TokType['arg'])
                    tok = ""
                continue
            if c == '\\':
                esc = True
                continue                
            if in_str is True:
                if c == str_chr:
                    str_chr = ""
                    in_str = False
                    if len(tok) > 0:
                        result.append(tok)
                        if key_val is True:
                            format.append(TokType['val'])
                            key_val = False
                        else:
                            format.append(TokType['string'])
                        tok = ""
                    continue
                else:
                    tok += c
                    continue
            if c == '"' or c == "'":
                in_str = True
                str_chr = c
                continue
            if c == '=':
                if len(tok) > 0:
                    result.append(tok)
                    format.append(TokType['arg'])
                    tok = ""                    
                if len(format) == 0 or format[-1] != TokType['arg']:
                    self.log.error("Invalid '=', not following valid token")
                    return None, None
                format[-1] = TokType['key']
                key_val = True
                result.append('=')
                format.append(TokType['op'])
                continue                
            if c == ' ':
                if len(tok) > 0:
                    result.append(tok)
                    if key_val is True:
                        format.append(TokType['val'])
                        key_val = False
                    else:
                        format.append(TokType['arg'])
                    tok = ""
                    continue
            if ind == len(text) - 1:
                tok += c
                result.append(tok)
                if key_val is True:
                    format.append(TokType['val'])
                    key_val = False
                else:
                    format.append(TokType['arg'])
                tok = ""
                continue
            tok += c
        if key_val is True:
            self.log.error("Unterminated assignment")
            return None, None
        if esc is True:
            self.log.error("Unterminated ESC sequence!")
            return None, None
        if in_str is True:
            self.log.error(f"Unterminated string, missing terminator {str_chr}")
            return None, None
        if len(tok) > 0:
            self.log.error(f"Parser error, unprocessed: {tok}")
            return None, None
        return result, format

    def parse(self, text:str) -> list[str]|None:
        result, _ = self.parse_full(text)
        return result
        
    def parse_keys(self, text:str) -> tuple[list[str], dict[str,str]]:
        result, format = self.parse_full(text)
        if result is None or format is None:
            return ([], {})
        list_args: list[str] = []
        key_args: dict[str,str] = {}

        prev_tok:str = ""
        for ind, tok in enumerate(result):
            if format[ind] == TokType['string']:
                if len(prev_tok)>0:
                    list_args.append(prev_tok)
                list_args.append(tok)
                prev_tok = ""
                continue
            # if '=' in tok:
                
        return list_args, key_args
            
