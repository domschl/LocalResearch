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
                
    def print_table(self, header:list[str], rows:list[list[str]], alignments:list[bool|None]|None=None, multi_line:bool=False, max_width:int=0) -> bool:
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
            # if alignments is not None:
            #     al = alignments[index]
            # else:
            al = None
            entry = self.shorten(col, col_width[index], al)
            line += " " + entry + " " + self.sep
        line += "[/]"
        rprint(line)
        bg = 231
        for row in rows:
            line = f"[black on color({bg})]{self.sep}"
            for index, col in enumerate(row):
                if alignments is not None:
                    al = alignments[index]
                else:
                    al = None
                entry = self.shorten(col, col_width[index], al)
                line += " " + entry + " " + self.sep
            line += "[/]"
            rprint(line)
        return True
