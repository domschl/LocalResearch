import logging
import threading
import queue
import sys
import os
import re
import termios
from typing import override

from .led_repl_io import ReplIO, InputEvent

class TextReplIO(ReplIO):
    def __init__(self, que:queue.Queue[InputEvent]):
        self.log: logging.Logger = logging.getLogger("TextReplIO")
        self.input_queue: queue.Queue[InputEvent] = que
        self.cur_x_offset: int
        self.cur_y_offset: int
        self.cols: int
        self.rows: int
        self.cols, self.rows = self.canvas_update_size()
        self.fg_color: list[int] = [0xff, 0xff, 0xff, 0xff]
        self.bg_color: list[int] = [0, 0, 0, 0xff]

        self.input_loop_active:bool = False
        self.key_reader_active:bool = False
        self.cur_x_offset, self.cur_y_offset = self.get_cursor_pos()
        self.key_queue:queue.Queue[bytearray] = queue.Queue()
        self.key_reader_active = True
        self.key_thread: threading.Thread = threading.Thread(target=self.key_reader, daemon=True)
        self.key_thread.start()
        self.input_loop_active = True
        self.input_thread: threading.Thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

    @override
    def exit(self):
        self.key_reader_active = False
        self.input_loop_active = False

    @override
    def event_loop_tick(self):
        _, _ = self.canvas_update_size()
        
    def get_ansi_char(self) -> str | None:
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        term = termios.tcgetattr(fd)
        ch: str | None = None
        try:
            term[3] &= ~(termios.ICANON | termios.ECHO | termios.IGNBRK | termios.BRKINT)
            termios.tcsetattr(fd, termios.TCSAFLUSH, term)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        return ch

    def key_reader(self):
        while self.key_reader_active is True:
            inp = self.get_ansi_char()
            if isinstance(inp, str):
                bytes = bytearray(inp, encoding='UTF-8')
                self.key_queue.put_nowait(bytes)

    @override
    def color_set(self, fg:list[int], bg:list[int] | None):
        self.fg_color = fg
        if bg is not None:
            self.bg_color = bg

    def input_loop(self):
        esc_state: bool = False
        esc_code = ""
        term_char:str = ""
        tinp: InputEvent = InputEvent("", "")
        while self.input_loop_active is True:
            try:
                inp = self.key_queue.get(timeout=0.01)
            except queue.Empty:
                if esc_state is True:
                    tinp = InputEvent("esc", "")
                    self.input_queue.put_nowait(tinp)
                esc_state = False
                esc_code = ""
                term_char = ""
                continue
            self.key_queue.task_done()
            if len(inp) > 0:
                if esc_state is True:
                    esc_code += chr(inp[0])
                    if len(esc_code) == 2:
                        if esc_code == "[A":
                            tinp = InputEvent("up", "")
                        elif esc_code == "[B":
                            tinp = InputEvent("down", "")
                        elif esc_code == "[C":
                            tinp = InputEvent("right", "")
                        elif esc_code == "[D":
                            tinp = InputEvent("left", "")
                        elif esc_code == "[F":
                            tinp = InputEvent("end", "")
                        elif esc_code == "[H":
                            tinp = InputEvent("home", "")
                        elif esc_code == "OP":
                            tinp = InputEvent("F1", "")
                        elif esc_code == "OQ":
                            tinp = InputEvent("F2", "")
                        elif esc_code == "OR":
                            tinp = InputEvent("F3", "")
                        elif esc_code == "OS":
                            tinp = InputEvent("F4", "")
                        elif esc_code[0] == "[" and esc_code[1] in "123456":
                            term_char = '~'
                        else:
                            tinp = InputEvent("err", "ESC-"+esc_code)
                        if tinp.cmd != "":
                            self.input_queue.put_nowait(tinp)
                            tinp = InputEvent("", "")
                            esc_code = ""
                            esc_state = False
                    if term_char != '' and esc_code.endswith(term_char):
                        if esc_code == "[5~":  # PgUp
                            tinp = InputEvent("PgUp", "")
                        elif esc_code == "[6~":
                            tinp = InputEvent("PgDown", "")
                        elif esc_code == "[5;2~":
                            tinp = InputEvent("Start", "")
                        elif esc_code == "[6;2~":
                            tinp = InputEvent("End", "")
                        else:
                            tinp = InputEvent("EscSeq", esc_code)
                        self.input_queue.put_nowait(tinp)
                        tinp = InputEvent("", "")
                        esc_code = ""
                        esc_state = False
                        term_char = ""
                else:
                    if inp == bytearray([0x7f]):  # BSP
                        tinp = InputEvent("bsp", "")
                    elif inp == bytearray([27]):  # ESC
                        esc_state = True
                        continue
                    elif inp == bytearray([0x05]):  # Ctrl-E
                        tinp = InputEvent("end", "")
                    elif inp == bytearray([0x0a]):
                        tinp = InputEvent("nl", "")
                    elif inp == bytearray([0x01]):  # ^A
                        tinp = InputEvent("home", "")
                    elif inp == bytearray([0x06]):  # ^F
                        tinp = InputEvent("right", "")
                    elif inp == bytearray([0x02]):  # ^B
                        tinp = InputEvent("left", "")
                    elif inp == bytearray([14]):  # ^N
                        tinp = InputEvent("down", "")
                    elif inp == bytearray([16]):  # ^P
                        tinp = InputEvent("up", "")
                    elif inp == bytearray([24]):  # ^X
                        tinp = InputEvent("exit", "")
                    else:
                        tinp = InputEvent("char", inp.decode('utf-8'))
                    # print(f"<Q:{tinp}>", end="")
                    # _ = sys.stdout.flush()
                    self.input_queue.put_nowait(tinp)
                    
        
    def get_cursor_pos(self) -> tuple[int, int]:
        if self.input_loop_active is False:
            _ = sys.stdout.write("\x1b[6n")
            _ = sys.stdout.flush()
            res = ""
            while res.endswith('R') is False:
                t = self.get_ansi_char()
                if t is not None:
                    res += t
            mt = re.match(r".*\[(?P<y>\d*);(?P<x>\d*)R", res)
            if mt is not None:
                x = int(mt.group("x"))
                y = int(mt.group("y"))
                return (x, y)
            else:
                return (-1, -1)
        else:
            return (-1, -1)
        
    @override
    def cursor_start_offset_get(self) -> tuple[int, int]:
        return self.cur_x_offset, self.cur_y_offset

    @override
    def canvas_update_size(self) -> tuple[int, int]:
        self.cols, self.rows = os.get_terminal_size()
        return (self.cols, self.rows)
    
    @override
    def canvas_init(self, size_x:int =0, size_y:int=0) -> bool:
        if size_x == 0 or size_x > self.cols:
            size_x = self.cols
        if size_y == 0 or size_y > self.rows:
            size_y = self.rows

        if self.cur_y_offset + size_y >= self.rows:
            for _ in range(size_y - 1):
                print()
            self.cur_y_offset -= size_y + self.cur_y_offset - self.rows
        return True

    @override
    def canvas_print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        cols, rows = os.get_terminal_size()
        print(f"\033[38;2;{self.fg_color[0]};{self.fg_color[1]};{self.fg_color[2]}m")  # Set foreground color as RGB.
        print(f"\033[48;2;{self.bg_color[0]};{self.bg_color[1]};{self.bg_color[2]}m")  # Set background color as RGB.
        if scroll is False:
            if x>=cols or y>=rows:
                if flush is True:
                    _ = sys.stdout.flush()
                return
        nmsg = ""
        for c in msg:
            if ord(c)<32:
                continue
            else:
                nmsg +=c
        if x+len(nmsg) > cols:
            nmsg = nmsg[:cols-x]
        print(f"\033[{y};{x}H{nmsg}", end="")
        if flush is True:
            _ = sys.stdout.flush()

    @override
    def canvas_render_start(self):
        return

    @override
    def canvas_render_show(self):
        _ = sys.stdout.flush()
        return
    
    @override
    def cursor_hide(self):
        print('\033[?25l', end="")
        _ = sys.stdout.flush()

    @override
    def cursor_show(self):
        print('\033[?25h', end="")
