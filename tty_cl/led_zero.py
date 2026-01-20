import logging
import queue
from dataclasses import dataclass

from led_repl_io import ReplIO, InputEvent
from led_text_repl_io import TextReplIO
try:
    from .led_sdl2_repl_io import Sdl2ReplIO
    sdl_available:bool = True
except ImportError:
    sdl_available = False
    Sdl2ReplIO = TextReplIO

@dataclass()
class Pad:
    screen_pos_x: int
    screen_pos_y: int
    width: int
    height: int
    left_border: int
    bottom_border: int
    cur_x: int
    cur_y: int
    buffer: list[str]
    buf_x: int
    buf_y: int
    screen: list[str]
    schema: dict[str, list[int]]

class Repl():
    def __init__(self, engine:str="TEXT"):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        self.log: logging.Logger = logging.getLogger("Repl")
        valid_engines = ["TEXT", "SDL2"]
        self.default_schema: dict[str, list[int]] = {
            'fg': [240,240,240,0xff],
            'bg': [15,15,15,0xff],
            'lb': [0,32,120,0xff],
            'bb': [20,20,160,0xff],
            }
        self.schema: dict[str, list[int]] = self.default_schema
        self.editor_esc: bool = False
        self.pads: list[Pad] = []
        if engine not in valid_engines:
            self.log.error(f"Unknown engine {engine}, use one of {valid_engines}")
            exit(1)
        self.engine:str = engine
        self.input_queue:queue.Queue[InputEvent] = queue.Queue()
        if self.engine == "TEXT":
            self.repl: ReplIO = TextReplIO(self.input_queue)
        else:
            self.repl = Sdl2ReplIO(self.input_queue)

    def pad_print_at(self, pad_index:int, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False, border:bool=False):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]
        if border is False:
            self.repl.canvas_print_at(msg, y+pad.screen_pos_y, x+pad.screen_pos_x, flush=flush, scroll=scroll)
        else:
            self.repl.canvas_print_at(msg, y+pad.screen_pos_y, x+pad.screen_pos_x-pad.left_border, flush=flush, scroll=scroll)

    def pad_create(self, buffer:list[str], height: int, width:int = 0, offset_y:int = 0, offset_x:int = 0, left_border:int=0, bottom_border:int=0, schema: dict[str, list[int]] | None = None) -> int:
        if schema is None:
            self.schema = self.default_schema
        cur_x_offset, cur_y_offset = self.repl.cursor_start_offset_get()
        if schema is None:
            schema = self.schema
        pad: Pad = Pad(
            screen_pos_x = cur_x_offset + offset_x + left_border,
            screen_pos_y = cur_y_offset + offset_y,
            width = width-left_border,
            height = height-bottom_border,
            left_border = left_border,
            bottom_border = bottom_border,
            cur_x = 0,
            cur_y = 0,
            schema = schema,
            screen = [' ' * width] * height,
            buffer = buffer,
            buf_x = 0,
            buf_y = 0
            )
        self.pads.append(pad)
        pad_index = len(self.pads)-1
        self.pad_display(pad_index)
        return pad_index
    
    def pad_get(self, padIndex: int) -> Pad | None:
        if padIndex >=0 and padIndex<len(self.pads):
            return self.pads[padIndex]
        else:
            self.log.error(f"Invalid padIndex: {padIndex}")
            return None

    def pad_display(self, pad_index:int, set_cursor:bool = True, update_from_buffer:bool=True):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]

        self.repl.cursor_hide()
        self.repl.canvas_render_start()

        self.repl.color_set(self.schema['fg'], self.schema['bg'])
        if update_from_buffer is True:
            for i in range(pad.height):
                if i+pad.buf_y < len(pad.buffer):
                    pad.screen[i] = buffer[i+pad.buf_y][pad.buf_x:pad.buf_x+pad.width]
                    pad.screen[i] += ' ' * (pad.width - len(pad.screen[i]))
                else:
                    pad.screen[i] = ' ' * pad.width
        for i in range(pad.height):
            self.pad_print_at(pad_index, pad.screen[i], i, 0)
        if pad.left_border > 0:
            self.repl.color_set(self.schema['fg'], self.schema['lb'])
            for i in range(pad.height):
                self.pad_print_at(pad_index, f"  {i+pad.buf_y:3d} ", i, 0, border=True)
        if pad.bottom_border > 0:
            self.repl.color_set(self.schema['fg'], self.schema['bb'])
            for i in range(pad.height, pad.height+pad.bottom_border):
                status_msg = ' ' * pad.left_border + f"Doms editor ({pad.cur_y+pad.buf_y},{pad.cur_x+pad.buf_x})"
                gl = pad.left_border + pad.width
                status_msg = status_msg[:gl]
                status_msg += ' ' * (gl - len(status_msg))
                self.pad_print_at(pad_index, status_msg, i, 0, border=True)
        if set_cursor is True:
            self.pad_print_at(pad_index, "", pad.cur_y, pad.cur_x)
        self.repl.cursor_show()
        self.repl.canvas_render_show()

    def pad_move(self, pad_id:int, dx:int | None = None, dy:int | None = None, x:int | None = None, y: int | None = None) -> bool:
        changed: bool = False
        if pad_id>= len(self.pads):
            return changed
        pad = self.pads[pad_id]
        if x is None:
            if dx is not None:
                if dx < 0:
                    if pad.cur_x + dx >=0:
                        pad.cur_x += dx
                        changed=True
                    elif pad.buf_x + pad.cur_x + dx >= 0:
                        pad.buf_x += dx
                        if pad.buf_x < 0:
                            pad.buf_x = 0
                            pad.cur_x = 0
                        changed = True
                    else:
                        pad.buf_x = 0
                        pad.cur_x = 0
                        changed = True
                elif dx > 0:
                    len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
                    if pad.buf_x + pad.cur_x < len_x:
                        if pad.cur_x < pad.width:
                            pad.cur_x += dx
                        else:
                            pad.buf_x += dx
                        changed = True
                    else:
                        pass  # EOL, don't expand
        else:
            if x == -1:
                len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
                len_w = len_x - pad.width
                if len_w < 0:
                    len_w = 0
                pad.buf_x = len_w
                pad.cur_x = len_x - pad.buf_x
                changed = True
            elif x == 0:
                pad.buf_x = 0
                pad.cur_x = 0
                changed = True
            else:
                len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
                if x > len_x:
                    x= len_x
                if x <= pad.width:
                    pad.buf_x = 0
                    pad.cur_x = x
                else:
                    pad.buf_x = x
                    pad.cur_x = 0
                changed = True
        if y is None:
            if dy is not None:
                if dy < 0:
                    if pad.cur_y + dy >=0:
                        pad.cur_y += dy
                        changed=True
                    elif pad.buf_y + pad.cur_y + dy >= 0:
                        pad.buf_y += dy
                        if pad.buf_y < 0:
                            pad.buf_y = 0
                            pad.cur_y = 0
                        changed = True
                    else:
                        pad.buf_y = 0
                        pad.cur_y = 0
                        changed = True
                elif dy > 0:
                    if pad.buf_y + pad.cur_y < len(pad.buffer) - dy:
                        if pad.cur_y < pad.height - 1:
                            pad.cur_y += dy
                        else:
                            pad.buf_y += dy
                        changed = True
        else:
            if y == -1:
                len_y = len(pad.buffer)
                len_h = len_y - pad.height
                if len_h < 0:
                    len_h = 0
                pad.buf_y = len_h
                pad.cur_y = len_y - pad.buf_y
                changed = True
            elif y == 0:
                pad.buf_y = 0
                pad.cur_y = 0
                changed = True
            else:
                len_y = len(pad.buffer)
                if y > len_y:
                    y= len_y
                if y <= pad.height:
                    pad.buf_y = 0
                    pad.cur_y = y
                else:
                    pad.buf_y = y
                    pad.cur_y = 0
                changed = True
        len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
        delta = len_x - (pad.buf_x + pad.cur_x)
        if delta < 0:
            if pad.cur_x + delta >= 0:
                pad.cur_x += delta
            else:
                pad.buf_x += delta
                if pad.buf_x < 0:
                    pad.buf_x = 0
                    pad.cur_x = 0
            changed = True
        if pad.cur_x == pad.width:
            pad.buf_x += 1
            pad.cur_x -= 1
            changed = True
        while pad.cur_y >= pad.height:
            pad.buf_y += 1
            if pad.cur_y > 0:
                pad.cur_y -= 1
        if pad.cur_y >= pad.height:
            print(f"Pad_y: {pad.cur_y} error")
            exit(1)
        return changed

    def create_editor(self, buffer: list[str], height: int, width:int = 0, offset_y:int =0, offset_x:int =0, schema: dict[str, list[int]] | None=None, line_no:bool=False, status_line:bool=False, debug:bool=False) -> int:
        tinp: InputEvent | None
        left_border:int = 0
        bottom_border:int = 0
        if line_no is True:
            left_border = 6
        if status_line is True:
            bottom_border = 1
        pad_id = self.pad_create(buffer, height, width, offset_y, offset_x, left_border, bottom_border, schema)
        self.repl.cursor_show()
        self.editor_esc = False
        pad = self.pad_get(pad_id)
        print("Starting editor loop")
        while self.editor_esc is False and pad is not None:
            try:
                tinp = self.input_queue.get(timeout=0.02)
            except queue.Empty:
                tinp = None
                self.repl.event_loop_tick()
                self.pad_display(pad_id)
                continue
            if debug is True:
                hex_msg = f"{bytearray(tinp.msg, encoding='utf-8')}"
                print(f"[{tinp.cmd},{tinp.msg},{hex_msg}]")
                self.input_queue.task_done()
            else:
                if tinp.cmd == "bsp":
                    if pad.cur_x + pad.buf_x > 0:
                        _ = self.pad_move(pad_id, dx = -1)
                        pad.buffer[pad.buf_y+pad.cur_y] = pad.buffer[pad.buf_y+pad.cur_y][:pad.buf_x+pad.cur_x] + pad.buffer[pad.buf_y+pad.cur_y][pad.buf_x+pad.cur_x+1:]
                    else:
                        if pad.cur_y + pad.buf_y > 0:
                            cur_idx = pad.cur_y+pad.buf_y
                            cur_line = pad.buffer[cur_idx]
                            _ = self.pad_move(pad_id, dy = -1)
                            _ = self.pad_move(pad_id, x = -1)
                            cur_idx_new = pad.cur_y+pad.buf_y
                            pad.buffer[cur_idx_new] += cur_line
                            del pad.buffer[cur_idx]
                    self.pad_display(pad_id)
                elif tinp.cmd == 'exit':
                    self.editor_esc = True
                elif tinp.cmd == "nl":
                    cur_ind = pad.cur_y+pad.buf_y
                    cur_pos = pad.cur_x + pad.buf_x
                    if cur_ind < len(pad.buffer):
                        cur_line: str = pad.buffer[cur_ind]
                    else:
                        print("error cur_line invl")
                        cur_line = ""
                        exit(1)
                    left = cur_line[:cur_pos]
                    right = cur_line[cur_pos:]
                    pad.buffer[cur_ind]=left
                    if cur_ind == len(pad.buffer) -1:
                        pad.buffer.append(right)
                    else:
                        pad.buffer.insert(cur_ind+1, right)
                    _ = self.pad_move(pad_id, dy=1, x=0)
                    self.pad_display(pad_id)
                elif tinp.cmd == "up":
                    _ = self.pad_move(pad_id, dy = -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "down":
                    _ = self.pad_move(pad_id, dy = 1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "left":
                    _ = self.pad_move(pad_id, dx = -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "right":
                    _ = self.pad_move(pad_id, dx = 1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "home":
                    _ = self.pad_move(pad_id, x=0)
                    self.pad_display(pad_id)
                elif tinp.cmd == "end":
                    _ = self.pad_move(pad_id, x= -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "PgUp":
                    _ = self.pad_move(pad_id, dy = -pad.height)
                    self.pad_display(pad_id)
                elif tinp.cmd == "PgDown":
                    _ = self.pad_move(pad_id, dy = pad.height)
                    self.pad_display(pad_id)
                elif tinp.cmd == "Start":
                    _ = self.pad_move(pad_id, x=0, y=0)
                    self.pad_display(pad_id)
                elif tinp.cmd == "End":
                    llen = len(pad.buffer) - 1
                    y = llen + pad.height
                    if y > llen:
                        y = llen
                    _ = self.pad_move(pad_id, y=y)
                    _ = self.pad_move(pad_id, x= -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "err":
                    print()
                    print(tinp.msg)
                    exit(1)
                elif tinp.cmd == "char":
                    cur_ind = pad.cur_y+pad.buf_y
                    cur_line = pad.buffer[cur_ind]
                    if ord(tinp.msg[0]) >= 32:
                        left = cur_line[:pad.buf_x+pad.cur_x]
                        right = cur_line[pad.buf_x+pad.cur_x:]
                        pad.buffer[cur_ind] = left + tinp.msg + right
                        _ = self.pad_move(pad_id, dx = 1)
                    self.pad_display(pad_id)
                else:
                    print(f"Bad state: cmd={tinp.cmd}, msg={tinp.msg}")
                    exit(1)
                self.input_queue.task_done()
                
        self.pad_display(pad_id, False)
        print("Exit edit-loop")
        return pad_id


if __name__ == "__main__":
    if sdl_available is False:
        repl = Repl(engine="TEXT")
    else:
        repl = Repl(engine="SDL2")
    if repl.repl.canvas_init(10,60) is False:
        repl.log.error("Init failed.")
        exit(1)
    buffer: list[str] = ["That", "is", "the", "initial", "long", "text"]
    id = repl.create_editor(buffer, 10, 60, 0, 0, None, True, True)
    print("Exit")
    repl.repl.exit()
