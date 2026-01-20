import logging
import queue
import os
from typing import override, cast

import sdl2  # pyright: ignore[reportMissingTypeStubs]
import sdl2.ext # pyright: ignore[reportMissingTypeStubs]
import sdl2.sdlttf # pyright: ignore[reportMissingTypeStubs]
import ctypes

from led_repl_io import ReplIO, InputEvent

class Sdl2ReplIO(ReplIO):
    def __init__(self, que:queue.Queue[InputEvent]):
        self.log: logging.Logger = logging.getLogger("TextReplIO")
        self.input_queue: queue.Queue[InputEvent] = que
        self.cur_x_offset: int = 0
        self.cur_y_offset: int = 0
        self.cur_pos_x: int = 0
        self.cur_pos_y: int = 0
        self.cur_active: bool = True
        self.fg_color: list[int] = [0xff, 0xff, 0xff, 0xff]
        self.bg_color: list[int] = [0, 0, 0, 0xff]

        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)  # pyright: ignore[reportUnknownMemberType]
        sdl2.sdlttf.TTF_Init()
        self.event_loop_active: bool = True
        WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
        window = sdl2.ext.Window("SDL2 Text Example", size=(WINDOW_WIDTH, WINDOW_HEIGHT),
                                 flags = (sdl2.SDL_WINDOW_ALLOW_HIGHDPI |  sdl2.SDL_RENDERER_ACCELERATED))
        window.show()
        self.renderer:sdl2.ext.Renderer = sdl2.ext.Renderer(window)

        rw: ctypes.c_int = ctypes.c_int(0)
        rh: ctypes.c_int = ctypes.c_int(0)
        #prw = ctypes.POINTER(ctypes.c_int(rw))
        #prh: ctypes.POINTER(ctypes.c_int)
        sdl2.SDL_GetRendererOutputSize(self.renderer.sdlrenderer, rw, rh);  # pyright: ignore[reportUnknownMemberType]
        if rw.value != WINDOW_WIDTH:
            widthScale = rw.value / WINDOW_WIDTH
            heightScale = rh.value / WINDOW_HEIGHT

            if widthScale != heightScale:
                self.log.warning("WARNING: width scale != height scale")
            else:
                print(f"Scale: {widthScale}")

            # sdl2.SDL_RenderSetScale(self.renderer.sdlrenderer, widthScale, heightScale);

        font_path = "./Resources/IosevkaNerdFontMono-Regular.ttf"
        # font_path = "./Resources/BabelStoneTibetan.ttf"
        if os.path.exists(font_path) is False:
            self.log.error(f"Font {font_path} does not exist")
        # sdl2.ext.RenderSetScale(self.renderer,2,2)
        self.font_mag:int = 1 # 2  # 1 normal 2 retina
        self.dpi:int = 144
        font_size = 10 * self.font_mag
        self.font: sdl2.sdlttf.TTF_Font = sdl2.sdlttf.TTF_OpenFontDPI(font_path.encode('utf-8'), font_size, self.dpi, self.dpi)  # pyright: ignore[reportUnknownMemberType] # , reportUnannotatedClassAttribute]
        sdl2.sdlttf.TTF_SetFontHinting(self.font, sdl2.sdlttf.TTF_HINTING_LIGHT_SUBPIXEL)  # pyright: ignore[reportUnknownMemberType]
        rect = self.render_text("a", 0, 0)
        if rect is not None:
            self.char_width: int = rect.w
            self.char_height: int = rect.h
            print(f"Char-sizes: {self.char_width}, {self.char_height}")
        else:
            self.log.error("Cannot determine character dimensions!")
        self.line_spacing_extra:int = 0
        script = "Tibt".encode('utf-8')
        sdl2.sdlttf.TTF_SetFontScriptName(self.font, script)  # pyright:ignore[reportUnknownMemberType]

    @override
    def exit(self):
        sdl2.sdlttf.TTF_CloseFont(self.font)  # pyright: ignore[reportUnknownMemberType]
        sdl2.sdlttf.TTF_Quit()
        sdl2.SDL_Quit()  # pyright: ignore[reportUnknownMemberType]

    @override
    def color_set(self, fg:list[int], bg:list[int] | None):
        self.fg_color = fg
        if bg is not None:
            self.bg_color = bg

    @override
    def canvas_init(self, size_x:int =0, size_y:int=0) -> bool:
        return True

    # Function to render text
    def render_text(self, text:str, x:int, y:int) -> sdl2.SDL_Rect | None:
        if text == "":
            return
        color_fg = sdl2.SDL_Color(self.fg_color[0], self.fg_color[1], self.fg_color[2])
        color_bg = sdl2.SDL_Color(self.bg_color[0], self.bg_color[1], self.bg_color[2])

        # Surface = sdl2.sdlttf.TTF_RenderUTF8_Solid(self.font, text.encode(), color)
        surface = sdl2.sdlttf.TTF_RenderUTF8_LCD(self.font, text.encode(), color_fg, color_bg)  # pyright:ignore[reportUnknownMemberType, reportUnknownVariableType]
        texture = sdl2.SDL_CreateTextureFromSurface(self.renderer.sdlrenderer, surface)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        rect = sdl2.SDL_Rect(x, y, surface.contents.w // self.font_mag, surface.contents.h // self.font_mag)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        sdl2.SDL_FreeSurface(surface)  # pyright: ignore[reportUnknownMemberType]
        sdl2.SDL_RenderCopy(self.renderer.sdlrenderer, texture, None, rect)  # pyright: ignore[reportUnknownMemberType]
        sdl2.SDL_DestroyTexture(texture)  # pyright: ignore[reportUnknownMemberType]
        return rect

    @override
    def canvas_print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        _ = self.render_text(msg, x*self.char_width, y*(self.char_height + self.line_spacing_extra))
        self.cur_pos_x = x + len(msg)
        self.cur_pos_y = y

    @override
    def event_loop_tick(self):
        events: list[sdl2.SDL_Event] = cast(list[sdl2.SDL_Event], sdl2.ext.get_events())
        for event in events:
            if event.type == sdl2.SDL_QUIT:  # pyright: ignore[reportAny]
                msg = InputEvent("exit", "")
                self.input_queue.put_nowait(msg)
                continue
            if event.type == sdl2.SDL_KEYDOWN:  # pyright: ignore[reportAny]
                key_sym:int = event.key.keysym.sym  # pyright: ignore[reportAny]
                key_code:int = event.key.keysym.scancode  # pyright: ignore[reportAny]
                key_mod:int = event.key.keysym.mod  # pyright: ignore[reportAny]

                print(f"{hex(key_sym)} {key_sym} {key_code} {key_mod}")
                if key_code == 82: # up
                    msg = InputEvent("up", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_code == 81: # down
                    msg = InputEvent("down", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_code == 80:  # left
                    msg = InputEvent("left", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_code == 79: # up
                    msg = InputEvent("right", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_sym == 8:
                    msg = InputEvent("bsp", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_sym == 13:
                    msg = InputEvent("nl", "")
                    self.input_queue.put_nowait(msg)
                    continue
                continue
            if event.type == sdl2.SDL_TEXTINPUT:  # pyright: ignore[reportAny]
                text_char:str = event.text.text.decode('utf-8')  # pyright: ignore[reportAny]
                _text_type:int = event.text.type  # pyright: ignore[reportAny]
                msg = InputEvent("char", text_char)
                self.input_queue.put_nowait(msg)
                continue
        #self.renderer.present()

    @override
    def canvas_render_start(self):
        self.renderer.clear()  # pyright: ignore[reportUnknownMemberType]
        return

    @override
    def canvas_render_show(self):
        if self.cur_active is True:
            sdl2.SDL_SetRenderDrawColor(self.renderer.sdlrenderer,self.fg_color[0],self.fg_color[1],self.fg_color[2],self.fg_color[3])  # pyright: ignore[reportUnknownMemberType]
            sdl2.SDL_RenderDrawLine(self.renderer.sdlrenderer, self.cur_pos_x * self.char_width, self.cur_pos_y * self.char_height, self.cur_pos_x * self.char_width, (self.cur_pos_y + 1) * self.char_height)  # pyright: ignore[reportUnknownMemberType]
        self.renderer.present()
        return

    @override
    def cursor_start_offset_get(self) -> tuple[int, int]:
        return self.cur_x_offset, self.cur_y_offset

    @override
    def cursor_show(self):
        self.cur_active = True
        return

    @override
    def cursor_hide(self):
        self.cur_active = False
        pass

    @override
    def canvas_update_size(self) -> tuple[int, int]:
        self.cols:int
        self.rows:int
        self.cols, self.rows = os.get_terminal_size()
        return (self.cols, self.rows)
