import queue
from dataclasses import dataclass
from abc import abstractmethod
from abc import ABC

@dataclass()
class InputEvent:
    cmd: str
    msg: str

class ReplIO(ABC):
    @abstractmethod
    def __init__(self, que:queue.Queue[InputEvent]):
        pass

    @abstractmethod
    def exit(self):
        pass
    
    @abstractmethod
    def cursor_hide(self):
        pass

    @abstractmethod
    def cursor_show(self):
        pass
    
    @abstractmethod
    def cursor_start_offset_get(self) -> tuple[int, int]:
        pass
    
    @abstractmethod
    def canvas_update_size(sel0f) -> tuple[int, int]:
        pass

    @abstractmethod
    def canvas_init(self, size_x:int =0, size_y:int=0) -> bool:
        pass

    @abstractmethod
    def canvas_print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        pass

    @abstractmethod
    def canvas_render_start(self):
        pass

    @abstractmethod
    def canvas_render_show(self):
        pass

    @abstractmethod
    def event_loop_tick(self):
        pass

    @abstractmethod
    def color_set(self, fg:list[int], bg:list[int] | None):
        pass
