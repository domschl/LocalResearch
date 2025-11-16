import logging
import os
from typing import TypedDict


class SyncTarget(TypedDict):
    types: list[str]
    path: str
    file_types: list[str]


class FileMetadata(TypedDict):
    path: str
    date_changed: float
    size: int
    
class SyncTools:
    def __init__(self, target:SyncTarget):
        self.log:logging.Logger = logging.Logger("SyncTool")
        self.target:SyncTarget = target
        
    def file_meta(self, filepath:str) -> tuple[float, int] | None:
        try:
            stat:os.stat_result = os.stat(filepath)
            return  stat.st_mtime, stat.st_size
        except Exception as e:
            self.log.warning(f"Error getting file modification date {filepath}: {e}")
        return None
        
    def update(self, rel_path:str):
        pass
