import logging
from typing import TypedDict


class SyncTarget(TypedDict):
    types: list[str]
    path: str
    file_types: list[str]


class SyncTools:
    def __init__(self, target:SyncTarget):
        self.log:logging.Logger = logging.Logger("SyncTool")
        self.target:SyncTarget = target
