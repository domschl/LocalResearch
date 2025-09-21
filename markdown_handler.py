import logging
from vector_store import Metadata


class MarkdownTools:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("MarkdownTools")

    def parse_markdown(self, md_text:str) -> tuple[metadata:Metadata, body:str]:
        pass
