from typing import TypedDict


class DocumentRepresentationEntry(TypedDict):
    doc_descriptor: str
    hash: str
    format: str
    doc_date: str

    
class MetadataEntry(TypedDict):
    uuid: str
    representations: list[DocumentRepresentationEntry]
    authors: list[str]
    identifiers: list[str]
    languages: list[str]
    context: str
    creation_date: str
    publication_date: str
    series: str
    tags: list[str]
    title: str
    title_sort: str
    normalized_filename: str
    description: str
    icon: str

    
