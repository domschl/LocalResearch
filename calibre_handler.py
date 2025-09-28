import logging
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from typing import TypedDict

from research_defs import DocumentRepresentationEntry, MetadataEntry


class CalibrePrefixes(TypedDict):
    de: dict[str, list[str]]
    en: dict[str, list[str]]


calibre_prefixes:dict[str, dict[str, list[str]]] = {
    "de": {"prefixes": ["Der", "Die", "Das", "Ein", "Eine"]},
    "en": {"prefixes": ["The", "A", "An"]},
}


class CalibreTools:
    def __init__(self, calibre_path:str):
        self.log:logging.Logger = logging.getLogger("CalibreTools")
        self.calibre_path:str = calibre_path
        

    def parse_calibre_metadata(self, filename:str) -> MetadataEntry|None:
        root_xml = ET.parse(filename).getroot()
        # Namespace map
        ns = {
            "opf": "http://www.idpf.org/2007/opf",
            "dc": "http://purl.org/dc/elements/1.1/",
        }
        # Extract xml_metadata
        xml_metadata = root_xml.find("opf:metadata", ns)

        if xml_metadata is None:
            self.log.error(f"No metadata found in OPF file for: {filename}")
            return None

        title_md = xml_metadata.find("dc:title", ns)
        title: str = str(title_md.text) if title_md is not None else ""
        description_md = xml_metadata.find("dc:description", ns)
        description: str = str(description_md.text) if description_md is not None else ""

        # creator = xml_metadata.find("dc:creator", ns)
        # creators = creator.text.split(", ") if creator is not None else []
        # Get all authors from 'role': <dc:creator opf:file-as="Berlitz, Charles &amp; Moore, William L." opf:role="aut">Charles Berlitz</dc:creator>
        # id.attrib["{http://www.idpf.org/2007/opf}scheme"]
        creators: list[str] = []
        for creator in xml_metadata.findall("dc:creator", ns):
            if "{http://www.idpf.org/2007/opf}role" in creator.attrib:
                if (
                    creator.attrib["{http://www.idpf.org/2007/opf}role"]
                    == "aut"
                ):
                    if isinstance(creator.text, str) and "," in creator.text:
                        self.log.error(
                            f"Author name contains comma: {creator.text}"
                        )
                    creators.append(str(creator.text))

        subjects_md = xml_metadata.findall("dc:subject", ns)
        subjects: list[str] = [str(subject.text) for subject in subjects_md]
        languages_md = xml_metadata.findall("dc:language", ns)
        languages: list[str] = [str(language.text) for language in languages_md]
        uuids_md = xml_metadata.findall("dc:identifier", ns)
        uuid: str = ""
        calibre_id: str = ""
        for u in uuids_md:
            if "id" not in u.attrib:
                continue
            if u.attrib["id"] == "calibre_id":
                calibre_id = str(u.text)
            if u.attrib["id"] == "uuid_id":
                uuid = str(u.text)

        publisher_md = xml_metadata.find("dc:publisher", ns)
        publisher: str = str(publisher_md.text) if publisher_md is not None else ""
        date_md = xml_metadata.find("dc:date", ns)
        date: str = str(date_md.text) if date_md is not None else ""
        # convert to datetime, add utc timezone
        if date != "":
            if '.' in date:                               
                pub_date = (
                    datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f%z")
                    .replace(tzinfo=timezone.utc)
                    .isoformat()
                )
            else:
                pub_date = (
                    datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
                    .replace(tzinfo=timezone.utc)
                    .isoformat()
                )

        series: str = ""
        date_added: str = ""
        title_sort: str = ""
        timestamp: str = ""
        for meta in xml_metadata.findall("opf:meta", ns):
            if "name" in meta.attrib:
                if meta.attrib["name"] == "calibre:series":
                    series = meta.attrib["content"]
                if meta.attrib["name"] == "calibre:timestamp":
                    timestamp = str(meta.attrib["content"])
                    # timestamp can be 2023-11-11T17:03:48.214591+00:00 or 2023-11-11T17:03:48+00:00
                    timestamp = timestamp.split(".")[0]
                    if timestamp.endswith("+00:00"):
                        date_added_dt = datetime.strptime(
                            timestamp, "%Y-%m-%dT%H:%M:%S%z"
                        )
                    else:
                        date_added_dt = datetime.strptime(
                            timestamp, "%Y-%m-%dT%H:%M:%S"
                        )

                    date_added = date_added_dt.replace(
                        tzinfo=timezone.utc
                    ).isoformat()
                if meta.attrib["name"] == "calibre:title_sort":
                    title_sort = meta.attrib["content"]
                    for (
                        lang
                    ) in (
                        calibre_prefixes
                    ):  # remove localized prefixes ", The", ", Der", etc. (curr: DE, EN)
                        prefixes = calibre_prefixes[lang]["prefixes"]
                        for prefix in prefixes:
                            ending = f", {prefix}"
                            if title_sort.endswith(ending):
                                title_sort = title_sort[: -len(ending)]
                                break
                    # Check if starts with lowercase
                    if title_sort[0].islower():
                        # check if second character is uppercase (iPad, jQuery, etc.)
                        if len(title_sort) > 1 and title_sort[1].islower():
                            self.log.warning(
                                f"Shortened title starts with lowercase: {title_sort}, consider fixing!"
                            )
                            # title_sort = title_sort[0].upper() + title_sort[1:]  # automatic fixing can go wrong (jQuery, etc.)
        identifiers = []
        # Find records of type:
        # <dc:identifier opf:scheme="MOBI-ASIN">B0BTX2378L</dc:identifier>
        for id in xml_metadata.findall("dc:identifier", ns):
            # self.log.info(f"ID: {id.attrib} {id.text}")
            if "{http://www.idpf.org/2007/opf}scheme" in id.attrib:
                scheme = id.attrib["{http://www.idpf.org/2007/opf}scheme"]
                sid = id.text
                if scheme not in ["calibre", "uuid"]:
                    identifiers.append(f"{scheme}/{sid}")
                    # self.log.info(f"{title} Identifier: {scheme}: {sid}")
        metadata: MetadataEntry
        return None  # XXX
    
    
