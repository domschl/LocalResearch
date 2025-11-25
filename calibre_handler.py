import logging
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import unicodedata
import os

from typing import TypedDict

from research_defs import DocumentRepresentationEntry, MetadataEntry, ResearchMetadata


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

    @staticmethod
    def _is_number(s: str) -> bool:
        roman = True
        arabic = True
        for c in s.strip():
            if c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                arabic = False
            if c not in ["I", "V", "X", "L", "C", "D", "M"]:
                roman = False
            if not arabic and not roman:
                return False
        return True

    @staticmethod
    def _clean_filename(s: str) -> str:
        bad_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        for c in bad_chars:
            s = s.replace(c, ",")
        s = s.replace("__", "_")
        s = s.replace(" _ ", ", ")
        s = s.replace("_ ", ", ")
        s = s.replace("  ", " ")
        s = s.replace("  ", " ")
        s = s.replace(",,", ",")
        s = s.replace(" ,", " ")
        s = s.replace("  ", " ")
        s = s.replace("  ", " ")
        s = s.strip()
        s = unicodedata.normalize("NFC", s)
        return s

        
    def get_human_filename(self, entry:MetadataEntry, use_series_as_subdir:bool = True) -> str:
        title = entry['title']
        short_title = entry['title_sort']
        # If title ends with roman or arabic numerals, store them as postfix:
        endfix = ""
        efs = title.split(" ")
        for ef in efs:
            if CalibreTools._is_number(ef):
                endfix = ef
                break
            else:
                ef = short_title.split(" ")[-1]
                if CalibreTools._is_number(ef):
                    endfix = ef
                    break

        short_title = CalibreTools._clean_filename(short_title)
        max_title_len = 70
        min_title_len = 30
        if len(short_title) > max_title_len:
            chars = [",", ".", "-", ":", ";"]
            p = min(
                (
                    short_title.find(c, min_title_len)
                    for c in chars
                    if short_title.find(c) != -1
                ),
                default=-1,
            )
            if p > min_title_len and p < max_title_len:
                short_title = short_title[:p]
            else:
                p = short_title.find(" ", min_title_len, max_title_len)
                if p == -1:
                    p = short_title.find("་", min_title_len, max_title_len)
                    if p != -1:
                        p = p + 1  # add 1 to include the tsheg
                if p == -1:
                    short_title = short_title[:max_title_len]
                else:
                    short_title = short_title[:p]

        short_title = short_title.strip()
        if endfix != "":
            if endfix not in short_title:
                short_title = f"{short_title} {endfix}"
        author = CalibreTools._clean_filename(entry['authors'][0])
        human_filename = f"{short_title.strip()} - {author}"
        if use_series_as_subdir is True:
            human_filename = f"{entry['series']}/{human_filename}"
        return human_filename
        
    def parse_calibre_metadata(self, filename:str, descriptor:str, existing_metadata: MetadataEntry | None = None) -> MetadataEntry|None:
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
        pub_date:str = ""
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
        identifiers:list[str] = []
        # Find records of type:
        # <dc:identifier opf:scheme="MOBI-ASIN">B0BTX2378L</dc:identifier>
        for id in xml_metadata.findall("dc:identifier", ns):
            # self.log.info(f"ID: {id.attrib} {id.text}")
            if "{http://www.idpf.org/2007/opf}scheme" in id.attrib:
                scheme:str = id.attrib["{http://www.idpf.org/2007/opf}scheme"]
                if id.text is not None:
                    sid:str = id.text
                else:
                    sid = ""
                if scheme not in ["calibre", "uuid"]:
                    identifiers.append(f"{scheme}/{sid}")
                    # self.log.info(f"{title} Identifier: {scheme}: {sid}")
        if calibre_id != "":
            identifiers.append(f"calibre_id/{calibre_id}")
        # Process cover image
        image_str = ""
        if existing_metadata and existing_metadata['icon']:
            image_str = existing_metadata['icon']
        
        if not image_str:
            cover_path = os.path.join(os.path.dirname(filename), "cover.jpg")
            image_str = ResearchMetadata.encode_image(cover_path)

        # Populate representations
        representations: list[DocumentRepresentationEntry] = []
        dir_path = os.path.dirname(filename)
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                ext = os.path.splitext(f)[1].lower().lstrip('.')
                if ext in ['txt', 'pdf', 'md']:
                    representations.append(DocumentRepresentationEntry(
                        doc_descriptor=descriptor,
                        hash="", # Hash not calculated here
                        format=ext,
                        doc_date=date_added if date_added else datetime.now(timezone.utc).isoformat()
                    ))

        metadata:MetadataEntry = MetadataEntry(
            uuid=uuid,
            representations=representations,
            authors=creators,
            identifiers=identifiers,
            languages=languages,
            context=self.calibre_path,
            creation_date=date_added,
            publication_date=pub_date,
            publisher=publisher,
            series=series,
            tags=subjects,
            title=title,
            title_sort=title_sort,
            normalized_filename=filename,
            description=description,
            icon=image_str
            )
        
        return metadata
    
    
