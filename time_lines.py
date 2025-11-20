import logging
import re

from typing import TypedDict, Any, cast

from indralib.indra_time import IndraTime, IndraTimeInterval
from indralib.indra_event import IndraEvent

from research_defs import MetadataEntry, TextLibraryEntry
from research_tools import DocumentTable
from search_tools import SearchTools

class TimeLineEvent(TypedDict):
    jd_event: tuple[float, float | None] | tuple[float]
    eventdata: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    metadata: dict[str, Any]   # MetadataEntry  # pyright: ignore[reportExplicitAny]

class TimeLines:
    def __init__(self):
        self.log: logging.Logger = logging.getLogger("IndraTools")
        # Event: (jd-date, content-dict, metadata-dict)
        self.tl_events: list[TimeLineEvent] = []

    def sort_tl_events(self):
        def jd_interval_sorter(jds: tuple[float, float | None] | tuple[float]):
            if len(jds) == 1:
                ls = jds + jds
            else:
                ls = jds
            return (ls[0], ls[1])
        self.tl_events = sorted(self.tl_events, key=lambda x: jd_interval_sorter(x['jd_event']))

    @staticmethod
    def search_keys(text: str, keys: list[str]):
        return SearchTools.match(text, keys)

    def search_events(
        self,
        time: str | None = None,
        domains: list[str] | str | None = None,
        keywords: list[str] | str | None = None,
        in_intervall: bool = True,
        full_overlap: bool = True,
        partial_overlap: bool = True,
    ) -> list[TimeLineEvent]:
        if time is not None:
            if time.startswith('"') and time.endswith('"'):
                time = time[1:-1]
            jd_time = IndraTime.string_time_to_julian(time)
            if jd_time is None:
                self.log.error(f"Invalid time specified: {time}")
                return []
            start_time = jd_time[0]
            if len(jd_time) > 1 and jd_time[1] is not None:
                end_time = jd_time[1]
            else:
                end_time = start_time
        else:
            start_time = None
            end_time = None
        if domains is not None:
            if not isinstance(domains, list):
                domains = [domains]
        if keywords is not None:
            if not isinstance(keywords, list):
                keywords = [keywords]
        result: list[TimeLineEvent] = []
        for event in self.tl_events:
            if time is not None and start_time is not None:
                b_time = False
                event_start = event['jd_event'][0]
                if len(event['jd_event']) > 1 and event['jd_event'][1] is not None:
                    event_end = event['jd_event'][1]
                else:
                    event_end = event['jd_event'][0]
                if event_start >= start_time and end_time is not None and event_end <= end_time:
                    if in_intervall:
                        b_time = True
                else:
                    if event_start < start_time and end_time is not None and event_end > end_time:
                        if full_overlap:
                            b_time = True
                    else:
                        if partial_overlap:
                            if (
                                event_start >= start_time
                                and end_time is not None
                                and event_start < end_time
                                and event_end > end_time
                            ):
                                b_time = True
                            if (
                                end_time is not None
                                and event_end <= end_time
                                and event_end > start_time
                                and event_start < start_time
                            ):
                                b_time = True
                if not b_time:
                    continue
            if domains is None:
                b_domains = True
            else:
                b_domains = False
                pos_dom = False
                meta_domain = cast(str, event['metadata']['context']).lower()
                for domain in domains:
                    if domain.startswith("!"):
                        continue
                    pos_dom = True
                    if domain.lower() in meta_domain or IndraEvent.mqcmp(
                        domain.lower(), meta_domain
                    ):
                        b_domains = True
                        break
                if pos_dom is False:
                    b_domains = True
                if b_domains is True:
                    for domain in domains:
                        if domain.startswith("!") is False:
                            continue
                        else:
                            domain = domain[1:]
                        if domain.lower() in meta_domain or IndraEvent.mqcmp(
                            domain.lower(), meta_domain
                        ):
                            b_domains = False
                            break
            if not b_domains:
                continue
            if keywords is None:
                b_keywords = True
            else:
                b_keywords = False
                event_keys: list[str] = []
                for k in event['eventdata']:
                    val = event['eventdata'][k]  # pyright: ignore[reportAny]
                    val_txt = f"{val}"
                    event_keys += [k, val_txt]
                for k in event['metadata']:
                    val = event['metadata'][k]  # pyright: ignore[reportAny]
                    val_txt = f"{val}"
                    event_keys += [k, val_txt]
                try:
                    event_keys_txt = ' '.join(event_keys)
                except Exception as e:
                    self.log.error(f"couldn't join [Err: {e}], event_keys: {event_keys}")
                    exit(1)
                b_keywords = self.search_keys(event_keys_txt, keywords)
            if not b_keywords:
                continue
            result.append(event)
        return result

    def get_date_string_from_event(self, jd_event: IndraTimeInterval) -> str | None:
        date_points : list[str] = []
        for date_part in jd_event:
            if date_part is not None:
                date_points.append(IndraTime.julian_to_string_time(date_part))
            date = None
        if len(date_points) == 1:
            date = date_points[0]
        elif len(date_points) == 2:
            date = f"{date_points[0]} - {date_points[1]}"
        else:
            self.log.warning(f"Invalid date range: {date_points}")
            date = None
        return date

    def get_event_text(self, event_data: dict[str, Any]) -> str:  # pyright: ignore[reportExplicitAny]
        event_text = ""
        for ev in event_data:
            event_text += f"{ev}: {event_data[ev]}, "
        if len(event_text) >= 2:
            event_text = event_text[:-2]
        return event_text

    def print_events(
        self,
        events: list[TimeLineEvent],
        filename: str | None = None,
        length: int | None = None,
        header: bool = False,
        format: str | None = None,
        emph_words: list[str] | None = None,
    ):
        emph_words_no_esc: list[str] = []
        if emph_words is None:
            emph_words = []
        for we in emph_words:
            if we.startswith("*"):
                we = we[1:]
            if we.endswith("*"):
                we = we[:-1]
            if "*" in we:
                we = we.split("*")
                for w in we:
                    if len(w) > 0:
                        emph_words_no_esc.append(w)
            else:
                if len(we) > 0:
                    emph_words_no_esc.append(we)
        if filename is not None:
            f = open(filename, "w")
        else:
            f = None
        if format is None:
            if header is True:
                if f is not None:
                    _ = f.write("| Date                      | Event |\n")
                    _ = f.write("|---------------------------|-------|\n")
                else:
                    print("| Date                      | Event |")
                    print("|---------------------------|-------|")
            for event in events:
                event_text = self.get_event_text(event['eventdata'])
                if length is not None and len(event_text) > length:
                    event_text = event_text[:length] + "..."
                date_text = self.get_date_string_from_event(event['jd_event'])
                if date_text is not None:
                    if f is not None:
                        _ = f.write(f"| {date_text:24s} | {event_text} |\n")
                    else:
                        print(f"| {date_text:24s} | {event_text} |")
        elif format == "ascii":
            for event in events:
                pass
        if f is not None:
            f.close()
        return

    def date_from_text(self, year: str) -> IndraTimeInterval | None:
        jd_date = IndraTime.string_time_to_julian(year)
        return jd_date

    def add_book_events(self, text_lib_entries: dict[str, TextLibraryEntry], metadata_lib:dict[str, MetadataEntry]) -> tuple[int, int]:
        date_regex = r"\b(18|19|20)\d{2}\b"
        n_dates = 0
        n_books = 0
        for entry_hash in text_lib_entries:
            text_lib_entry = text_lib_entries[entry_hash]
            if 'calibre' not in text_lib_entry['source_name'].lower():
                continue
            book_metadata = metadata_lib[entry_hash]
            n_books += 1
            text = text_lib_entry['text']
            # Find all occurences of date_regex in text:
            dates = [(match.start(), match.group()) for match in re.finditer(date_regex, text)]
            # self.text_lib[book_path]['dates'] = dates
            n_dates += len(dates)
            if len(dates) > 0:
                preamb = 50
                posttxt = 20
                for sample in dates:
                    snip = text[sample[0]-preamb:sample[0]+posttxt]
                    for c in ["\n", "\r", "\t"]:
                        snip = snip.replace(c, " ")
                    jd_date = self.date_from_text(sample[1])
                    if jd_date is not None:
                        meta_var:dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]
                        for k in book_metadata.keys():
                            meta_var[k] = book_metadata[k]
                            meta_var['position_offset'] = sample[0]
                        event: TimeLineEvent = {'jd_event': jd_date, 'eventdata': {'Title': book_metadata['title'], 'Text_snip': snip}, 'metadata': meta_var}
                        self.tl_events.append(event)
                        n_dates += 1
                    else:
                        self.log.error(f"Invalid date encountered: {sample[1]}")
        self.sort_tl_events()
        self.log.info(f"{n_dates} dates found")
        return n_books, n_dates

    def add_date_table_events(self, table: DocumentTable, tables_unique_domain_names: list[str], check_order: bool = True) -> tuple[int, int]:
        event_cnt = 0
        events_skipped = 0
        if len(table["columns"]) < 2:
            self.log.debug(
                f"Table {table['columns']}, {table['metadata']} has less than 2 columns, skipping"
            )
            return event_cnt, 0
        col_nr = len(table["columns"])
        for row in table["rows"]:
            if len(row) != col_nr:
                self.log.warning(
                    f"Table {table['columns']}: Row {row} has {len(row)} columns, expected {col_nr}, invalid Table"
                )
                return event_cnt, 0
        if len(table["columns"]) == 0 or table["columns"][0] != "Date":
            self.log.debug(
                f"Table {table['columns']}: First column is not 'Date', skipping"
            )
            return event_cnt, 0
        if "domain" not in table["metadata"]:
            self.log.debug(
                f"Table {table['columns']}: Metadata has no 'domain' key, skipping"
            )
            return event_cnt, 0
        if table["metadata"]["domain"] in tables_unique_domain_names:
            self.log.warning(
                f"Table {table['columns']}: Domain {table['metadata']['domain']} already exists, skipping"
            )
            return event_cnt, 0
        cur_dom: str = cast(str, table["metadata"]["domain"])
        tables_unique_domain_names.append(cur_dom)
        last_start_time = None
        last_end_time = None
        table_sorted = True
        for index, row in enumerate(table["rows"]):
            raw_date = row[0]
            if raw_date == 'Date' or (raw_date.startswith('-') and raw_date.endswith('-')):
                print(f"Table restarted in the middle! {table["columns"]} at row {index}: {raw_date}")
                continue
            try:
                jd_date = IndraTime.string_time_to_julian(raw_date)
                if jd_date is None:
                    self.log.error(f"Invalid date: {raw_date}")
                    continue
                for date_part in jd_date:
                    if date_part is None:
                        self.log.warning(
                            f"Table {table['columns']}: Row {row} has invalid date {raw_date}"
                        )
                        jd_date = None
                if jd_date is None:
                    events_skipped += 1
                    continue
                if len(jd_date) == 2 and jd_date[1] is not None:
                    if jd_date[1] < jd_date[0]:
                        self.log.error(
                            f"Table {table['columns']}: Row {row}: end-date is earlier than start-state, invalid!"
                        )
                        jd_date = None
                        events_skipped += 1
                        continue
            except ValueError:
                self.log.warning(
                    f"Table {table['columns']}: Row {row} has invalid date {raw_date}"
                )
                events_skipped += 1
                continue
            if last_start_time is not None:
                if last_start_time > jd_date[0]:
                    events_skipped += 1
                    table_sorted = False
                    if check_order is True:
                        self.log.error(
                            f"Table {table['columns']}: Row {row}: start-date is later than start-state of previous row, invalid order!"
                        )
                        self.log.warning(
                            f"{IndraTime.julian_to_string_time(last_start_time)} -> {IndraTime.julian_to_string_time(jd_date[0])}, {events_skipped}"
                        )
                    continue
                elif last_start_time == jd_date[0]:
                    if last_end_time is not None:
                        if len(jd_date) == 1 or jd_date[1] is None:
                            events_skipped += 1
                            table_sorted = False
                            if check_order is True:
                                self.log.error(
                                    f"Table {table['columns']}: Row {row}: interval-less record is later than interval with same start-date, it should be before, invalid order!"
                                )
                                continue
                        elif last_end_time > jd_date[1]:
                            events_skipped += 1
                            table_sorted = False
                            if check_order is True:
                                self.log.error(
                                    f"Table {table['columns']}: Row {row}: intervals with same start-date, record with earlier end-date after later end-date, invalid order!"
                                )
                                continue
            last_start_time = jd_date[0]
            if len(jd_date) > 1:
                last_end_time = jd_date[1]
            else:
                last_end_time = None
            event_data = {}
            for i in range(1, col_nr):
                event_data[table["columns"][i]] = row[i]
            event: TimeLineEvent = {'jd_event': jd_date, 'eventdata': event_data, 'metadata': table["metadata"]}
            self.tl_events.append(event)
            event_cnt += 1

        def jd_str_interval_sorter(row: list[str]) -> IndraTimeInterval:
            jdi = IndraTime.string_time_to_julian(row[0])
            if jdi is None:
                raise ValueError("f{row[0]} is not a valid time spec")
            if len(jdi) == 1:
                jdi += jdi
            return (jdi[0], jdi[1])

        if table_sorted is False:
            sorted_table = sorted(table["rows"], key=jd_str_interval_sorter)
            self.log.error(f"Table NOT sorted: {sorted_table}")
        #     print()
        #     print("-----------------------------------------------------------------")
        #     self.print_table(table["columns"], sorted_table)
        #     print("-----------------------------------------------------------------")
        #     print()

        self.sort_tl_events()
        return event_cnt, events_skipped

    def add_notes_events(self, tables:list[DocumentTable]):  
        event_cnt = 0
        skipped_cnt = 0
        tables_unique_domain_names: list[str] = []
        for table in tables:
            new_evs, new_skipped = self.add_date_table_events(
                table, tables_unique_domain_names, check_order=True
            )
            event_cnt += new_evs
            skipped_cnt += new_skipped
        self.log.info(
            f"Found {len(self.tl_events)} (added {event_cnt}) events in notes, skipped {skipped_cnt}"
        )

    def notes_rest(self, do_timeline: bool, format: str | None, timespec: str | None, domains: str | list[str] | None, keywords: str | list[str] | None):
        if do_timeline is True:
            if format != "ascii":
                format = None
            time_par = timespec
            if time_par is not None:
                if time_par == "":
                    time_par = None
            if isinstance(domains, list):
                domains_par = domains
            elif domains is None or domains == "":
                domains_par = None
            else:
                domains_par = domains.split(" ")
            if isinstance(keywords, list):
                keywords_par = keywords
            elif keywords is None or keywords == "":
                keywords_par = None
            else:
                keywords_par = keywords.split(" ")
            evts = self.search_events(
                time=time_par,
                domains=domains_par,
                keywords=keywords_par,
                in_intervall=False,
                full_overlap=True,
                partial_overlap=False,
            )
            emph_keys: list[str] = []
            if domains_par is not None:
                for dom in domains_par:
                    if dom.startswith("!"):
                        continue
                    emph_keys.append(dom)
            if keywords_par is not None:
                for k in keywords_par:
                    if k.startswith("!"):
                        continue
                    emph_keys += k.split("|")
            if time_par is not None:
                if len(evts) > 0:
                    print(" --------- < ----- > ---------")
                    self.print_events(evts, format=format, emph_words=emph_keys)
                evts = self.search_events(
                    time=time_par,
                    domains=domains_par,
                    keywords=keywords_par,
                    in_intervall=False,
                    full_overlap=False,
                    partial_overlap=True,
                )
                if len(evts) > 0:
                    print(" --------- <| ----- |> ---------")
                    self.print_events(evts, format=format, emph_words=emph_keys)
                evts = self.search_events(
                    time=time_par,
                    domains=domains_par,
                    keywords=keywords_par,
                    in_intervall=True,
                    full_overlap=False,
                    partial_overlap=False,
                )
                if len(evts) > 0:
                    print(" --------- | ----- | ---------")
                    self.print_events(evts, format=format, emph_words=emph_keys)
            else:
                self.print_events(evts, format=format, emph_words=emph_keys)
