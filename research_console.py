import logging
import readline
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import atexit
from typing import cast
import subprocess
from text_format import TextFormat

print("\rStarting...\r", end="", flush=True)

from research_defs import get_files_of_extensions, ProgressState, SearchResultEntry
from vector_store import VectorStore
from indralib.indra_time import IndraTime
from document_store import DocumentStore
from text_format import TextParse
from search_tools import SearchTools
from audiobook_handler import AudiobookGenerator
from timeline_handler import TimelineExtractor

def repl(ds: DocumentStore, vs: VectorStore, log: logging.Logger):
    history_file = os.path.join(os.path.expanduser("~/.config/local_research"), "repl_history")
    config_path = os.path.expanduser("~/.config/local_research")
    audiobook_gen = AudiobookGenerator(config_path)
    tf = TextFormat()
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Read history: {e}")
        
    readline.set_history_length(1000)
    _ = atexit.register(readline.write_history_file, history_file)
    previous_search_results: list[SearchResultEntry] = []
    tp = TextParse()
    print("`help` for overview, `exit` or Ctrl-D to exit")
    while True:
        try:
            # Get user input with a prompt
            line: str = input(">>> ")
            ind:int = line.find(' ')
            key_vals:dict[str,str]={}
            if ind != -1:
                command = line[:ind].lower()
                text_argument = line[ind+1:]
                parse_arguments, key_vals = tp.parse_keys(text_argument)
                arguments = parse_arguments
            else:
                command = line.lower()
                text_argument = ""
                arguments: list[str] = []
            if command == 'test':
                print(f"Command: {command}")
                print(f"Arguments: {arguments}")
                print(f"Key-Values: {key_vals}")
                continue
            elif command == 'sync':
                log.info("Starting sync...")
                if 'force' in arguments:
                    force = True
                else:
                    force = False
                if 'retry' in arguments:
                    retry = True
                else:
                    retry = False

                def progress_sync(ps: ProgressState):
                    cols, _= os.get_terminal_size()
                    blnk = "\r" + ' ' * (cols - 1)
                    print(blnk, end="")
                    try:
                        progress = tf.progress_bar_string(ps['percent_completion'], 8)
                    except:
                        progress = ' ' * 10
                    print(f"\r{progress} {ps['state']}", end="", flush=True)
                    if ps['finished'] is True:
                        print()
                    
                errors = ds.sync_texts(force, retry, progress_sync, None)
                print()

                if len(errors) > 0:
                    print("Errors and issues:")
                    for error in errors:
                        print(error)
                    
            elif command == 'check':
                if len(arguments) == 0 or 'pdf' in arguments or (len(arguments)==1 and 'clean' in arguments):
                    if 'clean' in arguments:
                        clean = True
                    else:
                        clean = False
                    entry_count, failure_count, orphan_count, orphan2_count, deleted_count, deleted2_count, missing_count, cache_changed = ds.check_pdf_cache(clean)
                    header = ["PDF Cache", "Count"]
                    al_p2: list[bool|None]|None = [True, False]
                    rows_p2: list[list[str]] = []
                    rows_p2.append(["Cache entries", f"{entry_count}"])
                    rows_p2.append(["Extract failures", f"{failure_count}"])
                    rows_p2.append(["Orphans", f"{orphan_count}+{orphan2_count}"])
                    if deleted_count > 0:
                        rows_p2.append(["Debris removed", f"{deleted_count}"])
                    if deleted_count > 0:
                        rows_p2.append(["Records removed", f"{deleted2_count}"])
                    rows_p2.append(["Missing cache files", f"{missing_count}"])
                    _ = tf.print_table(header, rows_p2, al_p2)
                    if cache_changed is True:
                        print("PDF cache file updated")
                    print()

                if len(arguments) == 0 or 'sha256' in arguments or (len(arguments)==1 and 'clean' in arguments):
                    if 'clean' in arguments:
                        clean = True
                    else:
                        clean = False
                    entries, debris, deleted = ds.check_sha256_cache(clean)
                    header = ["SHA256 Cache", "Count"]
                    al_p: list[bool|None]|None = [True, False]
                    rows_p: list[list[str]] = []
                    rows_p.append(["Cache entries", f"{entries}"])
                    rows_p.append(["Orphans", f"{debris}"])
                    if deleted > 0:
                        rows_p.append(["Debris removed", f"{deleted}"])
                    _ = tf.print_table(header, rows_p, al_p)
                    if ds.sha256_cache_changed is True:
                        ds.save_sha256_cache()
                        print("SHA256 cache file updated")
                    print()

                if len(arguments) == 0 or 'index' in arguments or (len(arguments)==1 and 'clean' in arguments):
                    doc_hashes: list[str] = list(ds.text_library.keys())

                    if 'clean' in arguments:
                        clean = True
                    else:
                        clean = False

                    def feedback(ps:ProgressState):
                        compl = tf.progress_bar_string(ps['percent_completion'], 8)
                        print(f"Checking indices {compl} {ps['state']} ", end="\r", flush=True)
                        
                    model_check = vs.check_indices(doc_hashes, clean, feedback)
                    print("")

                    hint_missing: bool = False
                    hint_clean: bool = False
                    c_rows: list[list[str]] = []
                    c_selected: int|None = None
                    for ind, ms in enumerate(model_check):
                        if ms['enabled'] is True:
                            if ms['missing_count'] > 0:
                                hint_missing = True
                            if ms['debris_count'] > 0:
                                hint_clean = True
                            if ms['selected'] is True:
                                c_rows.append([f">{ind+1}<", str(ms['document_count']), str(ms['embedding_count']), str(ms['embedding_dim']), str(ms['debris_count']), str(ms['deleted_count']), str(ms['missing_count']), ms['model_name']])
                                c_selected = ind
                            else:
                                c_rows.append([f" {ind+1} ", str(ms['document_count']), str(ms['embedding_count']), str(ms['embedding_dim']), str(ms['debris_count']), str(ms['deleted_count']), str(ms['missing_count']), ms['model_name']])
                        else:
                            c_rows.append([f">{ind+1}<", "DISABLED", "", "", "", ms['model_name']])                    
                    header = ["ID", "Documents", "Embeddings", "Dim", "Debris", "Deleted", "Missing", "Model"]
                    alignment: list[bool|None]|None = [True, False, False, False, False, False, False, True]
                    _ = tf.print_table(header, c_rows, alignment, selected=c_selected)
                    print()
                    if hint_missing is True:
                        log.info("To calculate the missing indices, use 'select <ID>' and 'index' for each model with missing indices")
                    if hint_clean is True:
                        log.info("Use 'check index clean' to clean up debris indices")
                    if hint_missing is False and hint_clean is False:
                        log.info("All model indices are fully up-to-date")                
            elif command == 'list':
                if 'models' in arguments or len(arguments) == 0:
                    header = ["ID", "Docs", "Model"]
                    m_rows: list[list[str]] = []
                    selected: int|None = None
                    for ind, model in enumerate(vs.model_list):
                        path = vs.model_embedding_path(model['model_name'])
                        cnt = len(get_files_of_extensions(path, ['pt']))
                        if model['enabled'] == False:
                            post = ' DISABLED'
                        else:
                            post = ''
                        if model['model_name'] == vs.config['embeddings_model_name']:
                            m_rows.append([f">{ind+1}<", str(cnt), f"{model['model_name']}{post}"])
                            selected = ind
                        else:
                            m_rows.append([f" {ind+1}", str(cnt), f"{model['model_name']}{post}"])
                    _ = tf.print_table(header, m_rows, [True, False, True], selected=selected)
                    print()
                    print("Use 'select <ID>' to change the active model, 'enable|disable <ID>' to activate/deactivate")

                if 'perf' in arguments or len(arguments) == 0:
                    print()
                    header = ["Metric", "Duration (s)"]
                    p_rows: list[list[str]] = []
                    for key in vs.perf.keys():
                        p_rows.append([key, f"{vs.perf[key]:2.3f}"])
                    for key in ds.perf.keys():
                        p_rows.append([key, f"{ds.perf[key]:2.3f}"])
                    _ = tf.print_table(header, p_rows, [True, False])
                    if len(p_rows)==0:
                        print("No performance data generated (yet)")
                    print()
                        
                if 'sources' in arguments or len(arguments) == 0:
                    sum_ext_cnts, sources_ext_cnts = ds.get_sources_ext_cnts()
                    exts = list(sum_ext_cnts.keys())
                    sum_cnt = 0
                    for ext in sum_ext_cnts:
                        sum_cnt += sum_ext_cnts[ext]
                    s_rows: list[list[str]] = []
                    for source_name in sources_ext_cnts:
                        ext_cnts = sources_ext_cnts[source_name]
                        cnt = 0
                        for ext in ext_cnts:
                            cnt += ext_cnts[ext]
                        row = [f"{source_name}", str(cnt)]
                        for ext in exts:
                            if ext in ext_cnts:
                                row.append(str(ext_cnts[ext]))
                            else:
                                row.append("0")
                        row.append(ds.config['document_sources'][source_name]['path'])
                        s_rows.append(row)

                    header = ["Source", "Docs"] + exts + ["Path"]
                    al_s: list[bool|None]|None = [True, False]
                    al_s += [False] * len(exts) + [True]

                    row = ["Total", str(sum_cnt)]
                    for ext in exts:
                        if ext in sum_ext_cnts:
                            row.append(str(sum_ext_cnts[ext]))
                        else:
                            row.append("0")
                    row.append("")
                    s_rows.append(row)
                    _ = tf.print_table(header, s_rows, al_s)
                    print()

                if 'timelines' in arguments or len(arguments) == 0:
                    tl_cnt = len(ds.tl.tl_events)
                    print()
                    header = ['Timeline data', 'Value']
                    al_t:list[bool|None]|None = [True, False]
                    rows_t: list[list[str]] = []
                    rows_t.append(['Table count', str(len(ds.tables))])
                    rows_t.append(['Event count', str(tl_cnt)])
                    _ = tf.print_table(header, rows_t, al_t)
                    print()

                if 'vars' in arguments or len(arguments) == 0:
                    print()
                    header = ['Variable', 'Value', 'Type']
                    al:list[bool|None]|None = [True, True, False]
                    rows_v: list[list[str]] = []
                    for name in ds.config['vars']:
                        val, type = ds.config['vars'][name]
                        rows_v.append([name, val, type])
                    _ = tf.print_table(header, rows_v, al)
                    print()

            elif command == 'set':
                comps = arguments
                if len(comps) != 2:
                    log.error('Usage: set <name> <value>')
                    log.info("Use `list vars` for a list of known variable names and types")
                    log.info("Additional:  set device auto|cpu|cuda|mps|xpu")
                else:
                    if comps[0] == 'device':
                        vs.set_device(comps[1])
                    else:
                        _ = ds.set_var(comps[0], comps[1])
            elif command == 'select':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind != -1:
                    new_model = vs.select(ind)
                    if new_model is None:
                        log.info("Model unchanged.")
                    else:
                        log.info(f"New active model is {new_model}")
                else:
                    log.error(f"Invalid ID {arguments}, integer required, use 'list models' for valid range")
            elif command == 'enable':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind != -1:
                    new_model = vs.enable(ind)
                else:
                    log.error(f"Invalid ID {arguments}, integer required, use 'list models' for valid range")
            elif command == 'disable':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind != -1:
                    new_model = vs.disable(ind)
                else:
                    log.error(f"Invalid ID {arguments}, integer required, use 'list models' for valid range")
            elif command == 'index':
                if ds.local_update_required() is True:
                    if 'force' not in arguments:
                        log.warning("Remote data is newer than local data! Please use import first! (or override with 'force'")
                        continue
                    else:
                        log.warning("Version override, starting indexing")

                def progress_index(ps:ProgressState):
                    cols, _= os.get_terminal_size()
                    blnk = "\r" + ' ' * (cols - 1)
                    print(blnk, end="")
                    progress = tf.progress_bar_string(ps['percent_completion'], 8)
                    print(f"\r{progress} {ps['state']}", end="", flush=True)
                    if ps['finished'] is True:
                        print()
                    
                if 'all' in arguments:
                    errors = vs.index_all(ds.text_library, progress_index)
                else:
                    errors = vs.index(ds.text_library, progress_index)
                print()
                if len(errors) > 0:
                    print("Errors and issues:")
                    for error in errors:
                        print(error)
                        
            elif command == 'index3d':
                if ds.local_update_required() is True:
                    if 'force' not in arguments:
                        log.warning("Remote data is newer than local data! Please use import first! (or override with 'force'")
                        continue
                    else:
                        log.warning("Version override, starting 3D-indexing")
                if 'all' in arguments:
                    vs.index3d_all(ds.text_library)
                else:
                    vs.index3d(ds.text_library, None)                
            elif command == 'search':
                search_results: int = cast(int, ds.get_var('search_results', key_vals))
                highlight: bool = cast(bool, ds.get_var('highlight', key_vals))
                cutoff = cast(float, ds.get_var('highlight_cutoff', key_vals))
                damp:float = cast(float, ds.get_var('highlight_dampening', key_vals))
                context_length:int = cast(int, ds.get_var('context_length', key_vals))
                context_steps:int = cast(int, ds.get_var('context_steps', key_vals))
                count = search_results
                search_string = ' '.join(arguments)
                print(f"Searching: {search_string}")
                if 'max_results' in key_vals:
                    try:
                        count = int(key_vals['max_results'])
                    except ValueError:
                        log.error(f"Invalid integer max_results={key_vals['max_results']}, keeping default {count}")

                def search_state(ps:ProgressState):
                    compl = tf.progress_bar_string(ps['percent_completion'], 8)
                    state = f"{ps['state'][:80]:80s}"
                    print(f"{compl} {state}", end="\r", flush=True)
                search_result_list = vs.search(search_string, ds.text_library, count,
                                               highlight, cutoff, damp,
                                               context_length, context_steps, search_state)
                previous_search_results = search_result_list
                keywords = tp.parse(search_string)
                if keywords is None:
                    keywords = []
                
                for index, result in enumerate(search_result_list):
                    header = [f"{result['cosine']:.3f}", result['entry']['descriptor']]
                    text = result['text']
                    if text is None:
                        text = ""
                    rows: list[list[str]] = [[str(len(search_result_list) - index), text]]
                    print()
                    _ = tf.print_table(header, rows, multi_line=True, keywords=keywords, significance=[[None, result['significance']]])
                print()
            elif command == 'ksearch':
                search_string = ' '.join(arguments)
                source = key_vals.get('source')
                print(f"Keyword Searching: {search_string} (Source: {source if source else 'All'})")
                search_result_list = ds.keyword_search(search_string, source=source)
                previous_search_results = search_result_list
                
                keywords = tp.parse(search_string)
                if keywords is None:
                    keywords = []
                else:
                    # Extract highlight terms using SearchTools to handle | and wildcards
                    keywords = SearchTools.extract_highlight_terms(keywords)

                for index, result in reversed(list(enumerate(search_result_list))):
                    header = [f"{result['cosine']:.1f}", result['entry']['descriptor']]
                    text = result['text']
                    if text is None:
                        text = ""
                    rows_k: list[list[str]] = [[str(len(search_result_list) - index), text]]
                    print()
                    _ = tf.print_table(header, rows_k, multi_line=True, keywords=keywords)
                print()
            elif command == 'show':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind < 1 or ind > len(previous_search_results):
                    log.error(f"Invalid ID {arguments}, integer required, use 'search' or 'ksearch' first")
                else:
                    result = previous_search_results[len(previous_search_results) - ind]
                    descriptor = result['entry']['descriptor']
                    metadata = ds.get_metadata(descriptor)
                    if metadata is not None:
                        print(f"\nMetadata for {descriptor}:")
                        for key in metadata:
                            print(f"{key}: {metadata[key][:50]}")
                        print()
                    else:
                        log.error(f"Metadata not found for {descriptor}")
            elif command == 'open':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind < 1 or ind > len(previous_search_results):
                    log.error(f"Invalid ID {arguments}, integer required, use 'search' or 'ksearch' first")
                else:
                    result = previous_search_results[len(previous_search_results) - ind]
                    descriptor = result['entry']['descriptor']
                    path = ds.get_path_from_descriptor(descriptor)
                    if os.path.exists(path):
                        log.info(f"Opening {path}...")
                        try:
                            if sys.platform == 'darwin':
                                _ = subprocess.run(['open', path], check=True)
                            else:
                                _ = subprocess.run(['xdg-open', path], check=True)
                        except Exception as e:
                            log.error(f"Failed to open file: {e}")
                    else:
                        log.error(f"File not found: {path}")

            elif command == 'audiobook':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind < 1 or ind > len(previous_search_results):
                    log.error(f"Invalid ID {arguments}, integer required, use 'search' or 'ksearch' first")
                else:
                    result = previous_search_results[len(previous_search_results) - ind]
                    descriptor = result['entry']['descriptor']
                    
                    # Get text content
                    text_content = result['entry']['text']
                    if not text_content:
                        log.error("No text content found for this document")
                        continue

                    # Get language from metadata
                    language = 'en' # Default
                    metadata = ds.get_metadata(descriptor)
                    if metadata and metadata['languages']:
                        # Use the first language found
                        langs = metadata['languages']
                        language = langs[0]
                        
                    # Generate output filename
                    # Use title if available, otherwise filename
                    title = "audiobook"
                    if metadata and metadata['title']:
                        title = metadata['title']
                    else:
                        title = os.path.splitext(os.path.basename(descriptor))[0]
                    
                    # Sanitize filename
                    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ' or c=='-']).rstrip()
                    output_path = os.path.join(os.getcwd(), f"{safe_title}.mp3")
                    
                    log.info(f"Generating audiobook for '{title}' in '{language}'...")
                    start_token = key_vals.get('start')
                    end_token = key_vals.get('end')
                    
                    # Extract metadata
                    authors = metadata['authors'] if metadata else []
                    author = ", ".join(authors)
                    icon_data = metadata['icon'] if metadata else None

                    success = audiobook_gen.generate_audiobook(
                        text=text_content, 
                        language=language, 
                        output_path=output_path,
                        start_token=start_token,
                        end_token=end_token,
                        title=title,
                        author=author,
                        icon_data=icon_data
                    )
                    if success:
                        print(f"Audiobook generated: {output_path}")
                    else:
                         print("Failed to generate audiobook. Check logs for details.")

            elif command == 'timeline_extract':
                ind = -1
                try:
                    ind = int(arguments[0])
                except:
                    pass
                if ind < 1 or ind > len(previous_search_results):
                    log.error(f"Invalid ID {arguments}, integer required, use 'search' or 'ksearch' first")
                else:
                    result = previous_search_results[len(previous_search_results) - ind]
                    descriptor = result['entry']['descriptor']
                    text_content = result['entry']['text']
                    
                    if not text_content:
                        log.error("No text content found")
                        continue
                        
                    model_name = key_vals.get('model', "google/gemma-7b-it")
                    log.info(f"Extracting timeline from '{descriptor}' using {model_name}...")
                    
                    try:
                        # Lazy init to avoid overhead if not used
                        extractor = TimelineExtractor(model_name=model_name)
                        events = extractor.extract_from_text(text_content[:12000])
                        
                        if not events:
                            print("No events found.")
                        else:
                            # Sort events by date
                            def event_sorter(ev):
                                try:
                                    jd = IndraTime.string_time_to_julian(ev['indra_str'])
                                    if jd:
                                        if len(jd) == 1:
                                            return jd[0]
                                        return jd[0]
                                except:
                                    pass
                                return -999999999.0 # Sort undefined/errors to start? or end?

                            try:
                                events.sort(key=event_sorter)
                            except Exception as e:
                                log.warning(f"Sorting failed: {e}")

                            header = ["Date", "Description"]
                            rows = []
                            for ev in events:
                                rows.append([ev['indra_str'], ev['event_description']])
                            
                            print(f"\nTimeline for {descriptor}:")
                            _ = tf.print_table(header, rows, multi_line=True)
                            print()
                            
                    except Exception as e:
                        log.error(f"Timeline extraction failed: {e}")
                        import traceback
                        traceback.print_exc() # Print full stack for user debugging if it crashes again

            elif command == 'open':
                if len(previous_search_results) == 0:
                    print("No previous search results available")
                else:
                    for index, result in enumerate(previous_search_results):
                        print(f"Id: {len(previous_search_results)-index}. {result['entry']['descriptor']}")
                        print(f"Score: {result['cosine']:3.3f}")
                        print(result['text'])
                        print()                                     
            elif command == "timeline":
                domains = key_vals.get('domains', None)
                if domains is not None:
                    domains = domains.split(' ')
                keywords = key_vals.get('keywords', None)
                if keywords is not None:
                    keywords = keywords.split(' ')
                time = key_vals.get('time', None)
                partial_overlap = key_vals.get('partial_overlap', "false")
                if partial_overlap.lower() == 'true':
                    partial = True
                else:
                    partial = False
                full_overlap = key_vals.get('full_overlap', "false")
                if full_overlap.lower() == 'true':
                    full = True
                else:
                    full = False
                
                tlel = ds.tl.search_events(time, domains, keywords, True, full, partial)
                header = ['Date', 'Event']
                rows = []
                for tle in tlel:
                    date = ds.tl.get_date_string_from_event(tle['jd_event'])
                    event = ds.tl.get_event_text(tle['eventdata'])
                    if date is not None:
                        rows.append([date, event])
                print()
                if keywords is None:
                    hl_keywords: list[str] = []
                else:
                    hl_keywords = SearchTools.extract_highlight_terms(keywords)
                _ = tf.print_table(header, rows, multi_line=True, keywords=hl_keywords)
               
            elif command == 'publish':
                _ = ds.publish(arguments)
            elif command == 'import':
                if ds.import_local(arguments) is True:
                    log.info("Import successful, reloading data...")
                    del ds
                    del vs
                    ds = DocumentStore()
                    vs = VectorStore(ds.storage_path, ds.config_path)
                else:
                    log.error("Import failed")
            elif command == 'force_load_docs':
                ds.load_document_data()
            elif command == 'help':
                header = ['Command', 'Options', 'Function']
                rows = []
                rows += [['list', '[models|sources|vars|perf]', 'List internal tables'],
                         ['sync', '[force] [retry]', 'Sync data sources, check for changed documents'],
                         ['check', '[index|pdf|sha256] [clean]', 'Verify datastructures'],
                         ['select', '<model-ID>', 'Select model <id> (1..n) as active model for search. `list models` shows IDs and currently active model'],
                         ['index',  '[force] [all]', 'Generate vector database indices for new or changed documents'],
                         ['search', '<search-string>', 'Do a vector search with currently active model'],
                         ['ksearch', '<search-string> [source=<source>]', 'Do a keyword search on metadata (supports | ! *)'],
                         ['text', '', 'Print previous result of `search` without formatting for copying'],
                         ['show', '<ID>', 'Show metadata for a search result'],
                         ['open', '<ID>', 'Open the document for a search result'],
                         ['audiobook', '<ID> [start="<token>"] [end="<token>"]', 'Generate an audiobook from the document text'],
                         ['timeline_extract', '<ID> [model="<model>"]', 'Extract timeline events from text using LLM'],
                         ['timeline', '[time=1999-01-01[-2100-01-01]] [domains="dom1[ dom2]"] [keywords="key1[ key2]"]', 'Compile a table of events'],                         
                         ['publish', '[force]', 'Publish newly created indices'],
                         ['import', '[force]', 'Import indices created remotely'],
                         ['force_load_docs', '', 'Load local document database, even if outdated compared to remote'],
                         ['set', '<var-name> <value>', 'Set a variable to a value. `list variables` shows available vars'],
                         ['exit', '', 'Exit (alterative `quite` or Ctrl-D'],
                         ]
                _ = tf.print_table(header, rows, multi_line=True)
                
            elif command == 'exit' or command == 'quit':
                break

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D to exit gracefully
            break
    

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("ResearchCLI")
    log.info("Local Research v1.0")

    # check of '--no-load' flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_load_libraries', action='store_false', default=True, help='Do not load document library')
    args = parser.parse_args()

    ds = DocumentStore(load_libraries=args.no_load_libraries)
    vs = VectorStore(ds.storage_path, ds.config_path)
    repl(ds, vs, log)
    
if __name__ == "__main__":
    main()
