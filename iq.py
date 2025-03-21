import logging
import os
import argparse
from argparse import ArgumentParser
from rich.console import Console
from icotq_store import IcoTqStore, SearchResult
from typing import cast

def iq_info(its: IcoTqStore, _logger:logging.Logger) -> None:
    its.list_sources()

def iq_index(its: IcoTqStore, _logger:logging.Logger, param:str):
    if param == 'purge':
        purge = True
    else:
        purge = False
    its.generate_embeddings(purge=purge)

def iq_search(its: IcoTqStore, logger:logging.Logger, search_spec: str):
    max_results = 3
    context_length = 16
    context_steps = 4
    yellow = True
    cols, _ = os.get_terminal_size()
    results: list[SearchResult] | None = its.search(search_text=search_spec, yellow_liner=yellow, context_length=context_length, context_steps=context_steps, max_results=max_results, compression_mode="full")
    print()
    print()
    console = Console()
    if len(results) > 0:
        for i in range(len(results)):
            result = results[i]
            y_min: float | None = None
            y_max: float | None = None
            ryel = result['yellow_liner']
            # print(ryel)
            yels: list[float] = []
            if ryel is not None:
                if len(ryel.shape) == 1:
                    lyel = ryel.tolist()
                    yels = cast(list[float], lyel)
                    for y in yels:
                            if y_min is None or y<y_min:
                                y_min = y
                            if y_max is None or y>y_max:
                                y_max = y
                else:
                    logger.error(f"Yellow-liner result has wrong shape: {ryel.shape}")
                    continue
            else:
                print(results[i]['chunk'])
            if y_min == None:
                    y_min = 0
            if y_max == None:
                    y_max = 1
            ind = result['desc'].rfind('/')
            if ind!=-1:
                short_desc = result['desc'][ind+1:]
            else:
                short_desc = result['desc']
            title_text = f"Document: {short_desc}, {result['cosine'] * 100.0:2.1f} %"
            if len(title_text) < cols: 
                title_text += " " * (cols - len(title_text))
            else:
                title_text = title_text[:cols]
            console.print("[#FFFFFF on #D0D0D0]"+"-"*cols+"[/]")
            console.print("[black on #E0E0E0]"+title_text+"[/]")
            console.print("[#FFFFFF on #E0E0E0]"+"-"*cols+"[/]")
            # sys.stdout.flush()
            # print(best_chunk)
            # print(y_min, y_max)
            if y_min == y_max:
                    print(f"Search gave no meaningful result: y_min: {y_min}, y_max: {y_max}, search-embedding vector is trivial (language not supported?)")
                    print(result['chunk'])
                    continue
            if yels != []:
                line = ""
                char_ind = 0
                for i, c in enumerate(result['chunk']):
                    y_ind = i//context_steps
                    if y_ind >= len(yels):
                        print(f"Index out of range: {y_ind}, len(yels): {len(yels)}")
                    else:
                        yel:float = (yels[y_ind]-y_min)/(y_max - y_min)
                        if yel < 0.5:
                            yel = 0.0
                        col = hex(255 - int(yel*127.0))[2:]
                        if c == "\n":
                            rest = "[black on #FFFFFF]" + " " * (cols - char_ind%cols) + "[/]"
                            line += rest
                            char_ind = 0
                        else:
                            line += f"[black on #FFFF{col}]"+c+"[/]"
                            if ord(c)>31:
                                char_ind += 1
                if char_ind > 0:
                    rest = "[black on #FFFFFF]" + "-" * (cols - char_ind%cols) + "[/]"
                    line += rest
                    char_ind = 0
                console.print(line) # , soft_wrap=True)
        console.print("[#FFFFFF on #E0E0E0]"+"-"*cols+"[/]")
    else:
        print("No search result available!")
    
def iq_export(its: IcoTqStore, logger:logging.Logger) -> None:
    if 'ebook_mirror' not in its.config:
        logger.error(f"Cannot export, destination 'ebook_mirror' not defined in config")
        return
    ebook_mirror_path = os.path.expanduser(its.config['ebook_mirror'])
    if os.path.isdir(ebook_mirror_path) is False:
        logger.error(f"Destination directory {ebook_mirror_path} does not exist, aborting export!")
        return
    print(f"Export to {ebook_mirror_path}")

def iq_sync(its: IcoTqStore, _logger:logging.Logger, max_imports_str:str|None=None):
    if max_imports_str is not None:
        try:
            max_imports = int(max_imports_str)
        except ValueError:
            max_imports = None
    else:
        max_imports = None
    its.sync_texts(max_imports=max_imports)

def iq_select(its: IcoTqStore, _logger:logging.Logger, model_id:str):
    try:
        ind = int(model_id)
        if ind > 0 and ind <= len(its.model_list):
            model_id = its.model_list[ind-1]['model_name']
    except ValueError:
        pass
    _ = its.load_model(model_id, its.config["embeddings_device"], its.config["embeddings_model_trust_code"])
    _ = its.load_tensor()

def iq_list(its: IcoTqStore, _logger:logging.Logger, param:str):
    if param == 'models':
        for ind, model in enumerate(its.model_list):
            if model['model_name'] == its.config["embeddings_model_name"]:
                sel:str = "[*]"
            else:
                sel = "   "
            print(f"{ind+1}: {sel} {model['model_name']}")
    elif param == 'sources':
        for ind, source in enumerate(its.config["tq_sources"]):
            cnt = 0
            for entry in its.lib:
                if entry['source_name'] == source['name']:
                    cnt += 1
            print(f"{ind+1}: {source['name']} at {source['path']} ({source['tqtype']}), {cnt} docs")
    elif param == 'docs':
        for ind, entry in enumerate(its.lib):
            print(f"{ind+1} {entry['desc_filename']}")
    else:
        print("Usage either 'list models', 'list sources', or 'list docs'.")

def iq_check(its: IcoTqStore, _logger:logging.Logger, param:str=""):
    if param == "" or "pdf" in param:
        its.check_clean(dry_run=True)

def iq_clean(its: IcoTqStore, _logger:logging.Logger, param:str=""):
    if param == "" or "pdf" in param:
        its.check_clean(dry_run=False)

def iq_serve(its: IcoTqStore, _logger:logging.Logger, param:str):
    background:bool = False
    if 'background' in param:
        background = True

    if param == "stop":
        its.stop_server()
    else:
        its.start_server(background=background)

def iq_help(parser:argparse.ArgumentParser, valid_actions:list[tuple[str, str]]):
    parser.print_help()
    print()
    print("Command can either be provided as command-line arguments or at the '> ' prompt.")
    print("Valid commands are: \n" + '\n    '.join([f"{command}: {help}" for command, help in valid_actions]))
    print("To exit, simply press Enter at the command prompt, or by 'exit' or 'quit'")

def parse_cmd(its: IcoTqStore, logger: logging.Logger) -> None:
    valid_actions = [('info', 'Overview of available data and sources'), 
                                            ('sync', "[max_docs] evaluate available sources and cache text information and metadata, optional max_docs limits number of imported docs, sync source repos with cached text for indexing. Use 'index' function afterwards to create the actual index!"), 
                                            ('index', "[purge] Generate embeddings index for currently active model. Option purge starts index from scratch. ('list models', 'select <model-id>' to change current model)"),
                                            ('list', "models|sources|docs"),
                                            ('select', "model-index as shown by: 'list models', use 'index' to create or update embeddings indices"),
                                            ('search', "Search for keywords given as repl argument or with '-k <keywords>' option. You need to 'sync' and 'index' first"),
                                            ('check', "Verify consistency of data references and indices. Use 'clean' to apply actions."),
                                            ('clean', "Repair consistency of data references and indices. Remove debris. Use 'check' first for dry-run."),
                                            ('serve', "Start web-server for search, 'serve stop' to stop server, 'serve background' to run server in background to be able to continue to use the console interactively."),
                                            ('help', 'Display usage information')]
    parser: ArgumentParser = argparse.ArgumentParser(description="IcoTq")
    _ = parser.add_argument(
        "action",
        nargs="*",
        default="",
        help="Actions: " + ','.join([f"'{command}': {help}" for command, help in valid_actions]),
    )
    _ = parser.add_argument(
            "-n",
            "--non-interactive",
            action="store_true",
            help="Non-interactive mode, do not enter repl",
        )
    _ = parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        default="",
        help="Restrict search to list of space separated keywords, leading '!' used for exclusion (negation)," +\
        " '*' for wildcards at beginning, middle or end of keywords." +\
        " Multiple space separated keywords are combined with AND, use '|' for OR combinations." +\
        "! Also to add parameters for 'import' (max_docs) and 'list' (models|sources|docs) commands",
    )

    args = parser.parse_args()
    quit:bool = False
    first:bool = True
    param = cast(str, args.keywords)
    actions: list[str] =  cast(list[str], args.action)
    while quit is False:
        for action in actions:
            if action not in [cmd for cmd, _ in valid_actions]:
                logger.error(f"Invalid action {action}, valid are: {valid_actions}, 'help' for more information")
        if 'info' in actions:
            iq_info(its, logger)
        if 'sync' in actions:
            iq_sync(its,  logger, max_imports_str=param)
        if 'help' in actions:
            iq_help(parser, valid_actions)
        if 'index' in actions:
            iq_index(its, logger, param)
        if 'search' in actions:
            iq_search(its, logger, param)
        if 'list' in actions:
            iq_list(its, logger, param)
        if 'select' in actions:
            iq_select(its, logger, param)
        if 'check' in actions:
            iq_check(its, logger, param)
        if 'clean' in actions:
            iq_clean(its, logger, param)
        if 'serve' in actions:
            iq_serve(its, logger, param)
        if cast(bool, args.non_interactive) is True:
            break
        if first is True:
            print("Enter 'help' for command summary.")
            first = False
        try:
            cmd = input("> ")
        except (EOFError, KeyboardInterrupt):
            quit = True
            continue
        # print(f"{len(cmd)}: >{cmd}<")
        cmd_inp = cmd.strip()
        if cmd_inp == 'quit' or cmd_inp == 'exit': 
            quit = True
        else:
            ind = cmd_inp.find(' ')
            if ind != -1:
                actions = [cmd_inp[:ind].strip()]
                param = cmd_inp[ind:].strip()
            else:
                actions = [cmd_inp]
                param = ""
    print()

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IQ")
    logger.info("Starting...")
     
    its = IcoTqStore()
    parse_cmd(its, logger)

if __name__ == "__main__":
    main()
