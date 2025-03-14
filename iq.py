import logging
import os
import argparse
from argparse import ArgumentParser
from rich.console import Console
from icotq_store import IcoTqStore, SearchResult
from typing import cast

def iq_info(its: IcoTqStore, _logger:logging.Logger) -> None:
    its.list_sources()

def iq_embed(its: IcoTqStore, logger:logging.Logger):
    its.generate_embeddings()
    logger.info("Embeddings generated.")

def iq_search(its: IcoTqStore, search_spec: str, logger:logging.Logger):
    max_results = 3
    context = 16
    context_steps = 4
    yellow = True
    cols, _ = os.get_terminal_size()
    results: list[SearchResult] | None = its.search(search_text=search_spec, yellow_liner=yellow, context=context, context_steps=context_steps, max_results=max_results, compress="full")
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
            title_text = f"Document: {result['desc']}[{result['index']}], {result['cosine'] * 100.0:2.1f} %"
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

def iq_import(its: IcoTqStore, _logger:logging.Logger):
    its.import_texts()

def iq_help(parser:argparse.ArgumentParser, valid_actions:list[tuple[str, str]]):
    parser.print_help()
    print()
    print("Command can either be provided as command-line arguments or at the '> ' prompt.")
    print("Valid commands are: \n" + '\n    '.join([f"{command}: {help}" for command, help in valid_actions]))
    print("To exit, simply press Enter at the command prompt, or by 'exit' or 'quit'")

def parse_cmd(its: IcoTqStore, logger: logging.Logger) -> None:
    valid_actions = [('info', 'Overview of available data and sources'), 
                                            ('export', 'NOT IMPLEMENTED'), 
                                            ('import', 'evaluate available sources and cache text information and metadata'), 
                                            ('embed', 'Generate embeddings for currently active model'),
                                            ('search', 'Search for keywords given with -k <keywords> option'),
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
        " Multiple space separated keywords are combined with AND, use '|' for OR combinations.",
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
        if 'export' in actions:
            iq_export(its, logger)
        if 'import' in actions:
            iq_import(its, logger)
        if 'help' in actions:
            iq_help(parser, valid_actions)
        if 'embed' in actions:
            iq_embed(its, logger)
        if 'search' in actions:
            iq_search(its, param, logger)
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
        if cmd_inp == "" or cmd_inp == 'quit' or cmd_inp == 'exit': 
            quit = True
        else:
            ind = cmd_inp.find(' ')
            if ind != -1:
                actions = [cmd_inp[:ind].strip()]
                param = cmd_inp[ind:].strip()
            else:
                actions = [cmd_inp]
    print()

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IQ")
    logger.info("Starting...")
     
    its = IcoTqStore()
    parse_cmd(its, logger)

if __name__ == "__main__":
    main()