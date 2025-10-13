import logging
import os
import subprocess
from datetime import datetime

class ResearchTools:
    def __init__(self):
        self.log = logging.Logger("ResearchTools")
    def _note_get_file_creation_date_from_git(self, root_folder:str, filepath: str) -> datetime | None:
        try:
            creation_date = subprocess.check_output(
                args = [
                    "git",
                    "-C",
                    root_folder,
                    "--no-pager",
                    "log",
                    "--follow",
                    "--format=%aI",
                    "--reverse",
                    filepath,
                ]
            )
            cr_date = creation_date.decode("utf-8").split("\n")[0]
            # datetime parse:
            try:
                dt = datetime.strptime(cr_date, "%Y-%m-%dT%H:%M:%S%z")
                return dt
            except Exception as e:
                self.log.debug(
                    f"Error file {filepath}, failed parsing date {cr_date}: {e}"
                )
                return None
        except Exception as _:
            return None

    def _note_get_file_modification_date(self, filepath: str) -> datetime | None:
        try:
            stat = os.stat(filepath)
            dt = datetime.fromtimestamp(stat.st_mtime)
            return dt
        except Exception as e:
            self.log.warning(f"Error getting file modification date {filepath}: {e}")
            return None

    def get_note_creation_date(self, root_folder:str|None, filepath: str) -> datetime:
        if root_folder is not None:
            dt_git: datetime | None = self._note_get_file_creation_date_from_git(root_folder, filepath)
        else:
            dt_git = None
        dt_stat = self._note_get_file_modification_date(filepath)
        # self.log.info(f"Creation date for {filepath} from git: {dt_git}, from stat: {dt_stat}")
        if dt_git is not None:
            return dt_git
        if dt_stat is not None:
            return dt_stat
        return datetime.now()
