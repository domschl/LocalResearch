import logging
import os
import shutil
from typing import TypedDict


class FileMetadata(TypedDict):
    path: str
    date_changed: float
    size: int
    

class SyncTarget(TypedDict):
    types: list[str]
    path: str
    file_types: list[str]
    metadata: dict[str, FileMetadata] | None


class SyncTools:
    def __init__(self, target:SyncTarget):
        self.log:logging.Logger = logging.Logger("SyncTool")
        self.target:SyncTarget = target
        
    def file_meta(self, filepath:str) -> tuple[float, int] | None:
        try:
            stat:os.stat_result = os.stat(filepath)
            return  stat.st_mtime, stat.st_size
        except Exception as e:
            self.log.warning(f"Error getting file modification date {filepath}: {e}")
        return None
        
    def update_target(self):
        root_path = os.path.expanduser(self.target['path'])
        self.target['metadata'] = self.get_files_and_metadata(root_path)

    def get_files_and_metadata(self, root_path:str) -> dict[str, FileMetadata]:
        files_and_meta: dict[str, FileMetadata] = {}
        errors: list[str] = []
        for root, _dirs, files in os.walk(root_path, topdown=True, onerror=lambda e: errors.append(f"Cannot access directory {e.filename}: {e.strerror}")):  # pyright:ignore[reportAny]
            if '.caltrash' in root:
                continue
            for filename in files:
                base, ext = os.path.splitext(filename)
                if base.startswith('.'):
                    continue
                if ext.startswith('.'):
                    ext = ext[1:]
                if ext not in self.target['file_types']:
                    continue
                abs_path = os.path.join(root, filename)
                rel_path = abs_path[root_path:]
                if rel_path.startswith('/'):
                    rel_path = rel_path[1:]
                meta = self.file_meta(abs_path)
                if meta is None:
                    continue
                md = FileMetadata({'path':abs_path, 'date_changed': meta[0], 'size': meta[1]})
                files_and_meta[rel_path] = md
        for error in errors:
            self.log.error(error)
        return files_and_meta

    def vacuum(self, dry_run:bool = True) -> int:
        root_path = os.path.expanduser(self.target['path'])
        errors: list[str] = []
        delete_cnt = 0
        for root, _dirs, files in os.walk(root_path, topdown=True, onerror=lambda e: errors.append(f"Cannot access directory {e.filename}: {e.strerror}")):  # pyright:ignore[reportAny]
            if '.caltrash' in root:
                continue
            for filename in files:
                base, ext = os.path.splitext(filename)
                if base.startswith('.'):
                    continue
                if ext.startswith('.'):
                    ext = ext[1:]
                if ext not in self.target['file_types']:
                    abs_path = os.path.join(root, filename)
                    if dry_run is False:
                        os.remove(abs_path)
                        self.log.info(f"Vacuum: deleted {abs_path}")
                    else:
                        self.log.info(f"Vacuum would delete debris-file {abs_path}")
                    delete_cnt += 1
        for error in errors:
            self.log.error(error)
        return delete_cnt

    def sync_to_target(self, source_path:str, rel_path_translation_table: dict[str, str]|None = None,  vacuum_target:bool = False, dry_run:bool = True) -> tuple[int, int, int]:
        source_root = os.path.expanduser(source_path)
        target_root = os.path.expanduser(self.target['path'])
        new_cnt: int = 0
        changed_cnt: int = 0
        delete_cnt: int = 0
        if os.path.isdir(source_root) is False:
            self.log.error(f"{source_path} is not a valid directory")
            return new_cnt, changed_cnt, delete_cnt
        source_files = self.get_files_and_metadata(source_root)
        target_files = self.get_files_and_metadata(target_root)
        for source_file in source_files:
            if rel_path_translation_table is not None:
                if source_file not in rel_path_translation_table:
                    self.log.error(f"Cannot translate {source_file} via translation table, ignoring!")
                    continue
                translated_file = rel_path_translation_table[source_file]
            else:
                translated_file = source_file
            if translated_file not in target_files:
                target_path = os.path.join(target_root, translated_file)
                if dry_run is False:
                    _ = shutil.copy2(source_files[source_file]['path'], target_path)
                    self.log.info(f"Copied new file {source_files[source_file]['path']} -> {target_path}")
                else:
                    self.log.info(f"Would copy new file {source_files[source_file]['path']} -> {target_path}")
                new_cnt += 1
            else:
                changed:bool = False
                if source_files[source_file]['size'] != target_files[translated_file]['size']:
                    self.log.info(f"Size of {source_file} has changed")
                    changed = True
                if source_files[source_file]['date_changed'] > target_files[translated_file]['date_changed']:
                    self.log.info(f"Newer version of {source_file} available")
                    changed = True
                if changed is True:
                    target_path = os.path.join(target_root, translated_file)
                    if dry_run is False:
                        _ = shutil.copy2(source_files[source_file]['path'], target_path)
                        self.log.info(f"Copied changed file {source_files[source_file]['path']} -> {target_path}")
                    else:
                        self.log.info(f"Would copy changed file {source_files[source_file]['path']} -> {target_path}")
                    changed_cnt += 1
                del target_files[translated_file]
        for target_file in target_files:
            target_path = os.path.join(target_root, target_file)
            if dry_run is False:
                os.remove(target_path)
                self.log.info(f"Removed orphan {target_path}")
            else:
                self.log.info(f"Would remove orphan {target_path}")
            delete_cnt += 1
        if vacuum_target is True:
            delete_cnt += self.vacuum(dry_run)
        return new_cnt, changed_cnt, delete_cnt
