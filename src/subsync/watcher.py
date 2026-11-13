import hashlib
import logging
import queue
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

SUPPORTED_MEDIA_EXTS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
}


class ProcessedStateStore:
    """基于 SQLite 的已处理文件去重存储"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def is_processed(self, file_hash: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM processed_files WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            )
            return cur.fetchone() is not None

    def mark(self, file_hash: str, file_path: Path, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO processed_files (file_hash, file_path, status, processed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = excluded.status,
                    processed_at = excluded.processed_at
                """,
                (file_hash, str(file_path), status, now),
            )
            conn.commit()


def compute_fast_fingerprint(file_path: Path, chunk_size: int = 65536) -> str:
    """结合文件大小与头尾各 64KB 数据计算快速指纹，避免大文件全量哈希耗时"""
    size = file_path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(str(size).encode("utf-8"))

    with file_path.open("rb") as f:
        hasher.update(f.read(chunk_size))
        if size > chunk_size:
            f.seek(max(0, size - chunk_size))
            hasher.update(f.read(chunk_size))

    return hasher.hexdigest()


def wait_until_file_stable(
    file_path: Path,
    interval: float = 1.5,
    max_checks: int = 20,
) -> bool:
    """等待文件写入/复制完成（连续两次检测大小不变且大于 0）"""
    last_size = -1
    for _ in range(max_checks):
        if not file_path.exists():
            return False
        current_size = file_path.stat().st_size
        if current_size > 0 and current_size == last_size:
            return True
        last_size = current_size
        time.sleep(interval)
    return False


class MediaFolderHandler(FileSystemEventHandler):
    """监听新建与重命名事件，将合规媒体文件推入任务队列"""

    def __init__(self, task_queue: queue.Queue[Path]) -> None:
        self.task_queue = task_queue

    def _enqueue_if_valid(self, raw_path: str) -> None:
        path = Path(raw_path)
        if path.suffix.lower() in SUPPORTED_MEDIA_EXTS:
            logger.debug("捕获媒体文件事件: %s", path.name)
            self.task_queue.put(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue_if_valid(str(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory and hasattr(event, "dest_path"):
            self._enqueue_if_valid(str(event.dest_path))


def watch_directory(
    watch_dir: Path,
    db_path: Path,
    process_callback: Callable[[Path], Path],
    recursive: bool = False,
    scan_existing: bool = False,
) -> None:
    """启动目录监听主循环，支持 Ctrl+C 优雅退出"""
    if not watch_dir.is_dir():
        raise NotADirectoryError(f"监听路径不是有效目录: {watch_dir}")

    store = ProcessedStateStore(db_path)
    task_queue: queue.Queue[Path] = queue.Queue()

    if scan_existing:
        pattern = "**/*" if recursive else "*"
        for item in sorted(watch_dir.glob(pattern)):
            if item.is_file() and item.suffix.lower() in SUPPORTED_MEDIA_EXTS:
                task_queue.put(item)

    handler = MediaFolderHandler(task_queue)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=recursive)
    observer.start()
    logger.info("已启动目录监听: %s (recursive=%s)", watch_dir, recursive)

    try:
        while True:
            try:
                target_path = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not wait_until_file_stable(target_path):
                logger.warning("文件已消失或仍在持续写入，跳过: %s", target_path.name)
                task_queue.task_done()
                continue

            expected_srt = target_path.with_suffix(".bilingual.srt")
            if expected_srt.exists():
                logger.info("字幕已存在，跳过处理: %s", expected_srt.name)
                task_queue.task_done()
                continue

            file_hash = compute_fast_fingerprint(target_path)
            if store.is_processed(file_hash):
                logger.info("命中 SQLite 去重记录，跳过: %s", target_path.name)
                task_queue.task_done()
                continue

            try:
                process_callback(target_path)
                store.mark(file_hash, target_path, status="success")
            except Exception:
                logger.exception("处理文件失败 %s", target_path.name)
                store.mark(file_hash, target_path, status="failed")
            finally:
                task_queue.task_done()

    except KeyboardInterrupt:
        logger.info("收到退出信号，正在停止目录监听...")
    finally:
        observer.stop()
        observer.join()
