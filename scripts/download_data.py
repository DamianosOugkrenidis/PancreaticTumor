import argparse
import hashlib
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RECORD = "5510650"
RECORD_URL = "https://zenodo.org/records/{record}/files/{name}?download=1"
API_URL = "https://zenodo.org/api/records/{record}"

KNOWN_FILES: list[dict] = [
    {"key": "Metadata.xlsx", "size": 14330,
     "md5": "327d8186e9c44dfc2fc9859b26830e1f", "kind": "metadata"},
    {"key": "Pancreas_segmentations.zip", "size": 560156,
     "md5": "a0c3c838496502105cf02a2217e1b586", "kind": "masks"},
    {"key": "Tumor_segmentations.zip", "size": 190746,
     "md5": "f045275daee60be12b68d3e63a651d6d", "kind": "masks"},
    {"key": "Volume files.zip", "size": 4282285313,
     "md5": "6068ad9fc389c56d6c448f4370af6051", "kind": "volumes"},
]

CHUNK_BYTES = 4 * 1024 * 1024
PARALLEL_CHUNK_BYTES = 256 * 1024 * 1024

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the pancreatic CT dataset from Zenodo.")
    parser.add_argument("--dest", default="datasets/tcia_pda",
                        help="Destination directory (archives in <dest>/raw, files in <dest>/extracted).")
    parser.add_argument("--record", default=DEFAULT_RECORD, help="Zenodo record id.")
    parser.add_argument("--files", nargs="*", default=None, help="Subset of file keys.")
    parser.add_argument("--small-only", action="store_true",
                        help="Skip the large volume archive.")
    parser.add_argument("--from-api", action="store_true",
                        help="Read the file list and checksums from the Zenodo API.")
    parser.add_argument("--connections", type=int, default=8,
                        help="Parallel HTTP range connections per file (1 disables chunking).")
    parser.add_argument("--no-extract", action="store_true", help="Only download the archives.")
    parser.add_argument("--force", action="store_true", help="Re-download even verified files.")
    parser.add_argument("--list", action="store_true", help="Print the file table and exit.")
    return parser.parse_args()

def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fetch_file_table(record: str) -> list[dict]:
    import json
    import urllib.request

    with urllib.request.urlopen(API_URL.format(record=record), timeout=30) as response:
        payload = json.load(response)
    table = []
    for entry in payload.get("files", []):
        checksum = entry.get("checksum", "")
        table.append({
            "key": entry["key"],
            "size": int(entry.get("size", 0)),
            "md5": checksum.split(":", 1)[1] if checksum.startswith("md5:") else "",
            "kind": "volumes" if "volume" in entry["key"].lower()
            else ("masks" if "segmentation" in entry["key"].lower() else "metadata"),
        })
    return table

def is_complete(path: Path, expected_md5: str, expected_size: int) -> bool:
    if not path.is_file():
        return False
    if expected_size and path.stat().st_size != expected_size:
        return False
    if expected_md5 and md5_of(path) != expected_md5:
        return False
    return True

class ProgressBar:
    def __init__(self, total: int, label: str) -> None:
        from tqdm import tqdm

        self._lock = threading.Lock()
        self._bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                         leave=False, desc=label)

    def update(self, amount: int) -> None:
        with self._lock:
            self._bar.update(amount)

    def close(self) -> None:
        self._bar.close()

def download_range(url: str, start: int, end: int, path: Path, progress: ProgressBar) -> None:
    import time

    import requests

    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        progress.update(expected)
        return

    last_error: Exception | None = None
    for attempt in range(1, 13):
        have = path.stat().st_size if path.is_file() else 0
        if have > expected:
            path.unlink(missing_ok=True)
            have = 0
        headers = {"Range": f"bytes={start + have}-{end}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as response:
                if response.status_code == 416:
                    return
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"range request rejected (HTTP {response.status_code})")
                if response.status_code == 200 and have:
                    have = 0
                    path.unlink(missing_ok=True)
                with path.open("ab" if have else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.update(len(chunk))
            if path.stat().st_size == expected:
                return
            raise RuntimeError(f"{path.name}: {path.stat().st_size} bytes, expected {expected}")
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt < 12:
                time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"{path.name}: giving up after 12 attempts ({last_error})")

def download_parallel(url: str, destination: Path, expected_md5: str, expected_size: int,
                      connections: int, force: bool = False) -> bool:
    if not force and is_complete(destination, expected_md5, expected_size):
        print(f"  already present and verified: {destination.name}")
        return False
    if not expected_size:
        return download_stream(url, destination, expected_md5, connections=1, force=force)

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_dir = destination.with_suffix(destination.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)

    ranges = []
    start = 0
    while start < expected_size:
        end = min(start + PARALLEL_CHUNK_BYTES, expected_size) - 1
        ranges.append((start, end))
        start = end + 1

    progress = ProgressBar(expected_size, f"  {destination.name}")
    for chunk_start, chunk_end in ranges:
        part = part_dir / f"part_{chunk_start:012d}"
        if part.is_file() and part.stat().st_size == chunk_end - chunk_start + 1:
            progress.update(part.stat().st_size)

    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = {pool.submit(download_range, url, chunk_start, chunk_end,
                                   part_dir / f"part_{chunk_start:012d}", progress): chunk_start
                       for chunk_start, chunk_end in ranges}
            for future in as_completed(futures):
                future.result()
    finally:
        progress.close()

    merged = destination.with_suffix(destination.suffix + ".part")
    with merged.open("wb") as sink:
        for chunk_start, _ in ranges:
            with (part_dir / f"part_{chunk_start:012d}").open("rb") as source:
                while True:
                    block = source.read(CHUNK_BYTES)
                    if not block:
                        break
                    sink.write(block)

    if expected_size and merged.stat().st_size != expected_size:
        raise RuntimeError(f"{destination.name}: merged size mismatch")
    if expected_md5:
        actual = md5_of(merged)
        if actual != expected_md5:
            raise RuntimeError(f"{destination.name}: MD5 mismatch (expected {expected_md5}, "
                               f"got {actual})")
    merged.replace(destination)
    for chunk_start, _ in ranges:
        (part_dir / f"part_{chunk_start:012d}").unlink(missing_ok=True)
    part_dir.rmdir()
    print(f"  downloaded and verified: {destination.name} "
          f"({len(ranges)} chunks of {PARALLEL_CHUNK_BYTES // 1024**2} MiB, {connections} workers)")
    return True

def download_stream(url: str, destination: Path, expected_md5: str = "",
                    expected_size: int = 0, connections: int = 1,
                    force: bool = False) -> bool:
    import requests
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force and is_complete(destination, expected_md5, expected_size):
        print(f"  already present and verified: {destination.name}")
        return False

    partial = destination.with_suffix(destination.suffix + ".part")
    resume_at = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        if resume_at and response.status_code != 206:
            resume_at = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) + resume_at
        mode = "ab" if resume_at else "wb"
        with partial.open(mode) as handle, tqdm(total=total, initial=resume_at, unit="B",
                                                unit_scale=True, unit_divisor=1024,
                                                desc=f"  {destination.name}", leave=False) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))

    if expected_size and partial.stat().st_size != expected_size:
        raise RuntimeError(f"{destination.name}: expected {expected_size} bytes, "
                           f"got {partial.stat().st_size}")
    if expected_md5:
        actual = md5_of(partial)
        if actual != expected_md5:
            raise RuntimeError(f"{destination.name}: MD5 mismatch (expected {expected_md5}, "
                               f"got {actual})")
    partial.replace(destination)
    print(f"  downloaded and verified: {destination.name}")
    return True

def safe_extract_zip(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename
            if member.is_dir():
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"Unsafe path inside {archive.name}: {name}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                print(f"  skipping symlink entry: {name}")
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as sink:
                sink.write(source.read())
            extracted += 1
    return extracted

def clean_macos_junk(root: Path) -> int:
    removed = 0
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.name == "__MACOSX" and path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    child.unlink()
                    removed += 1
            path.rmdir()
        elif path.is_file() and path.name.startswith("._"):
            path.unlink()
            removed += 1
    return removed

def main() -> int:
    arguments = parse_arguments()
    dest = Path(arguments.dest)
    raw_dir = dest / "raw"
    extracted_dir = dest / "extracted"

    files = fetch_file_table(arguments.record) if arguments.from_api else KNOWN_FILES
    if arguments.files:
        wanted = {name.lower() for name in arguments.files}
        files = [entry for entry in files if entry["key"].lower() in wanted]
    elif arguments.small_only:
        files = [entry for entry in files if entry["kind"] != "volumes"]

    if arguments.list:
        print(f"Zenodo record {arguments.record}:")
        for entry in files:
            print(f"  {entry['key']:<28} {entry['size'] / 1024**2:10.2f} MiB  md5 {entry['md5']}")
        print(f"\nTotal: {sum(entry['size'] for entry in files) / 1024**2:.1f} MiB")
        return 0

    if not files:
        print("Nothing selected; pass --files or drop --small-only.")
        return 1

    print(f"Destination: {dest.resolve()}")
    print(f"Downloading {len(files)} file(s), "
          f"{sum(entry['size'] for entry in files) / 1024**3:.2f} GiB total, "
          f"{arguments.connections} connection(s) per file")

    for entry in files:
        url = RECORD_URL.format(record=arguments.record, name=quote(entry["key"]))
        print(f"- {entry['key']} ({entry['size'] / 1024**2:.1f} MiB)")
        if arguments.connections > 1 and entry["size"] > 64 * 1024**2:
            download_parallel(url, raw_dir / entry["key"], entry.get("md5", ""),
                              entry.get("size", 0), arguments.connections,
                              force=arguments.force)
        else:
            download_stream(url, raw_dir / entry["key"], entry.get("md5", ""),
                            entry.get("size", 0), force=arguments.force)

    if arguments.no_extract:
        print("\nArchives stored in", raw_dir.resolve())
        return 0

    for entry in files:
        archive = raw_dir / entry["key"]
        if archive.suffix.lower() != ".zip":
            continue
        print(f"- extracting {archive.name}")
        count = safe_extract_zip(archive, extracted_dir)
        print(f"  {count} files extracted")

    removed = clean_macos_junk(extracted_dir)
    if removed:
        print(f"- removed {removed} macOS metadata entries")

    print(f"\nExtracted data: {extracted_dir.resolve()}")
    print('Next step: python -m scripts.prepare_data --raw '
          f'"{extracted_dir}" --out datasets/prepared')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
