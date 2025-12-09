from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Protocol
import csv
from pathlib import Path
import os
import concurrent.futures

@dataclass(order=True, frozen=True)
class Instance:
    """
    Một instance s = (feature, idx, x, y)
    Thứ tự: feature name, sau đó index – đúng mô tả Sec.3.
    """
    feature: str
    idx: int
    x: float
    y: float

    def __str__(self) -> str:
        return f"{self.feature}.{self.idx}"


@dataclass
class SpatialDataset:
    """
    Tập dữ liệu không gian (F,S) như Sec.2.
    """
    instances: List[Instance] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sort chuẩn theo paper: theo feature name rồi index
        self.instances.sort()
        self.feature_to_instances: Dict[str, List[Instance]] = {}
        for s in self.instances:
            self.feature_to_instances.setdefault(s.feature, []).append(s)

    @property
    def features(self) -> Set[str]:
        return set(self.feature_to_instances.keys())

    def feature_counts(self) -> Dict[str, int]:
        return {f: len(v) for f, v in self.feature_to_instances.items()}


# -------------------- CSV helpers with Multiprocessing --------------------

def _parse_csv_lines(lines: List[str]) -> List[Instance]:
    """Helper function to parse a list of CSV lines."""
    instances = []
    # If the chunk starts with header lines, the logic below might fail if we don't handle it.
    # But chunks are byte-based. We will strip the header in the main process.
    # However, intermediate chunks won't have headers.
    
    # We use csv.reader to handle quoted fields correctly if any, though our data is simple.
    reader = csv.reader(lines)
    for row in reader:
        if not row: continue
        # Expect 4 columns or more. We need to be robust.
        # This function assumes 'row' is a list of strings
        # We need to map columns to feature/idx/x/y manually if no header logic in chunks.
        # Given the data format is usually consistent, we try to detect if it's a header or data.
        
        # Heuristic: if first column is "feature" or "Feature", skip
        if row[0].lower() == "feature":
            continue
            
        try:
            # We assume order: feature, idx, x, y if no header is present in chunk
            # But the original load_csv used DictReader. This is tricky for chunks without headers.
            # We will assume STANDARD COLUMN ORDER: feature, idx, x, y.
            # If the CSV has different order, this chunking approach needs to know the order.
            # For simplicity in this optimization, we assume: feature, idx, x, y
            
            # Check length to avoid index error
            if len(row) < 4: continue
            
            feature = row[0]
            idx_str = row[1]
            x_str = row[2]
            y_str = row[3]
            
            instances.append(
                Instance(
                    feature=str(feature),
                    idx=int(idx_str),
                    x=float(x_str),
                    y=float(y_str),
                )
            )
        except ValueError:
            # Skip malformed lines
            continue
            
    return instances

def _read_chunk(path: str, start: int, size: int) -> List[str]:
    """Read a chunk of lines from a file."""
    with open(path, 'rb') as f:
        f.seek(start)
        # Verify we are at the start of a line or file
        if start != 0:
            # In general, 'start' calculated below determines valid split points.
            # But simpler approach: read 'size' bytes, then read until newline.
            pass
            
        chunk_bytes = f.read(size)
        
        # We need to ensure we read until the end of the last line
        remainder = f.readline()
        chunk_bytes += remainder
        
        # Decode
        content = chunk_bytes.decode('utf-8', errors='ignore')
        lines = content.splitlines()
        
        # If start != 0, we might have started in the middle of a line?
        # A robust way is:
        # Worker k: seek to k*(size/N). Read a line and discard it (it belongs to k-1).
        # EXCEPT if k=0.
        # Then read until (k+1)*(size/N). Finish that line.
        # This guarantees disjoint sets covering all lines.
        pass
    return lines

def _worker_task(path: str, start: int, end: int) -> List[Instance]:
    instances: List[Instance] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        f.seek(start)
        # If not at start of file, discard the first line (partial line from previous chunk)
        if start != 0:
            f.readline()
            
        # Read until we pass 'end'
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            
            # Parse line
            # Manual CSV parsing for speed and simplicity in this specific task
            # Assuming format: feature,idx,x,y
            parts = line.strip().split(',')
            if len(parts) < 4: continue
            
            # Skip header if it appears (only likely in first chunk, but handled by start check usually)
            # Actually, if start=0, we read the header.
            if start == 0 and (parts[0].lower() == 'feature' or parts[1].lower() == 'header'):
                continue
                
            try:
                instances.append(Instance(
                    feature=parts[0].strip(),
                    idx=int(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3])
                ))
            except ValueError:
                continue
    return instances

def load_csv(path: str | Path, workers: int = 1) -> SpatialDataset:
    """
    Đọc file CSV với cột: feature, idx, x, y.
    Supports multithreading (multiprocessing) with 'workers' > 1.
    Assumes CSV has standard order: feature, idx, x, y.
    """
    path = Path(path)
    if workers <= 1:
        # Use legacy single-threaded method (but robust with DictReader)
        instances: List[Instance] = []
        with path.open("r", newline="", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check if headers match expectation to fallback or error?
            # Existing code was flexible. We keep it flexible for serial.
            for row in reader:
                feature = row.get("feature") or row.get("Feature")
                idx = row.get("idx") or row.get("InstanceID") or row.get("Instance")
                x = row.get("x") or row.get("X")
                y = row.get("y") or row.get("Y")
                
                if feature and idx and x and y:
                    instances.append(
                        Instance(
                            feature=str(feature),
                            idx=int(idx),
                            x=float(x),
                            y=float(y),
                        )
                    )
        return SpatialDataset(instances)
    
    # Multiprocessing approach
    file_size = path.stat().st_size
    chunk_size = file_size // workers
    futures = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < workers - 1 else file_size
            futures.append(executor.submit(_worker_task, str(path), start, end))
            
    instances = []
    for future in concurrent.futures.as_completed(futures):
        instances.extend(future.result())
        
    return SpatialDataset(instances)


def save_csv(dataset: SpatialDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "idx", "x", "y"])
        for s in dataset.instances:
            writer.writerow([s.feature, s.idx, s.x, s.y])
