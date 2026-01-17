#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTX two-way batch tool (Export + Import) - UTF-16 TXT

Menu:
  1) batch export
     - Ask: folder containing .btx
     - Output: a NEW folder next to this tool, with suffix "_export"
       (example: <tool_dir>/<input_folder_name>_export/)
     - Keep file names (relative paths preserved). .btx -> .txt

  2) batch import
     - Ask: folder containing original .btx
     - Ask: folder containing translated .txt
     - Output: a NEW folder next to this tool, with suffix "_new"
       (example: <tool_dir>/<btx_folder_name>_new/)
     - Keep file names (relative paths preserved). Output files are .btx.

TXT format:
  Header:
    # Source: <file>
    # StringPoolOffset: 0x....
    # TXTEncoding: UTF-16LE+BOM (export default), import auto-detect
    # Format: index<TAB>offset_hex<TAB>text (text escaped: \\r, \\n, \\t)

  Records:
    0001\t0x00000180\t<text...>

Notes about Import:
- It rebuilds the sequential string pool using the translated texts (UTF-16LE + 0x0000).
- It updates the BTX pointer table offsets to keep internal references consistent:
    Each pointer keeps the same "byte offset inside its base string" as the original.
- If a .txt file is missing, the original .btx is copied unchanged to output.
"""

from __future__ import annotations

import shutil
import struct
from bisect import bisect_right
from pathlib import Path
from typing import List, Tuple, Dict

MAGIC = b"BTX "

# TXT export settings
TXT_EXPORT_ENCODING = "utf-16le"
TXT_EXPORT_BOM = b"\xff\xfe"  # UTF-16LE BOM


# ------------------ Low-level helpers ------------------

def u32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def p_u32_le(buf: bytearray, off: int, val: int) -> None:
    struct.pack_into("<I", buf, off, val & 0xFFFFFFFF)

def looks_like_btx(b: bytes) -> bool:
    return len(b) >= 16 and b[:4] == MAGIC

def read_utf16le_z(b: bytes, off: int) -> Tuple[str, int]:
    """Read UTF-16LE NUL-terminated string. Returns (text, next_offset_after_terminator)."""
    if off % 2 != 0:
        raise ValueError(f"UTF-16LE string offset must be even, got 0x{off:X}")
    out_units: List[int] = []
    i = off
    n = len(b)
    while i + 1 < n:
        code = struct.unpack_from("<H", b, i)[0]
        i += 2
        if code == 0:
            break
        out_units.append(code)
    raw = struct.pack("<" + "H" * len(out_units), *out_units) if out_units else b""
    return raw.decode("utf-16le", errors="replace"), i

def escape_one_line(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace("\r", "\\r")
         .replace("\n", "\\n")
         .replace("\t", "\\t")
    )

def unescape_one_line(s: str) -> str:
    # Reverse of escape_one_line (supports \\ \\r \\n \\t)
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nx = s[i + 1]
            if nx == "r":
                out.append("\r"); i += 2; continue
            if nx == "n":
                out.append("\n"); i += 2; continue
            if nx == "t":
                out.append("\t"); i += 2; continue
            if nx == "\\":
                out.append("\\"); i += 2; continue
            # unknown escape -> keep backslash literally
        out.append(ch)
        i += 1
    return "".join(out)

def read_text_auto(path: Path) -> Tuple[str, str]:
    """
    Auto-detect text encoding by BOM.
    Returns (text, encoding_name).
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16le", errors="replace"), "utf-16le"
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16be", errors="replace"), "utf-16be"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace"), "utf-8-sig"
    # fallback
    return raw.decode("utf-8", errors="replace"), "utf-8"

def write_text_utf16le_bom(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # normalize newline to \n for stable parsing
    if "\r\n" in text:
        text = text.replace("\r\n", "\n")
    path.write_bytes(TXT_EXPORT_BOM + text.encode("utf-16le", errors="surrogatepass"))


# ------------------ BTX parsing ------------------

def compute_pool_off_from_header(b: bytes) -> int:
    """
    Observed BTX layout:
      count (u32) at 0x14
      pointer table at 0x20 with (count-1) entries, each 8 bytes (id, rel_off)
      string pool begins at: 0x20 + (count-1)*8
    """
    if len(b) < 0x18:
        raise ValueError("File too small for BTX header")
    count = u32_le(b, 0x14)
    if count < 2 or count > 0x200000:
        raise ValueError(f"Suspicious count at 0x14: {count}")
    pool_off = 0x20 + (count - 1) * 8
    if pool_off % 2 != 0:
        pool_off += 1
    if pool_off >= len(b):
        raise ValueError("Computed pool offset beyond EOF")
    return pool_off

def guess_string_pool_offset(b: bytes) -> int:
    # Prefer header-based pool offset
    if looks_like_btx(b):
        pool_off = compute_pool_off_from_header(b)
        try:
            s, _ = read_utf16le_z(b, pool_off)
            if len(s) >= 1:
                return pool_off
        except Exception:
            pass

    # Fallback: find earliest even offset that looks like UTF-16 text
    for off in range(0, len(b) - 4, 2):
        try:
            s, _ = read_utf16le_z(b, off)
        except Exception:
            continue
        if len(s) >= 4:
            printable = sum(1 for ch in s[:64] if ch.isprintable() or ch in "\r\n\t")
            if printable / max(1, min(64, len(s))) > 0.7:
                return off
    raise ValueError("Could not locate string pool start")

def parse_pool_strings_with_bounds(b: bytes, pool_off: int) -> Tuple[List[int], List[int]]:
    """
    Return (starts, ends) for sequential UTF-16LE NUL-terminated strings in the pool.
    - starts[i] is absolute start offset
    - ends[i] is absolute offset right AFTER terminator
    """
    starts: List[int] = []
    ends: List[int] = []

    off = pool_off
    n = len(b)

    while off < n:
        if off + 1 >= n:
            break

        # skip 0x0000 padding
        if off % 2 == 0 and struct.unpack_from("<H", b, off)[0] == 0:
            off += 2
            continue
        if off % 2 != 0:
            off += 1
            continue

        start = off
        _, next_off = read_utf16le_z(b, off)
        starts.append(start)
        ends.append(next_off)
        off = next_off

    return starts, ends

def export_btx_to_txt(btx_path: Path, out_txt: Path) -> None:
    b = btx_path.read_bytes()
    if not looks_like_btx(b):
        raise ValueError("Not a BTX file")

    pool_off = guess_string_pool_offset(b)
    starts, ends = parse_pool_strings_with_bounds(b, pool_off)

    lines = []
    lines.append(f"# Source: {btx_path.name}")
    lines.append(f"# StringPoolOffset: 0x{pool_off:X}")
    lines.append("# TXTEncoding: UTF-16LE+BOM (export default), import auto-detect")
    lines.append("# Format: index<TAB>offset_hex<TAB>text (text escaped: \\\\r, \\\\n, \\\\t)")
    for idx, (s_off, e_off) in enumerate(zip(starts, ends), 1):
        raw = b[s_off:e_off]
        if len(raw) >= 2 and raw[-2:] == b"\x00\x00":
            raw = raw[:-2]
        text = raw.decode("utf-16le", errors="replace")
        lines.append(f"{idx:04d}\t0x{s_off:08X}\t{escape_one_line(text)}")

    # Write UTF-16LE+BOM
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    write_text_utf16le_bom(out_txt, "\n".join(lines) + "\n")

def parse_translated_txt(txt_path: Path) -> List[str]:
    """
    Parse exported TXT and return list of translated strings in pool order.
    Auto-detect TXT encoding (UTF-16 BOM / UTF-8 fallback).
    """
    text, _enc = read_text_auto(txt_path)
    lines = text.splitlines()
    records: Dict[int, str] = {}
    for ln in lines:
        if not ln or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t", 2)
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        records[idx] = unescape_one_line(parts[2])

    if not records:
        raise ValueError("No records found in TXT")

    max_idx = max(records.keys())
    out: List[str] = []
    for i in range(1, max_idx + 1):
        if i not in records:
            raise ValueError(f"Missing index {i} in TXT")
        out.append(records[i])
    return out


# ------------------ Import (rebuild) ------------------

def parse_pointer_table(b: bytes) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Returns (pool_off, entries)
    entries = list of (id, rel_off)
    """
    if not looks_like_btx(b):
        raise ValueError("Not a BTX file")
    count = u32_le(b, 0x14)
    if count < 2:
        raise ValueError("Invalid count")
    entry_count = count - 1
    table_off = 0x20
    table_size = entry_count * 8
    pool_off = table_off + table_size
    if pool_off > len(b):
        raise ValueError("Corrupt table/pool offsets")

    entries: List[Tuple[int,int]] = []
    for i in range(entry_count):
        idv, rel = struct.unpack_from("<II", b, table_off + i * 8)
        entries.append((idv, rel))
    return pool_off, entries

def build_new_btx(original_btx: bytes, translated_pool_strings: List[str]) -> bytes:
    pool_off, entries = parse_pointer_table(original_btx)

    starts, ends = parse_pool_strings_with_bounds(original_btx, pool_off)
    if len(starts) != len(translated_pool_strings):
        raise ValueError(f"TXT string count ({len(translated_pool_strings)}) != pool strings ({len(starts)})")

    pre_gap = original_btx[pool_off:starts[0]] if starts else b""
    gaps: List[bytes] = []
    for i in range(len(starts) - 1):
        gaps.append(original_btx[ends[i]:starts[i + 1]])
    tail = original_btx[ends[-1]:] if ends else original_btx[pool_off:]

    orig_starts = starts[:]
    base_and_within: List[Tuple[int,int]] = []
    for _idv, rel in entries:
        ptr_abs = pool_off + rel
        j = bisect_right(orig_starts, ptr_abs) - 1
        if j < 0:
            raise ValueError(f"Pointer before first string: rel=0x{rel:X}")
        within = ptr_abs - orig_starts[j]
        base_and_within.append((j, within))

    new_str_bytes: List[bytes] = []
    for s in translated_pool_strings:
        enc = s.encode("utf-16le", errors="surrogatepass") + b"\x00\x00"
        new_str_bytes.append(enc)

    new_pool = bytearray()
    new_pool += pre_gap

    new_abs_starts: List[int] = []
    cur_abs = pool_off + len(pre_gap)
    for i, enc in enumerate(new_str_bytes):
        new_abs_starts.append(cur_abs)
        new_pool += enc
        cur_abs += len(enc)
        if i < len(gaps):
            new_pool += gaps[i]
            cur_abs += len(gaps[i])
    new_pool += tail

    new_prefix = bytearray(original_btx[:pool_off])
    table_off = 0x20
    for i, ((_idv, _old_rel), (base_idx, within)) in enumerate(zip(entries, base_and_within)):
        new_ptr_abs = new_abs_starts[base_idx] + within
        new_rel = new_ptr_abs - pool_off
        if new_rel < 0 or new_rel >= 0xFFFFFFFF:
            raise ValueError(f"Bad new_rel computed: 0x{new_rel:X}")
        if new_ptr_abs < pool_off or new_ptr_abs >= (pool_off + len(new_pool)):
            raise ValueError(f"Pointer out of new pool: abs=0x{new_ptr_abs:X}")
        p_u32_le(new_prefix, table_off + i * 8 + 4, int(new_rel))

    return bytes(new_prefix) + bytes(new_pool)


# ------------------ Batch operations ------------------

def iter_files(root: Path, suffix: str):
    yield from root.rglob(f"*{suffix}")

def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def relpath_under(root: Path, fp: Path) -> Path:
    try:
        return fp.relative_to(root)
    except Exception:
        return Path(fp.name)

def do_batch_export():
    src = Path(input("Nhập thư mục chứa file .btx: ").strip().strip('"'))
    if not src.is_dir():
        print("[LỖI] Đường dẫn không phải thư mục.")
        return

    tool_dir = Path(__file__).resolve().parent
    out_dir = tool_dir / f"{src.name}_export"
    ensure_clean_dir(out_dir)

    ok = fail = 0
    for btx in iter_files(src, ".btx"):
        rel = relpath_under(src, btx)
        out_txt = (out_dir / rel).with_suffix(".txt")
        try:
            export_btx_to_txt(btx, out_txt)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {btx}: {e}")
    print(f"Xong EXPORT. OK={ok}, FAIL={fail}")
    print(f"Output: {out_dir}")
    print("TXT xuất ra: UTF-16LE + BOM")

def do_batch_import():
    btx_root = Path(input("Nhập thư mục chứa file .btx GỐC: ").strip().strip('"'))
    txt_root = Path(input("Nhập thư mục chứa file .txt DỊCH: ").strip().strip('"'))
    if not btx_root.is_dir():
        print("[LỖI] Thư mục BTX gốc không hợp lệ.")
        return
    if not txt_root.is_dir():
        print("[LỖI] Thư mục TXT dịch không hợp lệ.")
        return

    tool_dir = Path(__file__).resolve().parent
    out_dir = tool_dir / f"{btx_root.name}_new"
    ensure_clean_dir(out_dir)

    ok = miss = fail = 0
    for btx in iter_files(btx_root, ".btx"):
        rel = relpath_under(btx_root, btx)
        txt = (txt_root / rel).with_suffix(".txt")
        out_btx = out_dir / rel

        out_btx.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not txt.exists():
                shutil.copy2(btx, out_btx)
                miss += 1
                continue

            original = btx.read_bytes()
            translated_list = parse_translated_txt(txt)
            rebuilt = build_new_btx(original, translated_list)
            out_btx.write_bytes(rebuilt)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {btx}: {e}")

    print(f"Xong IMPORT. OK={ok}, MISS_TXT={miss}, FAIL={fail}")
    print(f"Output: {out_dir}")
    print("TXT đọc vào: auto-detect (UTF-16 BOM / UTF-8 fallback)")

def main():
    while True:
        print("\n===== BTX TOOL 2 CHIỀU =====")
        print("1) batch export (TXT UTF-16LE+BOM)")
        print("2) batch import (auto-detect TXT encoding)")
        print("0) thoát")
        c = input("Chọn: ").strip()
        if c == "1":
            do_batch_export()
        elif c == "2":
            do_batch_import()
        elif c == "0":
            break
        else:
            print("Chọn không hợp lệ.")

if __name__ == "__main__":
    main()
