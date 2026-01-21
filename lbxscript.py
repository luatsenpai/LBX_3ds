#!/usr/bin/env python3
# batch_dump_e_json_mirror.py
import os
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

PRINTABLE_RATIO_MIN = 0.60

def is_reasonable_text(s: str) -> bool:
    if not s:
        return True
    good = 0
    for ch in s:
        o = ord(ch)
        if ch in "\r\n\t":
            good += 1
        elif 0x20 <= o <= 0x7E:
            good += 1
        elif 0x00A0 <= o <= 0x02AF:  # Latin Extended
            good += 1
        elif 0x3000 <= o <= 0x30FF:  # JP punctuation + kana
            good += 1
        elif 0x4E00 <= o <= 0x9FFF:  # CJK
            good += 1
        elif 0xFF00 <= o <= 0xFFEF:  # fullwidth
            good += 1
    return (good / max(1, len(s))) >= PRINTABLE_RATIO_MIN

def read_u32le(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]

def read_u32be(buf: bytes, off: int) -> int:
    return struct.unpack_from(">I", buf, off)[0]

def try_parse_utf16le_strings(buf: bytes, start: int, count: int, hard_limit: int = 1_000_000) -> Tuple[List[str], int]:
    if start < 0 or start >= len(buf):
        raise ValueError("start out of range")
    p = start
    out: List[str] = []
    consumed = 0

    for _ in range(count):
        chars: List[int] = []
        while p + 1 < len(buf):
            w = struct.unpack_from("<H", buf, p)[0]
            p += 2
            consumed += 2
            if consumed > hard_limit:
                raise ValueError("too much consumed")
            if w == 0:
                break
            chars.append(w)

        if chars:
            raw = struct.pack("<" + "H" * len(chars), *chars)
            s = raw.decode("utf-16le", errors="replace")
        else:
            s = ""
        out.append(s)

    return out, p

def score_strings(strings: List[str]) -> int:
    score = 0
    for s in strings:
        if s == "":
            score += 1
            continue
        if "\ufffd" in s:
            score -= 2
        if is_reasonable_text(s):
            score += 5
        else:
            score -= 3
        if "<w" in s and ">" in s:
            score += 1
    return score

def parse_btx(buf: bytes, off: int) -> Dict[str, Any]:
    if buf[off:off+4] != b"BTX ":
        raise ValueError("Not a BTX chunk")

    ver = read_u32le(buf, off + 4)
    header_size = read_u32le(buf, off + 8)
    unk0 = read_u32le(buf, off + 12)
    group_count = read_u32le(buf, off + 16)
    string_count = read_u32le(buf, off + 20)
    unk1 = read_u32le(buf, off + 24)
    table_off = read_u32le(buf, off + 28)

    candidates = []
    candidates.append(off + 0x20 + max(0, string_count - 1) * 8)
    candidates.append(off + 0x20 + max(0, string_count) * 8)
    candidates.append(off + 0x20)
    if 0x20 <= table_off < len(buf):
        candidates.append(off + table_off)

    best = None  # (score, start, strings, endpos)
    for start in candidates:
        try:
            strings, endpos = try_parse_utf16le_strings(buf, start, string_count)
        except Exception:
            continue
        sc = score_strings(strings)
        if best is None or sc > best[0]:
            best = (sc, start, strings, endpos)

    if best is None:
        return {
            "offset": off,
            "version": ver,
            "header_size": header_size,
            "unk0": unk0,
            "group_count": group_count,
            "string_count": string_count,
            "unk1": unk1,
            "table_off": table_off,
            "strings": [],
            "string_data_start_rel": None,
            "parse_note": "BTX header read ok but failed to locate UTF-16LE string region with heuristics.",
        }

    _, start, strings, endpos = best
    return {
        "offset": off,
        "version": ver,
        "header_size": header_size,
        "unk0": unk0,
        "group_count": group_count,
        "string_count": string_count,
        "unk1": unk1,
        "table_off": table_off,
        "string_data_start_rel": start - off,
        "string_data_end_rel": endpos - off,
        "strings": strings,
    }

def parse_lip_chunks(buf: bytes) -> List[Dict[str, Any]]:
    out = []
    i = 0
    while True:
        j = buf.find(b"LIP ", i)
        if j == -1:
            break

        try:
            size_be = read_u32be(buf, j + 4)
        except Exception:
            break

        total_len = 4 + size_be
        if total_len <= 8 or j + total_len > len(buf):
            i = j + 1
            continue

        payload = buf[j + 8 : j + total_len]
        t = read_u32be(payload, 0) if len(payload) >= 4 else None
        r = read_u32be(payload, 4) if len(payload) >= 8 else None
        rest = payload[8:] if len(payload) >= 8 else b""

        out.append({
            "offset": j,
            "size_be": size_be,
            "total_len": total_len,
            "t_u32be": t,
            "r_u32be": r,
            "rest_hex": rest.hex(),
        })

        i = j + total_len
    return out

def collect_fourcc(buf: bytes) -> List[Dict[str, Any]]:
    tags = []
    for k in range(0, len(buf) - 4):
        b4 = buf[k:k+4]
        if all(0x20 <= c <= 0x7E for c in b4):
            if any((65 <= c <= 90) or (97 <= c <= 122) for c in b4):
                tags.append({"off": k, "tag": b4.decode("ascii", errors="ignore")})
    by_tag: Dict[str, List[int]] = {}
    for t in tags:
        by_tag.setdefault(t["tag"], []).append(t["off"])
    summary = []
    for tag, offs in by_tag.items():
        offs_sorted = sorted(offs)
        summary.append({"tag": tag, "count": len(offs_sorted), "first_off": offs_sorted[0]})
    summary.sort(key=lambda x: (-x["count"], x["tag"]))
    return summary

def dump_one_file(in_path: Path, out_path: Path) -> Dict[str, Any]:
    buf = in_path.read_bytes()

    btx_off = buf.find(b"BTX ")
    btx = None
    if btx_off != -1:
        try:
            btx = parse_btx(buf, btx_off)
        except Exception as e:
            btx = {"offset": btx_off, "error": str(e), "strings": []}

    lips = parse_lip_chunks(buf)

    data = {
        "file": str(in_path),
        "size": len(buf),
        "btx": btx,
        "lips": lips,
        "fourcc_summary": collect_fourcc(buf),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def iter_e_files(root: Path, recursive: bool) -> List[Path]:
    """Chỉ quét *.e (case-insensitive)."""
    out: List[Path] = []
    if root.is_file():
        if root.suffix.lower() == ".e":
            return [root]
        return []
    if not root.is_dir():
        return []

    if recursive:
        for dp, _, fnames in os.walk(root):
            for fn in fnames:
                p = Path(dp) / fn
                if p.suffix.lower() == ".e":
                    out.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() == ".e":
                out.append(p)

    out.sort()
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Batch dump *.e -> JSON (BTX/LIP), mirror subfolders when output is set.")
    ap.add_argument("input", nargs="?", help="Input folder (or single .e file). If omitted: interactive prompt.")
    ap.add_argument("-o", "--out", default=None, help="Output folder. Empty = write next to each .e.")
    ap.add_argument("-r", "--recursive", action="store_true", help="Scan recursively (default in interactive = yes).")
    args = ap.parse_args()

    if not args.input:
        in_str = input("Nhập thư mục chứa file .e (hoặc 1 file .e): ").strip().strip('"')
        out_str = input("Nhập thư mục output (để trống = xuất cạnh file): ").strip().strip('"')
        rec_str = input("Quét đệ quy? (y/n, mặc định y): ").strip().lower()
        args.input = in_str
        args.out = out_str if out_str else None
        args.recursive = (rec_str != "n")

    in_path = Path(args.input)
    files = iter_e_files(in_path, args.recursive)
    if not files:
        print("Không tìm thấy file *.e phù hợp.")
        return

    out_root = Path(args.out) if args.out else None

    ok = 0
    fail = 0
    for idx, fp in enumerate(files, 1):
        try:
            if out_root is None:
                out_json = fp.with_suffix(fp.suffix + ".json")
            else:
                # Luôn mirror subfolder khi input là folder
                if in_path.is_dir():
                    rel = fp.relative_to(in_path)
                    out_json = (out_root / rel).with_suffix(fp.suffix + ".json")
                else:
                    out_json = out_root / (fp.name + ".json")

            dump_one_file(fp, out_json)
            ok += 1
            print(f"[{idx}/{len(files)}] OK  : {fp} -> {out_json}")
        except Exception as e:
            fail += 1
            print(f"[{idx}/{len(files)}] FAIL: {fp} ({e})")

    print(f"\nDone. OK={ok} FAIL={fail} TOTAL={len(files)}")

if __name__ == "__main__":
    main()
