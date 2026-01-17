#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dmpfont_edit_tool.py

Tool moi de ban edit font bang PNG.

Chay tool -> menu:
  1) Xuat DMP font: hoi duong dan .dmp, xuat 1 PNG trong suot cung ten.
  2) Nhap DMP font: hoi duong dan .dmp va .png, xuat DMP moi cung ten them _new.dmp.

Mac dinh da set dung theo ban test:
- PACK_BOTH_NIBBLES = True (ghi ca 2 nibble)
- INVERT = True (dao palette: stored = 15 - intended)

PNG export:
- Pixel glyph: mau trang (255,255,255) + alpha theo intended 0..15.
- Nen trong suot (alpha 0).

Yeu cau:
  pip install pillow
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw

# ---------------- DMP constants ----------------
DMP_FMT_EXPECTED = 9

# ---------------- Engine quirks ----------------
# NOTE:
# DMP co the luu gia tri 4-bit o low-nibble, high-nibble, hoac ca hai.
# File ban dua len hien tai co dang 0xFv (high nibble = 0xF, data nam o low).
# De tranh xuat PNG bi trong, tool se tu dong detect nibble dung va tu dong
# chon co can dao (invert) hay khong dua tren mau nen (gia tri chiem da so).
AUTO_INVERT = True


def detect_nibble_mode(lin: bytes) -> str:
    """Auto-detect where the 4-bit value lives.

    Returns:
      - 'both' : hi == lo for most pixels (packed both nibbles)
      - 'lo'   : data mainly in low nibble
      - 'hi'   : data mainly in high nibble
    """
    if not lin:
        return 'lo'

    uniq_lo = set()
    uniq_hi = set()
    same = 0
    n = 0
    # sample to keep it fast even for bigger textures
    step = max(1, len(lin) // 20000)  # cap ~20k samples
    for b in lin[::step]:
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        uniq_lo.add(lo)
        uniq_hi.add(hi)
        if lo == hi:
            same += 1
        n += 1

    if n and (same / n) >= 0.90:
        return 'both'

    # If one nibble is basically constant, the other is likely the payload.
    if len(uniq_hi) == 1 and len(uniq_lo) > 1:
        return 'lo'
    if len(uniq_lo) == 1 and len(uniq_hi) > 1:
        return 'hi'

    # Fallback: pick the one with more variety.
    return 'lo' if len(uniq_lo) >= len(uniq_hi) else 'hi'


def get_stored_from_byte(b: int, mode: str) -> int:
    if mode == 'hi':
        return (b >> 4) & 0x0F
    # 'lo' or 'both'
    return b & 0x0F


def set_stored_into_byte(old_b: int, stored: int, mode: str) -> int:
    stored &= 0x0F
    if mode == 'both':
        return stored | (stored << 4)
    if mode == 'hi':
        return (old_b & 0x0F) | (stored << 4)
    # mode == 'lo'
    return (old_b & 0xF0) | stored


def detect_invert_needed(lin: bytes, mode: str) -> bool:
    """Auto decide invert.

    Idea: background usually dominates. We want background to become alpha=0 in PNG.
    - If most common stored value is 0 -> likely background=0 -> invert False.
    - If most common stored value is 15 -> likely background=15 -> invert True.
    """
    if not lin:
        return False
    step = max(1, len(lin) // 20000)
    hist = [0] * 16
    for b in lin[::step]:
        v = get_stored_from_byte(b, mode)
        hist[v] += 1
    # compare dominance of 0 vs 15
    return hist[15] > hist[0]

# Grid overlay color (unique-ish so we can ignore it on import)
GRID_RGBA = (0, 255, 0, 80)
GRID_TOL_RGB = 6
GRID_TOL_A = 12


def u32le(b: bytes) -> int:
    return struct.unpack('<I', b)[0]


def p32le(x: int) -> bytes:
    return struct.pack('<I', x)


def morton2(x: int, y: int, bits: int) -> int:
    n = 0
    for i in range(bits):
        n |= ((x >> i) & 1) << (2 * i)
        n |= ((y >> i) & 1) << (2 * i + 1)
    return n


def swizzle_tile_linear_pixel_morton(lin: bytes, w: int, h: int) -> bytearray:
    if w % 8 or h % 8:
        raise ValueError('w/h must be multiple of 8')
    tiles_x = w // 8
    tiles_y = h // 8
    out = bytearray(w * h)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_index = ty * tiles_x + tx
            base = tile_index * 64
            for py in range(8):
                for px in range(8):
                    dst = base + morton2(px, py, 3)
                    out[dst] = lin[(ty * 8 + py) * w + (tx * 8 + px)]
    return out


def unswizzle_tile_linear_pixel_morton(swz: bytes, w: int, h: int) -> bytearray:
    if w % 8 or h % 8:
        raise ValueError('w/h must be multiple of 8')
    tiles_x = w // 8
    tiles_y = h // 8
    out = bytearray(w * h)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_index = ty * tiles_x + tx
            base = tile_index * 64
            for py in range(8):
                for px in range(8):
                    src = base + morton2(px, py, 3)
                    out[(ty * 8 + py) * w + (tx * 8 + px)] = swz[src]
    return out


def parse_dmp(path: Path) -> Tuple[int, int, int, bytes]:
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError('DMP qua nho')
    fmt = u32le(data[0:4])
    w = u32le(data[4:8])
    h = u32le(data[8:12])
    pix = data[12:]
    if len(pix) != w * h:
        raise ValueError(f'DMP pixel data sai kich thuoc: {len(pix)} != {w*h}')
    return fmt, w, h, pix


def detect_active_nibble(swz: bytes) -> str:
    """Detect which nibble carries the meaningful 4-bit values.

    Returns: 'lo', 'hi', or 'both'
    - 'both': most bytes have lo==hi (packed)
    - 'lo'  : low nibble varies while high nibble is near-constant
    - 'hi'  : high nibble varies while low nibble is near-constant
    """
    # sample up to first 64k bytes (full sheet typically 65536)
    sample = swz[: min(len(swz), 65536)]
    lo_set = set()
    hi_set = set()
    eq_cnt = 0
    for b in sample:
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        lo_set.add(lo)
        hi_set.add(hi)
        if lo == hi:
            eq_cnt += 1

    if len(sample) > 0 and (eq_cnt / len(sample)) >= 0.95:
        return 'both'
    # If one nibble is almost constant and the other varies -> pick varying
    if len(hi_set) <= 2 and len(lo_set) > len(hi_set):
        return 'lo'
    if len(lo_set) <= 2 and len(hi_set) > len(lo_set):
        return 'hi'
    # Fallback: choose the nibble with more unique values
    return 'lo' if len(lo_set) >= len(hi_set) else 'hi'


def get_nibble(b: int, which: str) -> int:
    if which == 'hi':
        return (b >> 4) & 0x0F
    # default 'lo'
    return b & 0x0F


def set_nibble_keep_other(orig_b: int, which: str, v: int) -> int:
    v &= 0x0F
    if which == 'hi':
        return (orig_b & 0x0F) | (v << 4)
    # 'lo'
    return (orig_b & 0xF0) | v


def infer_cell_size_from_name(p: Path) -> Optional[int]:
    m = re.search(r'font(\d+)x\1', p.stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def draw_grid(img: Image.Image, cell: int) -> None:
    if cell <= 0:
        return
    w, h = img.size
    d = ImageDraw.Draw(img)
    # vertical
    for x in range(0, w + 1, cell):
        d.line([(x, 0), (x, h)], fill=GRID_RGBA, width=1)
    # horizontal
    for y in range(0, h + 1, cell):
        d.line([(0, y), (w, y)], fill=GRID_RGBA, width=1)


def is_grid_pixel(rgba: Tuple[int, int, int, int]) -> bool:
    r, g, b, a = rgba
    gr, gg, gb, ga = GRID_RGBA
    if abs(a - ga) > GRID_TOL_A:
        return False
    if abs(r - gr) > GRID_TOL_RGB:
        return False
    if abs(g - gg) > GRID_TOL_RGB:
        return False
    if abs(b - gb) > GRID_TOL_RGB:
        return False
    return True


def export_dmp_to_png(dmp_path: Path) -> Path:
    fmt, w, h, swz = parse_dmp(dmp_path)
    if fmt != DMP_FMT_EXPECTED:
        print(f'[WARN] fmt={fmt} (expected {DMP_FMT_EXPECTED}) - van thu xuat')

    lin = unswizzle_tile_linear_pixel_morton(swz, w, h)
    mode = detect_nibble_mode(lin)
    nib = 'lo' if mode == 'both' else mode
    invert = detect_invert_needed(lin, nib) if AUTO_INVERT else False
    print(f'[INFO] Nibble mode: {mode} | Invert: {invert}')

    # Convert to RGBA (white + alpha)
    out = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    px = out.load()

    for y in range(h):
        base = y * w
        for x in range(w):
            b = lin[base + x]
            stored = get_stored_from_byte(b, nib)
            intended = 15 - stored if invert else stored
            if intended <= 0:
                px[x, y] = (255, 255, 255, 0)
            else:
                alpha = int(round(intended * 255 / 15))
                px[x, y] = (255, 255, 255, alpha)

    # Khong ve luoi nua

    png_path = dmp_path.with_suffix('.png')
    out.save(png_path)
    return png_path


def import_png_to_dmp(dmp_path: Path, png_path: Path) -> Path:
    fmt, w, h, swz_old = parse_dmp(dmp_path)
    if fmt != DMP_FMT_EXPECTED:
        print(f'[WARN] fmt={fmt} (expected {DMP_FMT_EXPECTED}) - van thu nhap')

    img = Image.open(png_path).convert('RGBA')
    if img.size != (w, h):
        raise ValueError(f'PNG size {img.size} khong khop DMP {w}x{h}')

    pix = list(img.getdata())

    # Keep original other-nibble to preserve format (vd 0xFv)
    lin_old = unswizzle_tile_linear_pixel_morton(swz_old, w, h)
    mode = detect_nibble_mode(lin_old)
    nib = 'lo' if mode == 'both' else mode
    invert = detect_invert_needed(lin_old, nib) if AUTO_INVERT else False
    print(f'[INFO] Nibble mode: {mode} | Invert: {invert}')
    lin = bytearray(w * h)
    for i, rgba in enumerate(pix):
        if is_grid_pixel(rgba):
            alpha = 0
        else:
            alpha = rgba[3]

        intended = int(round(alpha * 15 / 255))
        if intended < 0:
            intended = 0
        if intended > 15:
            intended = 15

        stored = 15 - intended if invert else intended
        ob = lin_old[i]
        lin[i] = set_stored_into_byte(ob, stored, mode)

    swz = swizzle_tile_linear_pixel_morton(bytes(lin), w, h)
    out_bytes = p32le(fmt) + p32le(w) + p32le(h) + bytes(swz)

    out_path = dmp_path.with_name(dmp_path.stem + '_new' + dmp_path.suffix)
    out_path.write_bytes(out_bytes)
    return out_path


def main() -> None:
    print('=== DMP Font PNG Editor Tool ===')
    print('1) Xuat DMP -> PNG (trong suot)')
    print('2) Nhap PNG -> DMP (_new)')
    choice = input('Chon (1/2): ').strip()

    if choice == '1':
        p = input('Nhap duong dan DMP: ').strip().strip('"')
        dmp_path = Path(p)
        if not dmp_path.exists():
            raise SystemExit('DMP khong ton tai')
        png = export_dmp_to_png(dmp_path)
        print(f'[OK] Da xuat: {png}')

    elif choice == '2':
        p1 = input('Nhap duong dan DMP goc: ').strip().strip('"')
        p2 = input('Nhap duong dan PNG edit: ').strip().strip('"')
        dmp_path = Path(p1)
        png_path = Path(p2)
        if not dmp_path.exists():
            raise SystemExit('DMP khong ton tai')
        if not png_path.exists():
            raise SystemExit('PNG khong ton tai')
        outp = import_png_to_dmp(dmp_path, png_path)
        print(f'[OK] Da tao: {outp}')

    else:
        raise SystemExit('Lua chon khong hop le')


if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        if str(e):
            print('[ERROR]', e)
    except Exception as e:
        print('[EXCEPTION]', e)
        raise