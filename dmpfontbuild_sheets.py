#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dmpfont_create_tool.py

Tool tao DMP font tu TTF.

Chay tool -> hoi duong dan .ttf va size, xuat DMP theo bang ky tu.

Mac dinh:
- Nibble mode: 'lo' (data o low nibble, high=0xF)
- Invert: False (stored = intended)

DMP format:
- fmt=9
- w=h=256 (16x16 cells, moi cell 16x16 px)
- Ten: font<size>x<size>_10.dmp

Bang ky tu (256 chars, bao gom space):
 !"#$%&'()*+,-./
0123456789:;<=>?
@ABCDEFGHIJKLMNO
PQRSTUVWXYZ\abcd
efghijklmnopqrst
uvwxyz ¡©ª«®º»¿À
ÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔ
ÕÖÙÚÛÜßàáâäçèéêë
ìíîïñòóôõöùúûüŒœ
đĐảẢãÃạẠẻẺẽẼẹẸỉỈĩĨ
ịỊỏỎọỌủỦũŨụỤýÝỳỲỷỶỹ
ỸỵỴăĂắẮằẰẳẲẵẴặẶấẤầẦ
ẩẨẫẪậẬếẾềỀểỂễỄệỆốỐồ
ỒổỔỗỖộỘơƠớỚờỜởỞỡỠợỢ
ưƯứỨừỪửỬữỮựỰ

Yeu cau:
  pip install pillow
"""

import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------- DMP constants ----------------
DMP_FMT_EXPECTED = 9

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

def set_stored_into_byte(old_b: int, stored: int, mode: str) -> int:
    stored &= 0x0F
    if mode == 'both':
        return stored | (stored << 4)
    if mode == 'hi':
        return (old_b & 0x0F) | (stored << 4)
    # mode == 'lo'
    return (old_b & 0xF0) | stored

def create_dmp_from_ttf(ttf_path: Path, size: int) -> Path:
    charset = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ\\abcdefghijklmnopqrstuvwxyz ¡©ª«®º»¿ÀÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜßàáâäçèéêëìíîïñòóôõöùúûüŒœđĐảẢãÃạẠẻẺẽẼẹẸỉỈĩĨịỊỏỎọỌủỦũŨụỤýÝỳỲỷỶỹỸỵỴăĂắẮằẰẳẲẵẴặẶấẤầẦẩẨẫẪậẬếẾềỀểỂễỄệỆốỐồỒổỔỗỖộỘơƠớỚờỜởỞỡỠợỢưƯứỨừỪửỬữỮựỰ'
    if len(charset) < 256:
        charset += '\x00' * (256 - len(charset))  # Pad with null if short
    charset = charset[:256]  # Truncate if longer

    w = 256
    h = 256
    fmt = DMP_FMT_EXPECTED
    cell = 16

    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    font = ImageFont.truetype(str(ttf_path), size)
    draw = ImageDraw.Draw(img)

    ascent, descent = font.getmetrics()
    padding = 2

    for i, char in enumerate(charset):
        if char == '\x00':
            continue  # Skip null char
        row = i // 16
        col = i % 16
        px = col * cell + padding  # Cach left padding px
        py = row * cell
        baseline_y = py + padding + ascent  # Baseline at py + padding + ascent, so top at py + padding
        draw.text((px, baseline_y), char, font=font, fill=(255, 255, 255, 255), anchor="ls")

    pix = list(img.getdata())

    mode = 'lo'
    invert = False
    lin = bytearray(w * h)
    for i, rgba in enumerate(pix):
        alpha = rgba[3]
        intended = int(round(alpha * 15 / 255))
        intended = max(0, min(15, intended))
        stored = 15 - intended if invert else intended
        ob = 0xF0  # High nibble = F
        lin[i] = set_stored_into_byte(ob, stored, mode)

    swz = swizzle_tile_linear_pixel_morton(bytes(lin), w, h)
    out_bytes = p32le(fmt) + p32le(w) + p32le(h) + bytes(swz)

    out_path = Path(f'font{size}x{size}_10.dmp')
    out_path.write_bytes(out_bytes)
    return out_path

def main() -> None:
    print('=== DMP Font Create Tool ===')
    p = input('Nhap duong dan TTF: ').strip().strip('"')
    s = input('Nhap size: ').strip()
    size = int(s)
    ttf_path = Path(p)
    if not ttf_path.exists():
        raise SystemExit('TTF khong ton tai')
    outp = create_dmp_from_ttf(ttf_path, size)
    print(f'[OK] Da tao: {outp}')

if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        if str(e):
            print('[ERROR]', e)
    except Exception as e:
        print('[EXCEPTION]', e)
        raise