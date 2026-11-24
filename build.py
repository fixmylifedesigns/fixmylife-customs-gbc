#!/usr/bin/env python3
"""Build a Game Boy Color ROM that shows a full-screen 160x144 image.

Image -> 360 tiles (20x18), 8 BG palettes x 4 colors (15-bit RGB),
each tile picks its best palette. Tiles 0-255 live in VRAM bank 0,
tiles 256-359 in VRAM bank 1 (selected per tile via the attribute map).
The SM83 code is hand-assembled below, so no external toolchain is needed.

usage: python3 build.py input.png out.gbc [preview.png]
"""
import sys
import numpy as np
from PIL import Image

W, H = 160, 144
TW, TH = W // 8, H // 8
NPAL, NCOL = 8, 4
WEIGHT = np.array([2.0, 4.0, 3.0])  # rough perceptual weighting (R,G,B)
rng = np.random.default_rng(1234)


# ---------------------------------------------------------------- quantizer
def load(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((W, H), Image.Resampling.BOX)
    return np.asarray(im, dtype=np.float64)


def tiles_of(img):
    t = img.reshape(TH, 8, TW, 8, 3).transpose(0, 2, 1, 3, 4)
    return t.reshape(TH * TW, 64, 3)


def dist(px, pal):
    # px: (..., 3), pal: (k, 3) -> (..., k)
    d = px[..., None, :] - pal
    return (d * d * WEIGHT).sum(-1)


def kmeans(px, k, iters=12, init=None):
    px = px.reshape(-1, 3)
    uniq = np.unique(px, axis=0)
    if len(uniq) <= k:
        c = np.zeros((k, 3))
        c[: len(uniq)] = uniq
        c[len(uniq):] = uniq[0]
        return c
    if init is None:
        # k-means++ style seeding
        c = [px[rng.integers(len(px))]]
        for _ in range(1, k):
            dd = dist(px, np.array(c)).min(-1)
            if dd.sum() == 0:
                c.append(px[rng.integers(len(px))])
            else:
                c.append(px[rng.choice(len(px), p=dd / dd.sum())])
        c = np.array(c, dtype=np.float64)
    else:
        c = init.copy()
    for _ in range(iters):
        lab = dist(px, c).argmin(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = px[m].mean(0)
    return c


def snap(pal):
    # to 5-bit per channel, return both 5-bit ints and the 8-bit equivalent
    q = np.clip(np.round(pal / 255 * 31), 0, 31).astype(int)
    return q, q * 255.0 / 31


def tile_err(tiles, pal):
    # tiles (n,64,3), pal (4,3) -> (n,)
    return dist(tiles, pal).min(-1).sum(-1)


def quantize(img, restarts=30, rounds=30):
    tiles = tiles_of(img)
    n = len(tiles)
    best = None
    for r in range(restarts):
        # init: cluster tiles by mean colour + spread
        feat = np.concatenate([tiles.mean(1), tiles.std(1)], 1)
        cent = feat[rng.choice(n, NPAL, replace=False)]
        for _ in range(10):
            lab = ((feat[:, None] - cent) ** 2).sum(-1).argmin(-1)
            for j in range(NPAL):
                if (lab == j).any():
                    cent[j] = feat[lab == j].mean(0)
        pals = np.array([kmeans(tiles[lab == j] if (lab == j).any() else tiles, NCOL)
                         for j in range(NPAL)])
        for _ in range(rounds):
            errs = np.stack([tile_err(tiles, p) for p in pals], 1)  # (n, 8)
            lab = errs.argmin(1)
            for j in range(NPAL):
                m = lab == j
                if not m.any():
                    # steal the worst-served tile
                    w = errs[np.arange(n), lab].argmax()
                    m = np.zeros(n, bool)
                    m[w] = True
                    lab[w] = j
                pals[j] = kmeans(tiles[m], NCOL, iters=6, init=pals[j])
        q5 = []
        q8 = []
        for p in pals:
            a, b = snap(p)
            q5.append(a)
            q8.append(b)
        q8 = np.array(q8)
        errs = np.stack([tile_err(tiles, p) for p in q8], 1)
        lab = errs.argmin(1)
        total = errs[np.arange(n), lab].sum()
        print(f"  restart {r}: error {total:,.0f}")
        if best is None or total < best[0]:
            best = (total, np.array(q5), q8, lab)
    _, q5, q8, lab = best
    idx = np.stack([dist(tiles[i], q8[lab[i]]).argmin(-1) for i in range(n)])  # (n,64)
    return q5, q8, lab, idx


# ---------------------------------------------------------------- encoders
def encode_tile(ix):
    ix = ix.reshape(8, 8)
    out = bytearray()
    for row in ix:
        lo = hi = 0
        for x, v in enumerate(row):
            lo |= (v & 1) << (7 - x)
            hi |= ((v >> 1) & 1) << (7 - x)
        out += bytes([lo, hi])
    return bytes(out)


def encode_pals(q5):
    out = bytearray()
    for p in q5:
        for r, g, b in p:
            w = int(r) | int(g) << 5 | int(b) << 10
            out += bytes([w & 0xFF, w >> 8])
    return bytes(out)


# ---------------------------------------------------------------- assembler
class Asm:
    def __init__(self, base):
        self.base = base
        self.code = bytearray()
        self.labels = {}
        self.fix = []  # (offset, label, kind)

    @property
    def pc(self):
        return self.base + len(self.code)

    def label(self, name):
        self.labels[name] = self.pc

    def b(self, *bs):
        self.code += bytes(bs)

    def w16(self, op, val):  # opcode + 16-bit immediate or label
        self.code.append(op)
        if isinstance(val, str):
            self.fix.append((len(self.code), val, "abs"))
            self.code += b"\0\0"
        else:
            self.code += bytes([val & 0xFF, val >> 8])

    def jr(self, op, lbl):
        self.code.append(op)
        self.fix.append((len(self.code), lbl, "rel"))
        self.code.append(0)

    def resolve(self, extra):
        lab = {**self.labels, **extra}
        for off, name, kind in self.fix:
            t = lab[name]
            if kind == "abs":
                self.code[off:off + 2] = bytes([t & 0xFF, t >> 8])
            else:
                rel = t - (self.base + off + 1)
                assert -128 <= rel <= 127, name
                self.code[off] = rel & 0xFF
        return bytes(self.code)


def program(n_bank1_bytes):
    a = Asm(0x150)
    a.b(0xF3)                       # di
    a.w16(0x31, 0xFFFE)             # ld sp,$FFFE
    a.label("wait_vbl")
    a.b(0xF0, 0x44)                 # ldh a,[LY]
    a.b(0xFE, 144)                  # cp 144
    a.jr(0x38, "wait_vbl")          # jr c,wait_vbl
    a.b(0xAF, 0xE0, 0x40)           # xor a / ldh [LCDC],a  (LCD off)
    a.b(0xE0, 0x4F)                 # ldh [VBK],a  -> bank 0
    a.w16(0x21, "TILES0"); a.w16(0x11, 0x8000); a.w16(0x01, 4096); a.w16(0xCD, "memcpy")
    a.w16(0x21, "MAP");    a.w16(0x11, 0x9800); a.w16(0x01, 32 * TH); a.w16(0xCD, "memcpy")
    a.b(0x3E, 1, 0xE0, 0x4F)        # ld a,1 / ldh [VBK],a -> bank 1
    a.w16(0x21, "TILES1"); a.w16(0x11, 0x8000); a.w16(0x01, n_bank1_bytes); a.w16(0xCD, "memcpy")
    a.w16(0x21, "ATTR");   a.w16(0x11, 0x9800); a.w16(0x01, 32 * TH); a.w16(0xCD, "memcpy")
    a.b(0xAF, 0xE0, 0x4F)           # back to bank 0
    a.b(0x3E, 0x80, 0xE0, 0x68)     # ld a,$80 / ldh [BCPS],a (auto-increment)
    a.w16(0x21, "PAL")
    a.b(0x06, 64)                   # ld b,64
    a.label("pal_loop")
    a.b(0x2A, 0xE0, 0x69)           # ld a,[hl+] / ldh [BCPD],a
    a.b(0x05)                       # dec b
    a.jr(0x20, "pal_loop")          # jr nz
    a.b(0xAF, 0xE0, 0x42, 0xE0, 0x43)  # SCY=SCX=0
    a.b(0x3E, 0x91, 0xE0, 0x40)     # LCDC: on, tiles $8000, map $9800, BG on
    a.b(0x3E, 0x01, 0xE0, 0xFF)     # IE = VBlank
    a.b(0xAF, 0xE0, 0x0F)           # IF = 0
    a.b(0xFB)                       # ei
    a.label("idle")
    a.b(0x76)                       # halt
    a.jr(0x18, "idle")              # jr idle
    a.label("memcpy")               # hl=src de=dst bc=len
    a.b(0x2A, 0x12, 0x13, 0x0B, 0x78, 0xB1)
    a.jr(0x20, "memcpy")
    a.b(0xC9)
    return a


NINTENDO_LOGO = bytes.fromhex(
    "CEED6666CC0D000B03730083000C000D0008111F8889000E"
    "DCCC6EE6DDDDD999BBBB67636E0EECCCDDDC999FBBB9333E")


def build_rom(q5, lab, idx, title=b"FMLCUSTOMS"):
    n = len(lab)
    tiles = [encode_tile(idx[i]) for i in range(n)]
    t0 = b"".join(tiles[:256])
    t0 += b"\0" * (4096 - len(t0))
    t1 = b"".join(tiles[256:]) or b"\0" * 16
    mp = bytearray(32 * TH)
    at = bytearray(32 * TH)
    for ty in range(TH):
        for tx in range(TW):
            i = ty * TW + tx
            mp[ty * 32 + tx] = i if i < 256 else i - 256
            at[ty * 32 + tx] = int(lab[i]) | (0x08 if i >= 256 else 0)
    pal = encode_pals(q5)

    asm = program(len(t1))
    data_base = 0x400
    blobs = [("TILES0", t0), ("TILES1", t1), ("MAP", bytes(mp)), ("ATTR", bytes(at)), ("PAL", pal)]
    addrs, off = {}, data_base
    for name, blob in blobs:
        addrs[name] = off
        off += len(blob)
    assert off <= 0x8000, "ROM overflow"
    code = asm.resolve(addrs)
    assert 0x150 + len(code) <= data_base

    rom = bytearray(0x8000)
    rom[0x40] = 0xD9                         # VBlank vector: reti
    rom[0x100:0x104] = bytes([0x00, 0xC3, 0x50, 0x01])  # nop / jp $0150
    rom[0x104:0x134] = NINTENDO_LOGO
    rom[0x134:0x134 + len(title)] = title    # title (max 15 with CGB flag)
    rom[0x143] = 0xC0                        # CGB only
    rom[0x144:0x146] = b"00"
    rom[0x147] = 0x00                        # ROM only
    rom[0x148] = 0x00                        # 32 KiB
    rom[0x149] = 0x00
    rom[0x14A] = 0x01
    rom[0x14B] = 0x33
    rom[0x150:0x150 + len(code)] = code
    for name, blob in blobs:
        rom[addrs[name]:addrs[name] + len(blob)] = blob
    c = 0
    for i in range(0x134, 0x14D):
        c = (c - rom[i] - 1) & 0xFF
    rom[0x14D] = c
    g = (sum(rom) - rom[0x14E] - rom[0x14F]) & 0xFFFF
    rom[0x14E], rom[0x14F] = g >> 8, g & 0xFF
    return bytes(rom)


def preview(q8, lab, idx, path):
    out = np.zeros((H, W, 3), np.uint8)
    for i in range(len(lab)):
        ty, tx = divmod(i, TW)
        out[ty * 8:ty * 8 + 8, tx * 8:tx * 8 + 8] = q8[lab[i]][idx[i]].reshape(8, 8, 3).round()
    Image.fromarray(out).resize((W * 4, H * 4), Image.Resampling.NEAREST).save(path)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    img = load(src)
    print("quantizing...")
    q5, q8, lab, idx = quantize(img)
    rom = build_rom(q5, lab, idx)
    open(dst, "wb").write(rom)
    print(f"wrote {dst} ({len(rom)} bytes)")
    if len(sys.argv) > 3:
        preview(q8, lab, idx, sys.argv[3])
