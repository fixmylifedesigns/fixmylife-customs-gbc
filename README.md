# FixMyLife Customs — Game Boy Color ROM

A Game Boy Color ROM that displays the **FixMyLife Customs** title screen, made from art generated on Higgsfield.

![Emulator screenshot](emu_shot.png)

## Download

- **ROM:** [`fixmylife_customs.gbc`](https://github.com/fixmylifedesigns/fixmylife-customs-gbc/raw/main/fixmylife_customs.gbc) (32 KiB, CGB-only, no mapper)
- Source art: [`title_source.png`](title_source.png)

The ROM and images are built and committed by the **Build ROM** GitHub Action (`.github/workflows/build.yml`).

Runs on emulators (mGBA, SameBoy, PyBoy, RetroArch) and on flash carts such as the EverDrive GB or a GBxCart-flashed repro cart.

## How it works

- The 1008×896 art is downscaled to 160×144 (20×18 = 360 tiles).
- `build.py` fits **8 background palettes × 4 colours** (15-bit RGB) and picks the best palette for each 8×8 tile, using k-means with several restarts.
- Tiles 0–255 go in VRAM bank 0 and tiles 256–359 in VRAM bank 1. The attribute map selects the bank and the palette for each tile.
- The SM83 code that loads the tiles, map, attributes and palettes, turns the LCD on and idles with `halt` is assembled by hand inside `build.py`. You don't need RGBDS or GBDK. The header and global checksums are filled in for real hardware.

## Rebuild locally

```sh
pip install numpy pillow
curl -sSL -o title_source.png "$(head -n1 source_url.txt)"
python3 build.py title_source.png fixmylife_customs.gbc preview.png

# optional: screenshot in an emulator
pip install pyboy
python3 test_rom.py fixmylife_customs.gbc emu_shot.png
```

Swap in any image to put it on a GBC screen (it gets resized to 160×144). Pixel art with flat colours converts best.
