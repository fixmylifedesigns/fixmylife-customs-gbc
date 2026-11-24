import sys
from pyboy import PyBoy
from PIL import Image

rom, out = sys.argv[1], sys.argv[2]
pb = PyBoy(rom, window="null", cgb=True)
for _ in range(120):
    pb.tick()
img = pb.screen.image.convert("RGB")
img.resize((640, 576), Image.Resampling.NEAREST).save(out)
pb.stop(save=False)
print("ok", img.size)
