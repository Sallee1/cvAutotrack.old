from pathlib import Path
from PIL import Image
import numpy as np
import subprocess
import tempfile

def save_avif(img: Image.Image, path: Path, quality: int = 80, speed: int = 5):
    """
    将PIL图像保存为AVIF格式（CPU编码，Pillow自带）
    """
    img.save(path, format="AVIF", quality=quality, speed=speed)


def main():
    img = Image.open("../cvAutoTrack/resource/无损GIMAP.webp")
    # 转黑白
    img = img.convert("L")
    
    # CPU编码
    save_avif(img, Path("../cvAutoTrack/resource/GIMAP90.avif"), quality=90, speed=0)
    # GPU编码
    # save_avif_gpu(img, Path("../cvAutoTrack/resource/GIMAP_gpu.avif"), quality=95, speed=0)

if __name__ == "__main__":
    main()