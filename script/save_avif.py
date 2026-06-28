from pathlib import Path
from PIL import Image
import numpy as np
import subprocess
import os
import glob
import tempfile

def save_avif(img: Image.Image, path: Path, quality: int = 80, speed: int = 5):
    """
    将PIL图像保存为AVIF格式（CPU编码，Pillow自带）
    """
    img.save(path, format="AVIF", quality=quality, speed=speed)


def save_avif_png_dir(path: str, out_path, quality: int = 80, speed: int = 5, bg_color: str = "white"):
    """
    将PNG目录中的图片保存为AVIF格式
    bg_color: 透明区域的填充底色，如"white"、"black"、"gray"等
    """
    png_files = glob.glob(f"{path}/*.png")
    for png_file in png_files:
        img = Image.open(png_file)
        # 如果有透明度，先用指定底色填充，避免直接转黑白出现杂色背景
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, bg_color)
            background.paste(img, mask=img.split()[3])  # 用 alpha 通道作 mask
            img = background
        elif img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, bg_color)
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode == "LA":
            background = Image.new("L", img.size, bg_color)
            background.paste(img, mask=img.split()[1])
            img = background
        # 转黑白
        img = img.convert("L")
        file_name, file_ext = os.path.splitext(os.path.basename(png_file))
        
        out_file_name = f"{out_path}/{file_name}.avif"
        save_avif(img, Path(out_file_name))
        print(f'"{out_file_name}" saved')

def main():
    # img = Image.open("../cvAutoTrack/resource/无损GIMAP.webp")
    # # 转黑白
    # img = img.convert("L")
    
    # # CPU编码
    # save_avif(img, Path("../cvAutoTrack/resource/GIMAP90.avif"), quality=90, speed=0)
    save_avif_png_dir("../cvat_rc_beta/assets","../cvat_rc_beta/assets",90,0,bg_color='black')

if __name__ == "__main__":
    main()