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


def save_avif_gpu(img: Image.Image, path: Path, quality: int = 80, speed: int = 6):
    """
    使用 ffmpeg GPU 编码 AVIF
    需安装 ffmpeg 并支持 av1_nvenc（NVIDIA GPU）
    img: PIL.Image
    path: 保存路径
    quality: 1-100
    speed: 0-10（越大越快，越小质量越高）
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", tmp_path,
        "-pix_fmt", "yuv420p",  # 转换为 yuv420p，编码器兼容
        "-c:v", "av1_nvenc",    # NVIDIA GPU编码器，若无则用 libaom-av1
        "-qp", str(quality),     # 质量参数，越小越高
        str(path)
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except Exception as e:
        print(f"ffmpeg avif 编码失败: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    

def main():
    img = Image.open("../cvAutoTrack/resource/无损GIMAP.webp")
    # CPU编码
    save_avif(img, Path("../cvAutoTrack/resource/GIMAP90.avif"), quality=90, speed=0)
    # GPU编码
    # save_avif_gpu(img, Path("../cvAutoTrack/resource/GIMAP_gpu.avif"), quality=95, speed=0)

if __name__ == "__main__":
    main()