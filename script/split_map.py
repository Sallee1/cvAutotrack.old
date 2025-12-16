import os
from typing import Tuple,Dict
import cv2
import numpy as np
import math as m
import json
from pathlib import Path
import re
import hashlib
import shutil
import hashlib
from PIL import Image

def split_map(img:np.ndarray,
              gimap_json:Dict,
              )->Tuple[Dict[Tuple[int,int],np.ndarray],Dict]:
    """
    输入图像，按照参数裁剪为图像组
    img: 输入的大图
    gimap_json 写有裁剪参数的json
    return 图像的下标以及裁剪的子图，以及计算出来的spatial参数
    """
    origin = tuple(gimap_json["origin"])
    size = tuple(gimap_json["size"])
    padding_col = tuple(gimap_json["padding_col"])
    
    width = img.shape[1]
    height = img.shape[0]
    
    result = {}
    left_count = int(m.ceil(origin[0] / size[0]))
    rigth_count = int(m.ceil((width - origin[0]) / size[0]))
    top_count = int(m.ceil(origin[1] / size[1]))
    bottom_count = int(m.ceil((height - origin[1]) / size[1]))
    for r in range(-top_count,bottom_count):
        for c in range(-left_count,rigth_count):
            lt = (max(0,c * size[0] + origin[0]),max(0,r * size[1] + origin[1]))
            rb = (min(width,(c + 1) * size[0] + origin[0]),min(height,(r+1) * size[1] + origin[1]))
            img_roi = img[lt[1]:rb[1],lt[0]:rb[0]]
            if(img_roi.shape[0] == 0 or img_roi.shape[1] == 0):
                continue
            result[(c,r)] = img_roi

    # 计算spatial参数
    spatial = {}
    spatial["origin"] = gimap_json["origin"]
    spatial["center"] = gimap_json["center"]
    spatial["img_size"] = [width, height]
    spatial["tile_range_x"] = [-left_count,rigth_count - 1]
    spatial["tile_range_y"] = [-top_count,bottom_count - 1]
    spatial["tile_size"] = [size[0],size[1]]
    return result,spatial

def json_format(json_str:str):
    """
    将数组合并为一行
    """
    json_str = re.sub(r'(-?\d+)[\s\n]*,[\s\n]*(-?\d+),[\s\n]*',r"\1, \2, ",json_str)
    json_str = re.sub(r"\[[\n\s]*(-?\d+)", r"[\1",json_str)
    json_str = re.sub(r'(-?\d+)[\s\n]*,[\s\n]*(-?\d+)[\s\n]*\]',r"\1, \2]",json_str)
    json_str = re.sub(r"(-?\d+)[\n\s]*]",r"\1]",json_str )
    return json_str

if __name__ == "__main__":
    with open("../cvat_rc_beta_input/gimap.json") as gimap_json_fp:
        gimap_json = json.load(gimap_json_fp)
    if(gimap_json is None):
        raise ValueError("gimap.json format is invalid")
    
    origin = tuple(gimap_json["origin"])
    size = tuple(gimap_json["size"])
    padding_col = tuple(gimap_json["padding_col"])
    avif_quality = gimap_json["avif_quality"]
    avif_speed = gimap_json["avif_speed"]
    image_path = gimap_json["image"]
    image_filename = os.path.basename(image_path)
    image_filename = Path(image_filename).stem
    image_path = os.path.join("../cvat_rc_beta_input/",image_path)
    output_dir = gimap_json["output"]
    output_dir = os.path.join("../cvat_rc_beta_input/",output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    force_update = gimap_json.get("force_update", False)
    tile_hash_path = os.path.join("../cvat_rc_beta_input/tile_hashed.json")
    # 读取已存在的哈希缓存
    if os.path.exists(tile_hash_path):
        with open(tile_hash_path, "r", encoding="utf-8") as f:
            tile_hash_dict = json.load(f)
    else:
        tile_hash_dict = {}
    
    input_img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if(input_img is None):
        raise ValueError("input image is invalid")
    split_imgs,spatial = split_map(input_img,gimap_json)
    
    total = len(split_imgs)
    current = 0
    for pos, img in split_imgs.items():
        img_bytes = img.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        out_base = f"{image_filename}_{pos[0]}_{pos[1]}"
        avif_path = os.path.join(output_dir, out_base + ".avif")
        pos_key = f"{pos[0]}_{pos[1]}"
        # 检查 tile_hashed.json 是否已有该哈希，且未设置全量更新
        if not force_update and tile_hash_dict.get(pos_key) == img_hash:
            continue
        # 更新哈希缓存
        tile_hash_dict[pos_key] = img_hash
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img)
        # 保存压缩AVIF
        img_pil.save(avif_path, format="AVIF", quality=avif_quality, speed=avif_speed)
        # 保存 tile_hashed.json
        with open(tile_hash_path, "w", encoding="utf-8") as f:
            json.dump(tile_hash_dict, f, ensure_ascii=False, indent=2)
            # cv2.imwrite(out_path,img, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])

        print(f"编码中({current}/{total}),正在编码{avif_path}",end="\r")
        current += 1
    
    # 打开并解析mapper文件
    with open("../cvat_rc_beta_input/mapper.json",encoding="utf-8") as gimap_json_fp:
        mapper_json = json.load(gimap_json_fp)
    mapper_json["spatial"] = {}
    mapper_json["spatial"].update(spatial)
    
    with open(os.path.join(output_dir, "mapper.json"), "w", encoding='utf-8') as mapper_json_fp:
        mapper_json_str = json.dumps(mapper_json, ensure_ascii=False, indent=4)
        
        # 简单使用正则表达式处理数组换行
        # 之前实现了自定义编码器但发现无法使用
        mapper_json_str = json_format(mapper_json_str)
        mapper_json_fp.write(mapper_json_str)
        
    