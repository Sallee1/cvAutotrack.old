import os
from typing import Tuple,Dict
import cv2
import numpy as np
import math as m
import json
from pathlib import Path
import re

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
    center = tuple(gimap_json["center"])
    size = tuple(gimap_json["size"])
    align = gimap_json["align"]
    padding_col = tuple(gimap_json["padding_col"])
    
    width = img.shape[1]
    height = img.shape[0]
    # 根据对齐数据，将图片边缘补齐到给定的align
    left = origin[0] - ((origin[0] // size[0])) * size[0]
    left_pad = align - left % align
    if(left_pad == align):left_pad = 0
    left += left_pad
    
    right = ((width - origin[0]) // size[0]) * size[0]
    right_pad = align - right % align
    if(right_pad == align):right_pad = 0
    right += right_pad
    
    top = origin[1] - ((origin[1] // size[1])) * size[1]
    top_pad = align - top % align
    if(top_pad == align):top_pad = 0
    top += top_pad
    
    bottom = ((height - origin[1]) // size[1]) * size[1]
    bottom_pad = align - bottom % align
    if(bottom_pad == align):bottom_pad = 0
    bottom += bottom_pad
    # 补齐边缘
    img_pad = cv2.copyMakeBorder(img,top_pad,bottom_pad,left_pad,right_pad,cv2.BORDER_CONSTANT,value=padding_col)
    pad_width = img_pad.shape[1]
    pad_height = img_pad.shape[0]
    origin = (origin[0]+left_pad,origin[1]+top_pad)
    
    result = {}
    left_count = int(m.ceil(origin[0] / size[0]))
    rigth_count = int(m.ceil((width - origin[0]) / size[0]))
    top_count = int(m.ceil(origin[1] / size[1]))
    bottom_count = int(m.ceil((height - origin[1]) / size[1]))
    for r in range(-top_count,bottom_count):
        for c in range(-left_count,rigth_count):
            lt = (max(0,c * size[0] + origin[0]),max(0,r * size[1] + origin[1]))
            rb = (min(pad_width,(c + 1) * size[0] + origin[0]),min(pad_height,(r+1) * size[1] + origin[1]))
            img_roi = img_pad[lt[1]:rb[1],lt[0]:rb[0]]
            if(img_roi.shape[0] == 0 or img_roi.shape[1] == 0):
                continue
            result[(c,r)] = img_roi

    # 计算spatial参数
    spatial = {}
    spatial["origin"] = [origin[0],origin[1]]
    spatial["center"] = [center[0] + left_pad,center[1]+top_pad]
    spatial["img_size"] = [pad_width, pad_height]
    spatial["padding_col"] = list(padding_col)
    spatial["scale"] = gimap_json["scale"]
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
    align = gimap_json["align"]
    padding_col = tuple(gimap_json["padding_col"])
    webp_quality = gimap_json["webp_quality"]
    image_path = gimap_json["image"]
    image_filename = os.path.basename(image_path)
    image_filename = Path(image_filename).stem
    image_path = os.path.join("../cvat_rc_beta_input/",image_path)
    output_path = gimap_json["output"]
    output_path = os.path.join("../cvat_rc_beta_input/",output_path)
    
    input_img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if(input_img is None):
        raise ValueError("input image is invalid")
    split_imgs,spatial = split_map(input_img,gimap_json)
    
    for pos,img in split_imgs.items():
        cv2.imwrite(os.path.join(output_path,f"{image_filename}_{pos[0]}_{pos[1]}.webp"),img, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
    
    # 打开并解析mapper文件
    with open("../cvat_rc_beta_input/mapper.json",encoding="utf-8") as gimap_json_fp:
        mapper_json = json.load(gimap_json_fp)
    mapper_json["spatial"] = {}
    mapper_json["spatial"].update(spatial)
    
    with open(os.path.join(output_path, "mapper.json"), "w", encoding='utf-8') as mapper_json_fp:
        mapper_json_str = json.dumps(mapper_json, ensure_ascii=False, indent=4)
        
        # 简单使用正则表达式处理数组换行
        # 之前实现了自定义编码器但发现无法使用
        mapper_json_str = json_format(mapper_json_str)
        mapper_json_fp.write(mapper_json_str)
        
    