import os, hashlib
import requests
from tqdm import tqdm
import argparse, os, sys, datetime, glob, importlib

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def get_obj_from_str(string, reload=False):
    # 从右侧开始分割字符串
    module, cls = string.rsplit(".", 1)

    # 强制重新加载模块，以防模块代码已经更改，是一种保险措施
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    # 获取类对象
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    # 检查配置字典中是否包含 'target' 键
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")    
    
    # 调用 get_obj_from_str 函数获取类对象
    # 然后使用配置字典中的 'params' 键（如果存在），如果不存在，则使用空字典
    # ** 运算符用于将字典解包，将其键值对作为关键字参数，传递给类的构造函数
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def md5_hash(path):
    # 以二进制模式读取文件内容
    with open(path, "rb") as f:
        content = f.read()
    # 计算并返回文件内容的 MD5 哈希值，按照十六进制字符串格式返回
    return hashlib.md5(content).hexdigest()

def get_ckpt_path(name, root, check=False):
    # 确保 name 在 URL_MAP 中
    assert name in URL_MAP
    # 构建完整的文件路径
    path = os.path.join(root, CKPT_MAP[name])

    # 如果文件不存在，或者需要检查且 MD5 校验不通过，则下载文件
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        # 下载文件
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        # 下载完成后进行 MD5 校验
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    
    # 返回路径
    return path