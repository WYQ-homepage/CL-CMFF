import os
import time

import torch
import zipfile
import tarfile
import yaml
import logging
from logging.handlers import RotatingFileHandler
class DictConfig(dict):
    """支持点符号访问的字典包装器"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

class TrainConfig:
    @staticmethod
    def from_yaml(path):
        with open(path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
        return TrainConfig._dict_to_object(cfg_dict)

    @staticmethod
    def _dict_to_object(d):
        if isinstance(d, dict):
            cfg = DictConfig()
            for k, v in d.items():
                cfg[k] = TrainConfig._dict_to_object(v)
            return cfg
        elif isinstance(d, list):
            return [TrainConfig._dict_to_object(v) for v in d]
        else:
            return d



import os
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple, Optional

# ==========================================
# 独立的 Worker 函数 (定义在类外以避免 Pickling 问题)
# ==========================================
def _stats_worker(file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    子进程工作函数：计算分配到的文件列表的 Channel Sum 和 Channel Squared Sum
    """
    # 关键：在子进程中禁用 OpenCV 的多线程，防止与 multiprocessing 冲突导致死锁或性能下降
    cv2.setNumThreads(0)
    
    local_sum = np.zeros(3, dtype=np.float64)
    local_sq_sum = np.zeros(3, dtype=np.float64)
    local_count = 0

    for path in file_paths:
        try:
            # 读取图像
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # BGR -> RGB (这是最关键的一步，PyTorch通常使用RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 归一化到 0-1
            img = img.astype(np.float32) / 255.0
            
            # 统计
            h, w, _ = img.shape
            local_count += (h * w)
            
            img_reshaped = img.reshape(-1, 3)
            local_sum += np.sum(img_reshaped, axis=0)
            local_sq_sum += np.sum(img_reshaped ** 2, axis=0)
            
        except Exception as e:
            print(f"[Warning] Error reading {path}: {e}")

    return local_sum, local_sq_sum, local_count


# ==========================================
# 主统计类
# ==========================================
class DatasetStatsCalculator:
    def __init__(self, root_dir: str, img_subdir: str = "images", img_suffix: str = ".png", num_workers: int = 4):
        """
        初始化统计器
        Args:
            root_dir: 数据集根目录 (例如 ./dataset/IRSAMap_V2_split/train)
            img_subdir: 图片子文件夹名称 (例如 "images")
            img_suffix: 图片后缀 (例如 ".png" 或 ".jpg")
            num_workers: 并行进程数
        """
        self.images_dir = os.path.join(root_dir, img_subdir)
        self.img_suffix = img_suffix
        self.num_workers = num_workers

    def run(self) -> Tuple[List[float], List[float]]:
        """
        执行统计
        Returns:
            (mean, std): 两个包含3个浮点数的列表
        """
        print(f"[Stats] 正在扫描目录: {self.images_dir}")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory not found: {self.images_dir}")

        # 1. 收集所有文件路径
        files = [
            os.path.join(self.images_dir, f) 
            for f in os.listdir(self.images_dir) 
            if f.endswith(self.img_suffix)
        ]
        total_files = len(files)
        
        if total_files == 0:
            print("[Stats] 未找到符合后缀的图片文件。")
            return None, None

        print(f"[Stats] 发现 {total_files} 张图片，启动 {self.num_workers} 个进程进行计算...")

        # 2. 切分任务 (Chunking)
        # 确保每个 chunk 至少有1个文件，避免空 chunk
        chunk_size = max(1, total_files // self.num_workers)
        chunks = [files[i : i + chunk_size] for i in range(0, total_files, chunk_size)]

        # 3. 多进程 Map-Reduce
        global_sum = np.zeros(3, dtype=np.float64)
        global_sq_sum = np.zeros(3, dtype=np.float64)
        global_count = 0

        # 使用 imap + tqdm 显示进度
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_stats_worker, chunks), 
                total=len(chunks), 
                desc="[Stats] Computing"
            ))

        # 4. 汇总结果
        for r_sum, r_sq_sum, r_count in results:
            global_sum += r_sum
            global_sq_sum += r_sq_sum
            global_count += r_count

        if global_count == 0:
            print("[Stats] Error: 有效像素总数为0。")
            return None, None

        # 5. 计算 Mean & Std
        # E[X]
        mean = global_sum / global_count
        # Var(X) = E[X^2] - (E[X])^2
        variance = (global_sq_sum / global_count) - (mean ** 2)
        # 处理浮点数精度误差导致的极微小负数
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)

        mean_list = mean.tolist()
        std_list = std.tolist()

        print("-" * 40)
        print(f"Dataset Statistics Result:")
        print(f"Mean: {mean_list}")
        print(f"Std:  {std_list}")
        print("-" * 40)

        return mean_list, std_list

class FileUtils:
    """
    Utility class for file-related operations.
    
    This class provides static methods to handle file operations such as extracting archives.
    """
    @staticmethod
    def extract(file_path, target_folder=None):
        """
        Extracts the contents of a zip or tar archive to a specified directory.
        
        :param file_path: The path to the archive file.
        :param target_folder: The target directory for extraction. If None, extracts to the archive's directory.
        :return: The path to the extracted directory.
        :raises ValueError: If the file is not a zip or tar archive.
        """
        base_dir = os.path.dirname(file_path)
        data_dir, ext = os.path.splitext(file_path)
        if ext == '.zip':
            with zipfile.ZipFile(file_path, 'r') as fp:
                fp.extractall(base_dir)
        elif ext in ('.tar', '.gz'):
            with tarfile.open(file_path, 'r') as fp:
                fp.extractall(base_dir)
        else:
            raise ValueError('Only zip/tar files can be extracted')
        return os.path.join(base_dir, target_folder) if target_folder else data_dir

class ExecutionTimer:
    """
    Utility class for measuring execution time.
    
    This class provides methods to start, end, and calculate the execution time of a code block.
    """
    def __init__(self):
        """
        Initializes the ExecutionTimer instance.
        """
        self.start_time = 0
        self.end_time = 0

    def start(self):
        """
        Starts the timer.
        """
        self.start_time = time.time()

    def end(self):
        """
        Ends the timer.
        """
        self.end_time = time.time()

    def execution_time(self):
        """
        Returns the execution time in seconds.
        
        :return: The execution time formatted to two decimal places.
        """
        return self.end_time - self.start_time

class ConfigManager:
    """
    Utility class for managing configuration files.
    
    This class provides methods to read and write configuration files in YAML format.
    """
    def __init__(self, filename='parameters.yaml'):
        """
        Initializes the ConfigManager instance with a specified filename.
        
        :param filename: The name of the configuration file.
        """
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as file:
                yaml_dict = {"batch_size":16,"lr":0.001}
                yaml.dump(yaml_dict, file)

    def config_read(self):
        """
        Reads the configuration file and returns its contents.
        
        :return: A dictionary containing the configuration data.
        :raises ValueError: If the configuration file is empty or not a valid YAML.
        """
        try:
            with open(self.filename, 'r') as file:
                config_dict = yaml.safe_load(file)
            if config_dict is None:
                raise ValueError(f"The configuration file {self.filename} is empty or not a valid YAML.")
            return config_dict
        except Exception as e:
            print(f"Failed to read the configuration file: {e}")

class LogManager:
    """
    Utility class for managing logging.
    
    This class provides methods to configure and retrieve a logger with rotating file handler.
    """
    def __init__(self, name, level="INFO", max_bytes=1024*1024*1024, backup_count=5):
        """
        Initializes the LogManager instance with a specified logger name and configuration.
        
        :param name: The name of the logger.
        :param level: The logging level.
        :param max_bytes: The maximum size of the log file before rotation.
        :param backup_count: The number of backup log files to keep.
        """
        log_dir = "./log"  
        if not os.path.exists(log_dir):
            os.makedirs(log_dir) 
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        log_file = f"./log/{name}.log"   
  
        handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("\n-----------------------------Begin-----------------------------")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        
    def get_logger(self):
        """
        Returns the configured logger.
        
        :return: The logger instance.
        """
        return self.logger
    
if __name__ == "__main__":
    """
    Test the utility classes.
    """
    log_manager = LogManager('utils')
    logger = log_manager.get_logger()
    timer = ExecutionTimer()
    timer.start()

    timer.end()
    logger.info(f"execution_time:{timer.execution_time()}s")