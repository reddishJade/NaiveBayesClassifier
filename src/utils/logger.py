import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = 'logs') -> logging.Logger:
    """
    设置日志系统
    :param log_dir: 日志保存目录
    :return: 日志器实例
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取日志器
    logger = logging.getLogger('spam_classifier')
    
    # 如果日志器已经有处理器，说明已经初始化过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    log_file = os.path.join(log_dir, f'spam_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 防止日志向上传播
    logger.propagate = False
    
    return logger 