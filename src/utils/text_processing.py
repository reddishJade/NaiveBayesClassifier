import re
import jieba
import logging
from typing import List, Set

# 设置jieba的日志级别为WARNING以抑制初始化信息
jieba.setLogLevel(logging.WARNING)

def load_stopwords(stopwords_file: str = None) -> Set[str]:
    """
    加载停用词表
    :param stopwords_file: 停用词文件路径
    :return: 停用词集合
    """
    stopwords = set()
    if stopwords_file:
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f)
        except Exception as e:
            print(f"Warning: Could not load stopwords file: {e}")
    return stopwords

def clean_text(text: str) -> str:
    """
    清理文本
    :param text: 输入文本
    :return: 清理后的文本
    """
    # 去除非中文字符，保留英文字母和数字
    text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
    # 将多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_text(text: str, stopwords: Set[str] = None) -> List[str]:
    """
    对文本进行分词
    :param text: 输入文本
    :param stopwords: 停用词集合
    :return: 分词后的词列表
    """
    # 使用jieba分词
    words = jieba.cut(text)
    # 过滤停用词和空字符
    if stopwords:
        words = [word.strip() for word in words if word.strip() and word not in stopwords]
    else:
        words = [word.strip() for word in words if word.strip()]
    return words 