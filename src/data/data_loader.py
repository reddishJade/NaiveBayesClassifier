import os
import random
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from dataclasses import dataclass
from scipy.sparse import csr_matrix, save_npz, load_npz
from multiprocessing import Pool, cpu_count
from functools import partial

from ..utils import (
    load_stopwords,
    clean_text,
    tokenize_text,
    select_features_by_importance,
    texts_to_sparse,
    setup_logger
)

logger = setup_logger()

@dataclass
class DataSplit:
    """数据集划分"""
    texts: List[List[str]]
    labels: List[int]
    features: Optional[csr_matrix] = None

def process_single_file(args) -> Tuple[List[str], int]:
    """
    处理单个文件的函数（用于多进程）
    :param args: (file_path, label, max_text_length, stopwords)
    :return: (处理后的词列表, 标签)
    """
    file_path, label, max_text_length, stopwords = args
    try:
        with open(file_path, 'r', encoding='gb2312', errors='ignore') as f:
            text = f.read()
            text = clean_text(text)[-max_text_length:]
            return tokenize_text(text, stopwords), label
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return [], label

class DataLoader:
    def __init__(self, data_path: str, full_index_path: str, 
                 n_features: int = 10000, stopwords_file: str = None,
                 max_text_length: int = 500, min_df: int = 5,
                 train_ratio: float = 0.8, eval_ratio: float = 0.1,
                 random_seed: int = 42, n_jobs: int = None):
        """
        初始化数据加载器
        :param n_jobs: 并行处理的进程数，默认为CPU核心数
        """
        self.data_path = data_path
        self.full_index_path = full_index_path
        self.n_features = n_features
        self.max_text_length = max_text_length
        self.min_df = min_df
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.random_seed = random_seed
        self.n_jobs = n_jobs or cpu_count()
        
        # 加载停用词
        self.stopwords = load_stopwords(stopwords_file)
        
        # 初始化数据集
        self.train_data = DataSplit([], [])
        self.eval_data = DataSplit([], [])
        self.test_data = DataSplit([], [])
        
        # 特征相关
        self.vocabulary: Dict[str, int] = {}
        self.feature_scores: Dict[str, float] = {}
        
        logger.info(f"数据加载器初始化完成，使用 {self.n_jobs} 个进程进行并行处理")

    def save_processed_data(self, features_path: str, labels_path: str, vocab_path: str) -> None:
        """
        保存预处理后的数据
        :param features_path: 特征矩阵保存路径
        :param labels_path: 标签数组保存路径
        :param vocab_path: 词汇表保存路径
        """
        # 合并所有数据集的特征和标签
        from scipy.sparse import vstack
        features = vstack([
            self.train_data.features,
            self.eval_data.features,
            self.test_data.features
        ])
        labels = np.concatenate([
            self.train_data.labels,
            self.eval_data.labels,
            self.test_data.labels
        ])
        
        # 保存特征矩阵
        save_npz(features_path, features)
        
        # 保存标签
        np.save(labels_path, labels)
        
        # 保存词汇表和特征分数
        with open(vocab_path, 'wb') as f:
            pickle.dump({
                'vocabulary': self.vocabulary,
                'feature_scores': self.feature_scores
            }, f)
        
        logger.info(f"预处理数据已保存：")
        logger.info(f"特征矩阵: {features_path}")
        logger.info(f"标签数组: {labels_path}")
        logger.info(f"词汇表: {vocab_path}")

    def load_processed_data(self, features_path: str, labels_path: str, vocab_path: str) -> bool:
        """
        加载预处理后的数据
        :param features_path: 特征矩阵路径
        :param labels_path: 标签数组路径
        :param vocab_path: 词汇表路径
        :return: 是否成功加载
        """
        try:
            # 检查所有文件是否存在
            if not all(os.path.exists(f) for f in [features_path, labels_path, vocab_path]):
                return False
            
            # 加载特征矩阵
            features = load_npz(features_path)
            
            # 加载标签
            labels = np.load(labels_path).tolist()
            
            # 加载词汇表和特征分数
            with open(vocab_path, 'rb') as f:
                data = pickle.load(f)
                self.vocabulary = data['vocabulary']
                self.feature_scores = data['feature_scores']
            
            # 划分数据集
            total = len(labels)
            train_size = int(self.train_ratio * total)
            eval_size = int(self.eval_ratio * total)
            
            # 设置训练集
            self.train_data.features = features[:train_size]
            self.train_data.labels = labels[:train_size]
            
            # 设置验证集
            self.eval_data.features = features[train_size:train_size + eval_size]
            self.eval_data.labels = labels[train_size:train_size + eval_size]
            
            # 设置测试集
            self.test_data.features = features[train_size + eval_size:]
            self.test_data.labels = labels[train_size + eval_size:]
            
            logger.info("成功加载预处理数据")
            return True
            
        except Exception as e:
            logger.error(f"加载预处理数据失败: {str(e)}")
            return False

    def load_and_split_data(self) -> None:
        """使用多进程加载数据并划分数据集"""
        logger.info("开始加载数据...")
        
        # 读取文件列表和标签
        with open(self.full_index_path, 'r', encoding='utf-8') as f:
            file_info = []
            for line in f:
                label_path = line.strip().split(" ")
                label = 0 if label_path[0] == 'spam' else 1
                file_path = os.path.join(self.data_path, label_path[1].strip())
                file_info.append((file_path, label))
        
        total_files = len(file_info)
        logger.info(f"找到 {total_files} 个文件")
        
        # 准备多进程参数
        process_args = [
            (path, label, self.max_text_length, self.stopwords)
            for path, label in file_info
        ]
        
        # 使用多进程处理文件
        logger.info(f"使用 {self.n_jobs} 个进程并行处理文件...")
        with Pool(self.n_jobs) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(process_args),
                desc="处理文件"
            ))
        
        # 过滤空结果
        all_data = [(tokens, label) for tokens, label in results if tokens]
        
        logger.info(f"成功处理 {len(all_data)} 个文件")
        
        # 设置随机种子并打乱数据
        random.seed(self.random_seed)
        random.shuffle(all_data)
        
        # 划分数据集
        total = len(all_data)
        train_size = int(self.train_ratio * total)
        eval_size = int(self.eval_ratio * total)
        
        train_data = all_data[:train_size]
        eval_data = all_data[train_size:train_size + eval_size]
        test_data = all_data[train_size + eval_size:]
        
        # 分离特征和标签
        self.train_data.texts, self.train_data.labels = zip(*train_data)
        self.eval_data.texts, self.eval_data.labels = zip(*eval_data)
        self.test_data.texts, self.test_data.labels = zip(*test_data)
        
        logger.info(f"数据集划分完成:")
        logger.info(f"训练集: {len(self.train_data.texts)} 样本")
        logger.info(f"验证集: {len(self.eval_data.texts)} 样本")
        logger.info(f"测试集: {len(self.test_data.texts)} 样本")

    def prepare_features(self) -> None:
        """准备特征"""
        print("\n开始准备特征...")
        
        # 特征选择
        self.vocabulary, self.feature_scores = select_features_by_importance(
            self.train_data.texts,
            self.train_data.labels,
            n_features=self.n_features,
            min_df=self.min_df
        )
        
        # 转换为稀疏矩阵表示
        print("\n转换训练集...")
        self.train_data.features = texts_to_sparse(self.train_data.texts, self.vocabulary)
        
        print("转换验证集...")
        self.eval_data.features = texts_to_sparse(self.eval_data.texts, self.vocabulary)
        
        print("转换测试集...")
        self.test_data.features = texts_to_sparse(self.test_data.texts, self.vocabulary)
        
        print("特征准备完成！")

    def save_vocabulary(self, file_path: str) -> None:
        """
        保存词汇表
        :param file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'vocabulary': self.vocabulary,
                'feature_scores': self.feature_scores
            }, f)
        print(f"词汇表已保存到: {file_path}")

    def load_vocabulary(self, file_path: str) -> None:
        """
        加载词汇表
        :param file_path: 词汇表文件路径
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.vocabulary = data['vocabulary']
            self.feature_scores = data['feature_scores']
        print(f"词汇表已加载，包含{len(self.vocabulary)}个特征") 