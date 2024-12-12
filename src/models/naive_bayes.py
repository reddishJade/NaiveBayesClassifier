import numpy as np
from scipy.sparse import csr_matrix
import pickle
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

from ..utils import calculate_metrics

@dataclass
class ModelState:
    """模型状态数据类"""
    alpha: float
    class_log_priors: Optional[np.ndarray] = None
    feature_log_probs: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    n_features: Optional[int] = None
    class_counts: Optional[np.ndarray] = None
    feature_counts: Optional[np.ndarray] = None

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        """
        初始化朴素贝叶斯分类器
        :param alpha: 拉普拉斯平滑参数
        """
        self.state = ModelState(alpha=alpha)

    @property
    def alpha(self) -> float:
        return self.state.alpha

    @property
    def n_features(self) -> int:
        return self.state.n_features

    @property
    def classes(self) -> np.ndarray:
        return self.state.classes

    @property
    def class_counts(self) -> np.ndarray:
        return self.state.class_counts

    def _calculate_priors(self, y: np.ndarray) -> None:
        """
        计算类别先验概率
        :param y: 标签数组
        """
        n_samples = len(y)
        self.state.class_counts = np.bincount(y)
        self.state.class_log_priors = np.log(
            (self.state.class_counts + self.alpha) / 
            (n_samples + self.alpha * len(self.state.classes))
        )

    def _calculate_feature_probs(self, X: csr_matrix, y: np.ndarray) -> None:
        """
        计算特征条件概率
        :param X: 特征矩阵
        :param y: 标签数组
        """
        self.state.feature_counts = np.zeros((len(self.state.classes), self.state.n_features))
        
        for i, c in enumerate(tqdm(self.state.classes, desc="计算特征概率")):
            X_c = X[y == c]
            self.state.feature_counts[i] = np.array(X_c.sum(axis=0)).ravel()
        
        # 计算每个类别的总词数
        total_counts = self.state.feature_counts.sum(axis=1).reshape(-1, 1)
        
        # 使用对数概率避免数值下溢
        self.state.feature_log_probs = np.log(
            (self.state.feature_counts + self.alpha) /
            (total_counts + self.alpha * self.state.n_features)
        )

    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        """
        训练朴素贝叶斯模型
        :param X: 训练数据稀疏矩阵 [n_samples, n_features]
        :param y: 标签数组 [n_samples]
        """
        print("开始训练模型...")
        self.state.classes = np.unique(y)
        self.state.n_features = X.shape[1]
        
        print("计算类别先验概率...")
        self._calculate_priors(y)
        
        print("计算特征条件概率...")
        self._calculate_feature_probs(X, y)
        
        print("模型训练完成！")

    def _joint_log_likelihood(self, X: csr_matrix) -> np.ndarray:
        """
        计算联合对数似然
        :param X: 输入样本稀疏矩阵 [n_samples, n_features]
        :return: 对数似然数组 [n_samples, n_classes]
        """
        ret = np.zeros((X.shape[0], len(self.state.classes)))
        for i in range(len(self.state.classes)):
            ret[:, i] = self.state.class_log_priors[i]
            ret[:, i] += X.dot(self.state.feature_log_probs[i].T)
        return ret

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """
        预测样本属于各个类别的概率
        :param X: 输入样本稀疏矩阵 [n_samples, n_features]
        :return: 概率数组 [n_samples, n_classes]
        """
        jll = self._joint_log_likelihood(X)
        log_prob_x = np.max(jll, axis=1).reshape(-1, 1) + \
                     np.log(np.sum(np.exp(jll - np.max(jll, axis=1).reshape(-1, 1)), axis=1)).reshape(-1, 1)
        return np.exp(jll - log_prob_x)

    def predict(self, X: csr_matrix) -> np.ndarray:
        """
        预测样本的类别
        :param X: 输入样本稀疏矩阵 [n_samples, n_features]
        :return: 预测的类别数组 [n_samples]
        """
        jll = self._joint_log_likelihood(X)
        return self.state.classes[np.argmax(jll, axis=1)]

    def evaluate(self, X: csr_matrix, y: np.ndarray) -> Tuple[float, Dict]:
        """
        评估模型性能
        :param X: 测试数据稀疏矩阵
        :param y: 测试标签
        :return: 准确率和详细评估指标
        """
        print("评估模型性能...")
        y_pred = self.predict(X)
        metrics = calculate_metrics(y, y_pred)
        return metrics['overall']['accuracy'], metrics

    def save_model(self, file_path: str) -> None:
        """
        保存模型
        :param file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.state, f)

    def load_model(self, file_path: str) -> None:
        """
        加载模型
        :param file_path: 模型文件路径
        """
        with open(file_path, 'rb') as f:
            self.state = pickle.load(f)
        print("模型加载完成！") 