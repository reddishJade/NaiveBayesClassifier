import pickle
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
import numpy as np
from dataclasses import dataclass

from ..utils import clean_text, tokenize_text, texts_to_sparse
from .naive_bayes import NaiveBayesClassifier

@dataclass
class PredictionResult:
    """预测结果"""
    label: int
    probability: float
    important_features: List[Tuple[str, float]]

    @property
    def class_name(self) -> str:
        return "垃圾邮件" if self.label == 0 else "正常邮件"

    def format_result(self) -> str:
        """格式化预测结果输出"""
        output = [
            "\n预测结果:",
            f"类别: {self.class_name}",
            f"置信度: {self.probability*100:.2f}%"
        ]
        
        if self.important_features:
            output.append("\n重要特征词:")
            for word, score in self.important_features:
                output.append(f"- {word}: {score:.4f}")
        
        return '\n'.join(output)

class EmailPredictor:
    def __init__(self, model_path: str, vocab_path: str):
        """
        初始化��件预测器
        :param model_path: 模型文件路径
        :param vocab_path: 词汇表文件路径
        """
        # 加载模型和词汇表
        self.model = self._load_model(model_path)
        vocab_data = self._load_vocabulary(vocab_path)
        self.vocabulary = vocab_data['vocabulary']
        self.feature_scores = vocab_data['feature_scores']
        
        print("预测器初始化完成！")
        print(f"词汇表大小: {len(self.vocabulary)}")

    def _load_model(self, model_path: str) -> NaiveBayesClassifier:
        """加载模型"""
        try:
            classifier = NaiveBayesClassifier()
            classifier.load_model(model_path)
            return classifier
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")

    def _load_vocabulary(self, vocab_path: str) -> Dict:
        """加载词汇表"""
        try:
            with open(vocab_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"加载词汇表失败: {str(e)}")

    def _preprocess_text(self, text: str) -> List[str]:
        """
        预处理文本
        :param text: 输入文本
        :return: 处理后的词列表
        """
        cleaned_text = clean_text(text)
        return tokenize_text(cleaned_text)

    def _get_important_features(self, tokens: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        获取文本中最重要的特征
        :param tokens: 分词后的文本
        :param top_k: 返回前k个重要特征
        :return: [(特征, 重要性得分)]
        """
        feature_importance = [
            (token, self.feature_scores[token])
            for token in tokens
            if token in self.vocabulary and token in self.feature_scores
        ]
        return sorted(feature_importance, key=lambda x: x[1], reverse=True)[:top_k]

    def predict_text(self, text: str) -> PredictionResult:
        """
        预测单个文本
        :param text: 输入文本
        :return: 预测结果对象
        """
        # 文本预处理
        tokens = self._preprocess_text(text)
        
        # 转换为特征向量
        X = texts_to_sparse([tokens], self.vocabulary)
        
        # 获取预测结果
        probs = self.model.predict_proba(X)
        predicted_class = self.model.predict(X)[0]
        prediction_prob = probs[0, predicted_class]
        
        # 获取重要特征
        important_features = self._get_important_features(tokens)
        
        return PredictionResult(
            label=predicted_class,
            probability=prediction_prob,
            important_features=important_features
        )

    def batch_predict(self, texts: List[str]) -> List[PredictionResult]:
        """
        批量预测多个文本
        :param texts: 文本列表
        :return: 预测结果列表
        """
        # 文本预处理
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # 转换为特征向量
        X = texts_to_sparse(processed_texts, self.vocabulary)
        
        # 获取预测结果
        probs = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # 整理结果
        results = []
        for i, tokens in enumerate(processed_texts):
            pred_class = predictions[i]
            pred_prob = probs[i, pred_class]
            important_features = self._get_important_features(tokens)
            
            results.append(PredictionResult(
                label=pred_class,
                probability=pred_prob,
                important_features=important_features
            ))
        
        return results

    def interactive_predict(self):
        """交互式预测"""
        print("\n=== 交互式邮件预测系统 ===")
        print("输入 'quit' 退出预测")
        
        while True:
            print("\n请输入要预测的邮件内容:")
            text = input().strip()
            
            if text.lower() == 'quit':
                print("退出预测系统")
                break
            
            if not text:
                print("输入为空，请重新输入")
                continue
            
            try:
                # 进行预测
                result = self.predict_text(text)
                # 输出格式化结果
                print(result.format_result())
            except Exception as e:
                print(f"预测出错: {str(e)}")
                print("请重新输入") 