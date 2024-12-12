import numpy as np
from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.feature_selection import chi2
from tqdm import tqdm

def calculate_chi2_scores(X: csr_matrix, y: np.ndarray, vocabulary: Dict[str, int]) -> Dict[str, float]:
    """
    计算每个特征的卡方统计量
    :param X: 特征矩阵
    :param y: 标签数组
    :param vocabulary: 词汇表
    :return: 特征重要性得分字典
    """
    chi2_scores, _ = chi2(X, y)
    return {word: score for word, score in zip(vocabulary.keys(), chi2_scores)}

def select_features_by_importance(texts: List[List[str]], labels: List[int], 
                                n_features: int = 10000, min_df: int = 5) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    基于特征重要性选择特征
    :param texts: 文本列表
    :param labels: 标签列表
    :param n_features: 要选择的特征数量
    :param min_df: 最小文档频率
    :return: 词汇表和特征重要性得分
    """
    print("开始特征选择...")
    
    # 统计词频和文档频率
    doc_freq = Counter()
    for text in tqdm(texts, desc="统计词频"):
        unique_words = set(text)
        doc_freq.update(unique_words)
    
    # 根据文档频率过滤低频词
    qualified_words = {word for word, freq in doc_freq.items() if freq >= min_df}
    vocabulary = {word: idx for idx, word in enumerate(qualified_words)}
    
    # 创建特征矩阵
    X = texts_to_sparse(texts, vocabulary, use_tfidf=False)
    y = np.array(labels)
    
    # 计算特征重要性
    importance_scores = calculate_chi2_scores(X, y, vocabulary)
    
    # 选择top特征
    selected_words = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
    final_vocab = {word: idx for idx, (word, _) in enumerate(selected_words)}
    final_scores = {word: score for word, score in selected_words}
    
    print(f"特征选择完成，选择了{len(final_vocab)}个特征")
    return final_vocab, final_scores

def texts_to_sparse(texts: List[List[str]], vocabulary: Dict[str, int], use_tfidf: bool = True) -> csr_matrix:
    """
    将文本转换为稀疏矩阵表示
    :param texts: 文本列表
    :param vocabulary: 词汇表
    :param use_tfidf: 是否使用TF-IDF
    :return: 稀疏矩阵
    """
    rows, cols, data = [], [], []
    
    # 如果使用TF-IDF，首先计算IDF
    if use_tfidf:
        doc_freq = Counter()
        total_docs = len(texts)
        for text in texts:
            unique_words = set(text)
            for word in unique_words:
                if word in vocabulary:
                    doc_freq[word] += 1
        
        idf = {word: np.log(total_docs / (freq + 1)) + 1 
               for word, freq in doc_freq.items()}
    
    # 构建稀疏矩阵
    for idx, text in enumerate(texts):
        word_counts = Counter(text)
        for word, count in word_counts.items():
            if word in vocabulary:
                rows.append(idx)
                cols.append(vocabulary[word])
                if use_tfidf:
                    tf = 1 + np.log(count)  # 对TF取对数平滑
                    data.append(tf * idf.get(word, 1.0))
                else:
                    data.append(count)
    
    return csr_matrix((data, (rows, cols)), 
                     shape=(len(texts), len(vocabulary))) 