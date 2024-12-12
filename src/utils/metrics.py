import numpy as np
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    计算评估指标
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 评估指标字典
    """
    metrics = {}
    classes = np.unique(y_true)
    
    # 计算每个类别的指标
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = '垃圾邮件' if c == 0 else '正常邮件'
        metrics[class_name] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    # 计算总体准确率
    accuracy = np.mean(y_true == y_pred) * 100
    metrics['overall'] = {'accuracy': accuracy}
    
    return metrics

def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    格式化评估指标输出
    :param metrics: 评估指标字典
    :return: 格式化的字符串
    """
    output = []
    
    if 'overall' in metrics:
        output.append(f"总体准确率: {metrics['overall']['accuracy']:.2f}%\n")
    
    for class_name, class_metrics in metrics.items():
        if class_name != 'overall':
            output.append(f"\n{class_name}:")
            if 'precision' in class_metrics:
                output.append(f"精确率: {class_metrics['precision']:.2f}%")
            if 'recall' in class_metrics:
                output.append(f"召回率: {class_metrics['recall']:.2f}%")
            if 'f1' in class_metrics:
                output.append(f"F1分数: {class_metrics['f1']:.2f}%")
    
    return '\n'.join(output) 