import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os
import warnings
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# 完全抑制所有警告
warnings.filterwarnings('ignore')

def get_chinese_font():
    """获取中文字体"""
    try:
        # 尝试使用系统中的中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
            'C:/Windows/Fonts/msyh.ttf',    # Windows 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # Windows 宋体
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                # 设置全局字体
                mpl.rcParams['font.sans-serif'] = [os.path.splitext(os.path.basename(font_path))[0]]
                mpl.rcParams['axes.unicode_minus'] = False
                return FontProperties(fname=font_path)
        
        return None
    except:
        return None

# 在模块开始时就设置字体
chinese_font = get_chinese_font()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> str:
    """
    绘制混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_feature_importance(feature_scores: Dict[str, float], top_n: int = 20, save_dir: str = None) -> str:
    """
    绘制特征重要性图
    :param feature_scores: 特征重要性得分字典
    :param top_n: 显示前n个特征
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    plt.figure(figsize=(12, 6))
    
    # 获取前N个重要特征
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_features)
    
    # 绘制条形图
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.title(f'Top {top_n} 特征重要性')
    plt.xlabel('重要性得分')
    
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_dir: str) -> str:
    """
    绘制ROC曲线
    :param y_true: 真实标签
    :param y_prob: 预测概率
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    plt.figure(figsize=(8, 6))
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.title('接收者操作特征(ROC)曲线')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_dir: str) -> str:
    """
    绘制精确率-召回率曲线
    :param y_true: 真实标签
    :param y_prob: 预测概率
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    plt.figure(figsize=(8, 6))
    
    # 计算精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # 绘制精确率-召回率曲线
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR曲线 (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.title('精确率-召回率曲线')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.legend(loc="lower left")
    
    save_path = os.path.join(save_dir, 'precision_recall_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_class_distribution(class_counts: Dict[str, int], save_dir: str) -> str:
    """
    绘制类别分布图
    :param class_counts: 类别计数字典
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    plt.figure(figsize=(8, 6))
    
    # 准备数据
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    total = sum(sizes)
    
    # 计算百分比
    percentages = [count/total*100 for count in sizes]
    
    # 设置颜色
    colors = ['#ff9999', '#66b3ff']
    
    # 绘制饼图
    plt.pie(sizes, 
            labels=[f'{label}\n({perc:.1f}%)' for label, perc in zip(labels, percentages)],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True)
    
    plt.title('数据集类别分布')
    plt.axis('equal')  # 保持饼图为圆形
    
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_metrics_comparison(metrics_history: List[Dict[str, float]], save_dir: str) -> str:
    """
    绘制不同评估指标的比较图
    :param metrics_history: 评估指标历史记录
    :param save_dir: 保存目录
    :return: 保存的文件路径
    """
    plt.figure(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    x = range(len(metrics_history))
    
    for metric, name, color in zip(metrics, metric_names, colors):
        values = [m.get(metric, 0) for m in metrics_history]
        plt.plot(x, values, marker='o', label=name, color=color, linewidth=2, markersize=8)
    
    plt.title('模型评估指标比较')
    plt.xlabel('评估次数')
    plt.ylabel('指标值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path