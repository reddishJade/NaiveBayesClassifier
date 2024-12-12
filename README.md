# 邮件垃圾分类系统

基于朴素贝叶斯算法的邮件垃圾分类系统，支持中文邮件分类，具有完整的训练、评估和预测功能。

## 主要特性

- 支持中文邮件内容处理
- 多进程并行数据处理，提高训练速度
- 数据预处理结果持久化，提升重复训练效率
- 完整的模型评估指标和可视化
- 交互式预测功能
- 自动化的日志记录系统

## 系统要求

- Python 3.9+
- 操作系统：Windows/Linux/macOS

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/reddishJade/NaiveBayesClassifier.git
cd NaiveBayesClassifier
```

2. 创建并激活虚拟环境：
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 目录结构

```
NaiveBayesClassifier/
├── data/                  	# 数据目录
│   ├── processed/        	# 预处理后的数据
│   │   ├── features.npz  	# 特征矩阵
│   │   ├── labels.npy   	# 标签数组
│   │   └── vocab.pkl    	# 词汇表
│   └── trec06c/         	# 原始数据
├── logs/                 	# 日志文件
├── saved_models/        	# 保存的模型
├── visualizations/      	# 可视化结果
├── src/                 	# 源代码
│   ├── data/           	# 数据处理模块
│   ├── models/         	# 模型实现
│   └── utils/          	# 工具函数
├── config.py           	# 配置文件
├── main.py             	# 主程序
└── requirements.txt    	# 依赖包列表
```

## 使用说明

### 训练新模型

```bash
python main.py
# 选择选项 1 开始训练
```

训练过程会自动：
- 检查是否存在预处理数据
  - 如果存在，直接加载预处理数据
  - 如果不存在，从原始数据开始处理并保存结果
- 多进程并行处理数据
- 生成评估指标
- 保存可视化结果
- 记录训练日志

### 交互式预测

```bash
python main.py
# 选择选项 2 进行预测
```

## 数据处理流程

1. **预处理阶段**：
   - 多进程并行读取原始邮件
   - 文本清理和分词
   - 特征选择和向量化
   - 保存处理结果到 `data/processed/` 目录

2. **预处理数据格式**：
   - `features.npz`: 稀疏矩阵格式的特征向量
   - `labels.npy`: NumPy数组格式的标签
   - `vocab.pkl`: 词汇表和特征重要性分数

3. **数据复用**：
   - 后续训练自动使用预处理数据
   - 显著提高训练效率
   - 保证特征提取的一致性

## 可视化评估

系统会自动生成以下可视化结果：

1. 混淆矩阵 (confusion_matrix.png)
2. ROC曲线 (roc_curve.png)
3. 精确率-召回率曲线 (precision_recall_curve.png)
4. 类别分布图 (class_distribution.png)
5. 评估指标比较图 (metrics_comparison.png)
6. 特征重要性图 (feature_importance.png)

所有可视化结果保存在 `visualizations` 目录下。

## 评估指标

系统提供以下评估指标：

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- ROC曲线下面积 (AUC-ROC)
- 精确率-召回率曲线下面积 (AUC-PR)

## 日志系统

系统自动记录所有操作日志，包括：
- 训练过程
- 评估结果
- 错误信息
- 性能指标

日志文件保存在 `logs` 目录下。

## 配置说明

主要配置参数（在 `config.py` 中）：

- `FEATURE_COUNT`: 特征数量 (默认: 10000)
- `MAX_TEXT_LENGTH`: 最大文本长度 (默认: 500)
- `MIN_DOC_FREQ`: 最小文档频率 (默认: 5)
- `ALPHA`: 拉普拉斯平滑参数 (默认: 1.0)
- `TRAIN_TEST_SPLIT`: 测试集比例 (默认: 0.2)
- `N_JOBS`: 并行处理进程数 (默认: None，使用所有可用CPU)
- `VISUALIZATION_DPI`: 图像DPI (默认: 300)
- `TOP_FEATURES_TO_SHOW`: 显示的重要特征数量 (默认: 20)

## 性能优化

- 使用多进程并行处理数据
- 预处理数据持久化
- 使用稀疏矩阵存储特征
- 优化的特征选择算法
- 高效的文本预处理流程

## 注意事项

1. 首次运行时会自动创建必要的目录结构
2. 训练数据需要放在 `data/trec06c` 目录下
3. 确保系统有足够的内存处理大规模数据
4. 对于大型数据集，建议调整 `MAX_TEXT_LENGTH` 和 `FEATURE_COUNT` 参数
5. 如需重新处理数据，删除 `data/processed/` 目录下的文件即可
