import os

class Config:
    # 基础路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据相关路径
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DATA_RAW_DIR = os.path.join(DATA_DIR, "trec06c")  # 原始数据目录
    DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # 处理后的数据目录
    
    # 预处理数据文件
    PROCESSED_FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features.npz")
    PROCESSED_LABELS_PATH = os.path.join(DATA_PROCESSED_DIR, "labels.npy")
    PROCESSED_VOCAB_PATH = os.path.join(DATA_PROCESSED_DIR, "vocab.pkl")
    
    # 邮件数据路径
    MAIL_DATA_DIR = os.path.join(DATA_RAW_DIR, "data")
    MAIL_INDEX_PATH = os.path.join(DATA_RAW_DIR, "full", "index")
    
    # 模型相关路径
    MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
    MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
    VOCAB_PATH = os.path.join(MODEL_DIR, "vocabulary.pkl")
    
    # 可视化结果路径
    VISUALIZATION_DIR = os.path.join(BASE_DIR, "visualizations")
    
    # 日志路径
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # 模型参数
    FEATURE_COUNT = 10000  # 特征数量
    MAX_TEXT_LENGTH = 500  # 最大文本长度
    MIN_DOC_FREQ = 5      # 最小文档频率
    ALPHA = 1.0           # 拉普拉斯平滑参数
    
    # 训练参数
    TRAIN_TEST_SPLIT = 0.2  # 测试集比例
    RANDOM_SEED = 42        # 随机种子
    
    # 并行处理参数
    N_JOBS = None  # None表示使用所有可用CPU核心
    
    # 可视化参数
    VISUALIZATION_DPI = 300  # 图像DPI
    TOP_FEATURES_TO_SHOW = 20  # 显示前N个重要特征
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录都存在"""
        directories = [
            cls.DATA_DIR,
            cls.DATA_PROCESSED_DIR,
            cls.MODEL_DIR,
            cls.VISUALIZATION_DIR,
            cls.LOG_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    @classmethod
    def get_data_info(cls):
        """获取数据相关信息"""
        return {
            "数据根目录": cls.DATA_DIR,
            "原始数据目录": cls.DATA_RAW_DIR,
            "处理后数据目录": cls.DATA_PROCESSED_DIR,
            "邮件数据目录": cls.MAIL_DATA_DIR,
            "邮件索引文件": cls.MAIL_INDEX_PATH,
            "模型保存目录": cls.MODEL_DIR,
            "可视化结果目录": cls.VISUALIZATION_DIR,
            "日志目录": cls.LOG_DIR
        }