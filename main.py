import time
from src import (
    DataLoader, 
    NaiveBayesClassifier, 
    EmailPredictor,
    setup_logger,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_metrics_comparison
)
from src.utils import format_metrics
from config import Config

# 设置日志系统
logger = setup_logger(Config.LOG_DIR)

class EmailClassificationSystem:
    def __init__(self):
        """初始化邮件分类系统"""
        self.data_loader = None
        self.classifier = None
        self.predictor = None
        self.metrics_history = []
        
        # 确保所有必要的目录都存在
        Config.ensure_directories()
        
        # 显示数据路径信息
        self._print_data_info()

    def _print_data_info(self):
        """打印数据路径信息"""
        logger.info("\n=== 数据路径配置 ===")
        for name, path in Config.get_data_info().items():
            logger.info(f"{name}: {path}")

    def train_model(self):
        """训练新模型"""
        logger.info("\n=== ��始始训练新模型 ===")
        start_time = time.time()

        # 初始化数据加载器
        self.data_loader = DataLoader(
            data_path=Config.MAIL_DATA_DIR,
            full_index_path=Config.MAIL_INDEX_PATH,
            n_features=Config.FEATURE_COUNT,
            max_text_length=Config.MAX_TEXT_LENGTH,
            min_df=Config.MIN_DOC_FREQ,
            train_ratio=1 - Config.TRAIN_TEST_SPLIT,
            random_seed=Config.RANDOM_SEED,
            n_jobs=Config.N_JOBS
        )

        # 数据准备
        self._prepare_data()

        # 训练和评估
        self._train_and_evaluate()

        # 保存模型和词汇表
        self._save_model_and_vocab()

        # 输出训练信息
        self._print_training_info(start_time)

    def _prepare_data(self):
        """准备训练数据"""
        # 尝试加载预处理数据
        if self.data_loader.load_processed_data(
            Config.PROCESSED_FEATURES_PATH,
            Config.PROCESSED_LABELS_PATH,
            Config.PROCESSED_VOCAB_PATH
        ):
            logger.info("使用已有的预处理数据")
            return

        # 如果没有预处理数据，从原始数据开始处理
        logger.info("从原始数据开始处理...")
        self.data_loader.load_and_split_data()
        
        # 绘制类别分布图
        class_counts = {
            "垃圾邮件": sum(1 for label in self.data_loader.train_data.labels if label == 0),
            "正常邮件": sum(1 for label in self.data_loader.train_data.labels if label == 1)
        }
        dist_path = plot_class_distribution(class_counts, Config.VISUALIZATION_DIR)
        logger.info(f"类别分布图已保存到: {dist_path}")
        
        self.data_loader.prepare_features()
        
        # 保存预处理数据供下次使用
        self.data_loader.save_processed_data(
            Config.PROCESSED_FEATURES_PATH,
            Config.PROCESSED_LABELS_PATH,
            Config.PROCESSED_VOCAB_PATH
        )

    def _train_and_evaluate(self):
        """训练模型并评估性能"""
        self.classifier = NaiveBayesClassifier(alpha=Config.ALPHA)
        self.classifier.fit(self.data_loader.train_data.features, 
                          self.data_loader.train_data.labels)

        # 评估各个数据集
        datasets = [
            ("训练集", self.data_loader.train_data),
            ("验证集", self.data_loader.eval_data),
            ("测试集", self.data_loader.test_data)
        ]

        for name, dataset in datasets:
            logger.info(f"\n{name}性能:")
            y_pred = self.classifier.predict(dataset.features)
            y_prob = self.classifier.predict_proba(dataset.features)[:, 1]  # 获取正类的概率
            _, metrics = self.classifier.evaluate(dataset.features, dataset.labels)
            logger.info(format_metrics(metrics))
            
            # 保存评估指标历史
            self.metrics_history.append(metrics['overall'])
            
            # 对测试集生成详细的可视化评估
            if name == "测试集":
                # 混淆矩阵
                cm_path = plot_confusion_matrix(
                    dataset.labels, 
                    y_pred,
                    Config.VISUALIZATION_DIR
                )
                logger.info(f"混淆矩阵已保存到: {cm_path}")
                
                # ROC曲线
                roc_path = plot_roc_curve(
                    dataset.labels,
                    y_prob,
                    Config.VISUALIZATION_DIR
                )
                logger.info(f"ROC曲线已保存到: {roc_path}")
                
                # 精确率-召回率曲线
                pr_path = plot_precision_recall_curve(
                    dataset.labels,
                    y_prob,
                    Config.VISUALIZATION_DIR
                )
                logger.info(f"精确率-召回率曲线已保存到: {pr_path}")

        # 绘制评估指标比较图
        metrics_path = plot_metrics_comparison(
            self.metrics_history,
            Config.VISUALIZATION_DIR
        )
        logger.info(f"评估指标比较图已保存到: {metrics_path}")

        # 生成并保存特征重要性图
        if hasattr(self.data_loader, 'feature_scores'):
            fi_path = plot_feature_importance(
                self.data_loader.feature_scores,
                Config.TOP_FEATURES_TO_SHOW,
                Config.VISUALIZATION_DIR
            )
            logger.info(f"特征重要性图已保存到: {fi_path}")

    def _save_model_and_vocab(self):
        """保存模型和词汇表"""
        logger.info(f"\n保存模型到: {Config.MODEL_PATH}")
        self.classifier.save_model(Config.MODEL_PATH)
        logger.info(f"保存词汇表到: {Config.VOCAB_PATH}")

    def _print_training_info(self, start_time):
        """打印训练信息和统计数据"""
        total_time = time.time() - start_time
        logger.info(f"\n总训练时间: {total_time:.2f} 秒")

        logger.info("\n模型信息:")
        logger.info(f"特征数量: {self.classifier.n_features}")
        logger.info(f"类别数量: {len(self.classifier.classes)}")
        logger.info(f"拉普拉斯平滑参数: {self.classifier.alpha}")

        total_samples = sum(self.classifier.class_counts)
        logger.info("\n类别分布:")
        for i, count in enumerate(self.classifier.class_counts):
            class_name = "垃圾邮件" if i == 0 else "正常邮件"
            percentage = (count / total_samples) * 100
            logger.info(f"{class_name}: {percentage:.2f}%")

    def interactive_predict(self):
        """交互式预测"""
        logger.info("\n=== 开始交互式预测 ===")
        try:
            self.predictor = EmailPredictor(Config.MODEL_PATH, Config.VOCAB_PATH)
            self.predictor.interactive_predict()
        except Exception as e:
            logger.error(f"预测系统初始化失败: {str(e)}")

def main():
    system = EmailClassificationSystem()
    while True:
        print("\n=== 邮件分类系统 ===")
        print("1. 训练新模型")
        print("2. 交互式预测")
        print("3. 退出系统")
        
        choice = input("\n请选择操作 (1-3): ").strip()
        
        if choice == '1':
            system.train_model()
        elif choice == '2':
            system.interactive_predict()
        elif choice == '3':
            logger.info("已退出系统")
            break
        else:
            print("无效的选择，请重新输入")

if __name__ == "__main__":
    main()
