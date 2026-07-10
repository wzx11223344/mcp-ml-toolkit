"""
模型评估工具模块

提供8个MCP模型评估工具：
- train_test_split_data: 数据集分割
- cross_validate_model: 交叉验证
- classification_report_tool: 分类报告
- confusion_matrix_tool: 混淆矩阵
- roc_curve_analysis: ROC曲线分析
- feature_importance_analysis: 特征重要性分析
- hyperparameter_grid_search: 网格搜索调参
- learning_curve_analysis: 学习曲线分析
"""

import sys
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    classification_report as sk_classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 兼容直接运行和模块导入两种方式获取 mcp 实例
if "server" in sys.modules:
    from server import mcp
else:
    from __main__ import mcp

from utils import (
    parse_array,
    parse_matrix,
    format_number,
    format_array,
    format_markdown_table,
    format_matrix,
)


def _get_classifier(model_type, **kwargs):
    """根据类型名称获取分类器实例。"""
    classifiers = {
        "logistic": lambda: LogisticRegression(max_iter=1000, **kwargs),
        "knn": lambda: KNeighborsClassifier(**kwargs),
        "svm": lambda: SVC(probability=True, **kwargs),
        "tree": lambda: DecisionTreeClassifier(**kwargs),
        "forest": lambda: RandomForestClassifier(**kwargs),
    }
    key = model_type.strip().lower()
    if key not in classifiers:
        raise ValueError(
            f"不支持的模型类型 '{model_type}'，可选: {', '.join(classifiers.keys())}"
        )
    return classifiers[key]()


@mcp.tool()
def train_test_split_data(
    X: str,
    y: str,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    stratify: str = "",
) -> str:
    """将数据集分割为训练集和测试集。

    支持随机分割和分层抽样（保持各类别比例）。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2;3,4;5,6;7,8"
        y: 目标向量，逗号分隔，如 "0,1,0,1"
        test_size: 测试集比例（0~1之间），默认 0.2
        random_state: 随机种子，保证结果可复现，默认 42
        shuffle: 是否在分割前打乱数据，默认 True
        stratify: 分层依据，传入与y相同的格式进行分层抽样；空字符串表示不分层

    返回:
        Markdown 格式字符串，包含分割后的数据形状和各集的样本分布
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        stratify_data = None
        if stratify.strip():
            stratify_data = parse_array(stratify).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_data,
        )

        lines = []
        lines.append("## 数据集分割结果\n")
        lines.append(f"**原始样本数**: {X_data.shape[0]}")
        lines.append(f"**特征数**: {X_data.shape[1]}")
        lines.append(f"**测试集比例**: {test_size}")
        lines.append(f"**随机种子**: {random_state}")
        lines.append(f"**是否打乱**: {shuffle}")
        lines.append(f"**是否分层**: {'是' if stratify_data is not None else '否'}\n")

        lines.append("### 分割结果\n")
        lines.append(format_markdown_table(
            ["数据集", "样本数", "特征数"],
            [["训练集", X_train.shape[0], X_train.shape[1]],
             ["测试集", X_test.shape[0], X_test.shape[1]]],
        ))

        # 各类别分布
        unique_classes = np.unique(y_data)
        lines.append("\n### 各类别分布\n")
        lines.append(format_markdown_table(
            ["类别"] + [f"训练集", "测试集", "原始集"],
            [[int(c), int(np.sum(y_train == c)),
              int(np.sum(y_test == c)),
              int(np.sum(y_data == c))]
             for c in unique_classes],
        ))

        # 训练集数据
        lines.append("\n### 训练集特征（X_train）\n")
        lines.append(format_matrix(X_train))

        lines.append("\n### 测试集特征（X_test）\n")
        lines.append(format_matrix(X_test))

        lines.append(f"\n**训练集标签**: {format_array(y_train)}")
        lines.append(f"**测试集标签**: {format_array(y_test)}")

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def cross_validate_model(
    X: str,
    y: str,
    model_type: str = "logistic",
    cv: int = 5,
    scoring: str = "accuracy",
) -> str:
    """对模型进行交叉验证评估。

    使用K折交叉验证评估模型性能，返回每折的评分和汇总统计。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔
        model_type: 模型类型，可选 "logistic"/"knn"/"svm"/"tree"/"forest"，默认 "logistic"
        cv: 交叉验证折数，默认 5
        scoring: 评估指标，如 "accuracy"/"precision_macro"/"recall_macro"/"f1_macro"/"r2"，默认 "accuracy"

    返回:
        Markdown 格式字符串，包含每折评分和汇总统计
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        model = _get_classifier(model_type)
        scores = cross_validate(model, X_data, y_data, cv=cv, scoring=scoring)

        test_scores = scores["test_score"]
        fit_times = scores["fit_time"]

        lines = []
        lines.append("## 交叉验证结果\n")
        lines.append(f"**模型类型**: {model_type}")
        lines.append(f"**交叉验证折数**: {cv}")
        lines.append(f"**评估指标**: {scoring}")
        lines.append(f"**样本数**: {X_data.shape[0]}")
        lines.append(f"**特征数**: {X_data.shape[1]}\n")

        lines.append("### 各折评分\n")
        lines.append(format_markdown_table(
            ["折数", "评分", "训练时间(秒)"],
            [[i + 1, format_number(test_scores[i]), format_number(fit_times[i], 3)]
             for i in range(cv)],
        ))

        lines.append("\n### 汇总统计\n")
        lines.append(format_markdown_table(
            ["统计量", "评分", "训练时间(秒)"],
            [["平均值", format_number(test_scores.mean()), format_number(fit_times.mean(), 3)],
             ["标准差", format_number(test_scores.std()), format_number(fit_times.std(), 3)],
             ["最小值", format_number(test_scores.min()), format_number(fit_times.min(), 3)],
             ["最大值", format_number(test_scores.max()), format_number(fit_times.max(), 3)]],
        ))

        lines.append(f"\n**平均评分**: {test_scores.mean():.4f} +/- {test_scores.std():.4f}")

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def classification_report_tool(
    X: str,
    y: str,
    model_type: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """生成分类模型评估报告。

    训练分类模型并生成详细的分类报告，包含每个类别的精确率、召回率、F1值和支持数。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔
        model_type: 模型类型，可选 "logistic"/"knn"/"svm"/"tree"/"forest"，默认 "logistic"
        test_size: 测试集比例，默认 0.2
        random_state: 随机种子，默认 42

    返回:
        Markdown 格式字符串，包含分类报告和整体准确率
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
        )

        model = _get_classifier(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = sk_classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        accuracy = report.get("accuracy", 0.0)

        lines = []
        lines.append("## 分类评估报告\n")
        lines.append(f"**模型类型**: {model_type}")
        lines.append(f"**训练集大小**: {X_train.shape[0]}")
        lines.append(f"**测试集大小**: {X_test.shape[0]}")
        lines.append(f"**测试集准确率**: {accuracy:.4f}\n")

        lines.append("### 各类别指标\n")
        # 获取类别标签（排除 'accuracy', 'macro avg', 'weighted avg'）
        class_labels = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        lines.append(format_markdown_table(
            ["类别", "精确率", "召回率", "F1值", "支持数"],
            [[k, format_number(report[k]["precision"]),
              format_number(report[k]["recall"]),
              format_number(report[k]["f1-score"]),
              int(report[k]["support"])]
             for k in class_labels],
        ))

        lines.append("\n### 汇总指标\n")
        lines.append(format_markdown_table(
            ["指标类型", "精确率", "召回率", "F1值", "支持数"],
            [["宏平均 (macro avg)", format_number(report["macro avg"]["precision"]),
              format_number(report["macro avg"]["recall"]),
              format_number(report["macro avg"]["f1-score"]),
              int(report["macro avg"]["support"])],
             ["加权平均 (weighted avg)", format_number(report["weighted avg"]["precision"]),
              format_number(report["weighted avg"]["recall"]),
              format_number(report["weighted avg"]["f1-score"]),
              int(report["weighted avg"]["support"])]],
        ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def confusion_matrix_tool(
    X: str,
    y: str,
    model_type: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """生成混淆矩阵。

    训练分类模型并生成混淆矩阵，显示实际类别与预测类别的对应关系。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔
        model_type: 模型类型，可选 "logistic"/"knn"/"svm"/"tree"/"forest"，默认 "logistic"
        test_size: 测试集比例，默认 0.2
        random_state: 随机种子，默认 42

    返回:
        Markdown 格式字符串，包含混淆矩阵和相关统计指标
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
        )

        model = _get_classifier(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_data)

        lines = []
        lines.append("## 混淆矩阵\n")
        lines.append(f"**模型类型**: {model_type}")
        lines.append(f"**测试集大小**: {X_test.shape[0]}")
        lines.append(f"**类别数**: {len(classes)}\n")

        lines.append("### 混淆矩阵\n")
        # 表头：预测类别
        headers = ["实际\\预测"] + [f"预测为{c}" for c in classes] + ["总计"]
        rows = []
        for i, actual in enumerate(classes):
            row_total = int(cm[i].sum())
            rows.append(
                [f"实际为{actual}"]
                + [int(cm[i, j]) for j in range(len(classes))]
                + [row_total]
            )
        # 添加总计行
        col_totals = [int(cm[:, j].sum()) for j in range(len(classes))]
        rows.append(["总计"] + col_totals + [int(cm.sum())])
        lines.append(format_markdown_table(headers, rows))

        # 对角线元素（正确预测）之和
        correct = int(np.trace(cm))
        total = int(cm.sum())
        accuracy = correct / total if total > 0 else 0

        lines.append(f"\n**正确预测数**: {correct} / {total}")
        lines.append(f"**准确率**: {accuracy:.4f}")

        # 二分类时计算更多指标
        if len(classes) == 2:
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            lines.append("\n### 二分类详细指标\n")
            lines.append(format_markdown_table(
                ["指标", "值"],
                [["真正例 (TP)", tp],
                 ["真负例 (TN)", tn],
                 ["假正例 (FP)", fp],
                 ["假负例 (FN)", fn],
                 ["精确率", format_number(precision)],
                 ["召回率 (灵敏度)", format_number(recall)],
                 ["特异度", format_number(specificity)],
                 ["F1值", format_number(f1)]],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def roc_curve_analysis(
    X: str,
    y: str,
    model_type: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """ROC曲线分析。

    计算并分析ROC曲线，包括AUC值和不同阈值下的真阳性率和假阳性率。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔（二分类标签）
        model_type: 模型类型，可选 "logistic"/"knn"/"svm"/"tree"/"forest"，默认 "logistic"
        test_size: 测试集比例，默认 0.2
        random_state: 随机种子，默认 42

    返回:
        Markdown 格式字符串，包含AUC值和ROC曲线关键点
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        unique_classes = np.unique(y_data)
        if len(unique_classes) != 2:
            return f"**错误**: ROC曲线分析仅适用于二分类任务，当前类别数为 {len(unique_classes)}"

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
        )

        model = _get_classifier(model_type)
        model.fit(X_train, y_train)

        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        auc_value = auc(fpr, tpr)

        lines = []
        lines.append("## ROC曲线分析\n")
        lines.append(f"**模型类型**: {model_type}")
        lines.append(f"**测试集大小**: {X_test.shape[0]}")
        lines.append(f"**正类标签**: {unique_classes[1]}")
        lines.append(f"**AUC值**: {auc_value:.4f}\n")

        lines.append("### AUC解读\n")
        if auc_value >= 0.9:
            lines.append("- AUC >= 0.9: 优秀分类器")
        elif auc_value >= 0.8:
            lines.append("- 0.8 <= AUC < 0.9: 良好分类器")
        elif auc_value >= 0.7:
            lines.append("- 0.7 <= AUC < 0.8: 可接受的分类器")
        elif auc_value >= 0.5:
            lines.append("- 0.5 <= AUC < 0.7: 较差的分类器")
        else:
            lines.append("- AUC < 0.5: 分类器性能不如随机猜测")

        # 选取部分阈值点展示
        n_points = min(15, len(thresholds))
        step = max(1, len(thresholds) // n_points)

        lines.append("\n### ROC曲线关键点\n")
        lines.append(format_markdown_table(
            ["阈值", "假阳性率 (FPR)", "真阳性率 (TPR)", "约登指数 (J)"],
            [[format_number(thresholds[i], 3), format_number(fpr[i]),
              format_number(tpr[i]), format_number(tpr[i] - fpr[i])]
             for i in range(0, len(thresholds), step)],
        ))

        # 最佳阈值（约登指数最大）
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        lines.append(f"\n### 最佳阈值（约登指数最大）\n")
        lines.append(f"**阈值**: {thresholds[best_idx]:.4f}")
        lines.append(f"**FPR**: {fpr[best_idx]:.4f}")
        lines.append(f"**TPR**: {tpr[best_idx]:.4f}")
        lines.append(f"**约登指数**: {youden_j[best_idx]:.4f}")

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def feature_importance_analysis(
    X: str,
    y: str,
    model_type: str = "forest",
    task_type: str = "classification",
) -> str:
    """特征重要性分析。

    使用树模型或线性模型分析各特征对预测任务的重要性。
    支持基于树模型的不纯度减少和基于线性模型的系数绝对值两种方式。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔
        model_type: 模型类型，"forest"为随机森林，"tree"为决策树，
                    "logistic"为逻辑回归（仅分类），默认 "forest"
        task_type: 任务类型，"classification"为分类，"regression"为回归，默认 "classification"

    返回:
        Markdown 格式字符串，包含各特征的重要性排名
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y)
        if task_type == "classification":
            y_data = y_data.astype(int)

        model_type_key = model_type.strip().lower()

        if model_type_key == "forest":
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_data, y_data)
            importances = model.feature_importances_
            method = "基于不纯度减少 (MDI)"
        elif model_type_key == "tree":
            if task_type == "classification":
                from sklearn.tree import DecisionTreeClassifier as DTC
                model = DTC(random_state=42)
            else:
                from sklearn.tree import DecisionTreeRegressor as DTR
                model = DTR(random_state=42)
            model.fit(X_data, y_data)
            importances = model.feature_importances_
            method = "基于不纯度减少 (MDI)"
        elif model_type_key == "logistic":
            if task_type != "classification":
                return "**错误**: 逻辑回归仅支持分类任务"
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_data, y_data)
            importances = np.abs(model.coef_[0])
            method = "基于系数绝对值"
        else:
            return f"**错误**: 不支持的模型类型 '{model_type}'，可选: forest, tree, logistic"

        # 排序
        indices = np.argsort(importances)[::-1]
        n_features = X_data.shape[1]

        lines = []
        lines.append("## 特征重要性分析\n")
        lines.append(f"**模型**: {model_type}")
        lines.append(f"**任务类型**: {task_type}")
        lines.append(f"**分析方法**: {method}")
        lines.append(f"**特征数**: {n_features}\n")

        lines.append("### 特征重要性排名\n")
        lines.append(format_markdown_table(
            ["排名", "特征索引", "重要性", "占比", "累计占比"],
            [[rank + 1, idx, format_number(importances[idx]),
              f"{importances[idx] / importances.sum() * 100:.1f}%",
              f"{importances[indices[:rank + 1]].sum() / importances.sum() * 100:.1f}%"]
             for rank, idx in enumerate(indices)],
        ))

        # 筛选建议
        cumulative = np.cumsum(importances[indices])
        threshold_90 = np.argmax(cumulative >= 0.9) + 1
        threshold_95 = np.argmax(cumulative >= 0.95) + 1

        lines.append(f"\n### 特征选择建议\n")
        lines.append(f"- 解释 90% 重要性需要前 **{threshold_90}** 个特征")
        lines.append(f"- 解释 95% 重要性需要前 **{threshold_95}** 个特征")
        lines.append(f"- 最重要特征: 特征{indices[0]} (重要性={importances[indices[0]]:.4f})")

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def hyperparameter_grid_search(
    X: str,
    y: str,
    model_type: str = "svm",
    param_grid: str = "C:0.1,1,10;kernel:rbf,linear",
    cv: int = 5,
    scoring: str = "accuracy",
) -> str:
    """网格搜索超参数调优。

    遍历所有参数组合，通过交叉验证找到最佳超参数组合。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔
        model_type: 模型类型，可选 "logistic"/"knn"/"svm"/"tree"/"forest"，默认 "svm"
        param_grid: 参数网格，格式为 "参数名:值1,值2;参数名2:值1,值2"，
                    如 "C:0.1,1,10;kernel:rbf,linear"
        cv: 交叉验证折数，默认 5
        scoring: 评估指标，默认 "accuracy"

    返回:
        Markdown 格式字符串，包含最佳参数、最佳分数和排名前N的组合
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        # 解析参数网格
        grid = {}
        param_pairs = param_grid.split(";")
        for pair in param_pairs:
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split(":")
            if len(parts) != 2:
                continue
            param_name = parts[0].strip()
            param_values = [v.strip() for v in parts[1].split(",")]

            # 尝试将值转换为数值
            parsed_values = []
            for v in param_values:
                try:
                    if "." in v:
                        parsed_values.append(float(v))
                    else:
                        parsed_values.append(int(v))
                except ValueError:
                    parsed_values.append(v)
            grid[param_name] = parsed_values

        if not grid:
            return "**错误**: 参数网格解析失败，请使用格式 '参数名:值1,值2;参数名2:值1,值2'"

        model = _get_classifier(model_type)

        search = GridSearchCV(
            model, grid, cv=cv, scoring=scoring, return_train_score=True
        )
        search.fit(X_data, y_data)

        lines = []
        lines.append("## 网格搜索结果\n")
        lines.append(f"**模型类型**: {model_type}")
        lines.append(f"**交叉验证折数**: {cv}")
        lines.append(f"**评估指标**: {scoring}")
        lines.append(f"**参数组合数**: {len(search.cv_results_['params'])}\n")

        lines.append("### 最佳参数\n")
        for k, v in search.best_params_.items():
            lines.append(f"- **{k}**: {v}")
        lines.append(f"\n**最佳分数**: {search.best_score_:.4f}\n")

        # 排名前10的组合
        results = search.cv_results_
        mean_scores = results["mean_test_score"]
        std_scores = results["std_test_score"]
        rank_indices = np.argsort(mean_scores)[::-1]
        top_n = min(10, len(rank_indices))

        lines.append(f"### 排名前{top_n}的参数组合\n")
        rows = []
        for rank, idx in enumerate(rank_indices[:top_n]):
            params_str = ", ".join(f"{k}={v}" for k, v in results["params"][idx].items())
            rows.append([
                rank + 1,
                params_str,
                format_number(mean_scores[idx]),
                format_number(std_scores[idx]),
                format_number(results["mean_fit_time"][idx], 3),
            ])
        lines.append(format_markdown_table(
            ["排名", "参数组合", "平均分数", "标准差", "训练时间(秒)"],
            rows,
        ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def learning_curve_analysis(
    X: str,
    y: str,
    model_type: str = "logistic",
    cv: int = 5,
    train_sizes: str = "0.1,0.3,0.5,0.7,1.0",
) -> str:
    """学习曲线分析。

    分析模型在不同训练数据量下的训练得分和验证得分，判断模型是欠拟合还是过拟合。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标向量，逗号分隔
        model_type: 模型类型，可选 "logistic"/"knn"/"svm"/"tree"/"forest"，默认 "logistic"
        cv: 交叉验证折数，默认 5
        train_sizes: 训练集比例列表，逗号分隔，如 "0.1,0.3,0.5,0.7,1.0"

    返回:
        Markdown 格式字符串，包含学习曲线数据和分析建议
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y).astype(int)

        # 解析训练集比例
        sizes = [float(s.strip()) for s in train_sizes.split(",") if s.strip()]
        sizes = np.array(sizes)

        model = _get_classifier(model_type)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_data, y_data, cv=cv, train_sizes=sizes, scoring="accuracy"
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        lines = []
        lines.append("## 学习曲线分析\n")
        lines.append(f"**模型类型**: {model_type}")
        lines.append(f"**交叉验证折数**: {cv}")
        lines.append(f"**总样本数**: {X_data.shape[0]}")
        lines.append(f"**特征数**: {X_data.shape[1]}\n")

        lines.append("### 学习曲线数据\n")
        lines.append(format_markdown_table(
            ["训练样本数", "训练分数(均值)", "训练分数(标准差)",
             "验证分数(均值)", "验证分数(标准差)", "差距"],
            [[int(train_sizes_abs[i]),
              format_number(train_mean[i]),
              format_number(train_std[i]),
              format_number(val_mean[i]),
              format_number(val_std[i]),
              format_number(train_mean[i] - val_mean[i])]
             for i in range(len(train_sizes_abs))],
        ))

        # 分析建议
        final_gap = train_mean[-1] - val_mean[-1]
        final_val = val_mean[-1]

        lines.append("\n### 分析建议\n")

        if final_val < 0.7:
            lines.append("- 验证分数较低，模型可能**欠拟合**")
            lines.append("  - 建议: 增加模型复杂度、添加更多特征或使用更强的模型")
        elif final_gap > 0.15:
            lines.append("- 训练分数与验证分数差距较大，模型可能**过拟合**")
            lines.append("  - 建议: 增加训练数据、使用正则化或降低模型复杂度")
        else:
            lines.append("- 训练分数与验证分数接近，模型拟合状态良好")

        # 检查增加数据是否有帮助
        if len(val_mean) >= 2:
            val_improvement = val_mean[-1] - val_mean[0]
            if val_improvement > 0.05:
                lines.append(f"- 验证分数从 {val_mean[0]:.4f} 提升到 {val_mean[-1]:.4f}，"
                             f"增加数据对模型有帮助")
            else:
                lines.append("- 增加训练数据对模型提升有限")

        lines.append(f"\n**最终训练分数**: {train_mean[-1]:.4f}")
        lines.append(f"**最终验证分数**: {val_mean[-1]:.4f}")
        lines.append(f"**过拟合差距**: {final_gap:.4f}")

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"
