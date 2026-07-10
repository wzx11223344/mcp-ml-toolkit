"""
分类工具模块

提供6个MCP分类工具：
- logistic_regression: 逻辑回归分类
- knn_classifier: K近邻分类
- svm_classifier: 支持向量机分类
- decision_tree_classifier: 决策树分类
- random_forest_classifier: 随机森林分类
- naive_bayes_classifier: 朴素贝叶斯分类
"""

import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

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
)


@mcp.tool()
def logistic_regression(
    X: str,
    y: str,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    X_test: str = "",
) -> str:
    """逻辑回归分类器。

    用于二分类或多分类任务的逻辑回归模型。通过正则化参数控制模型复杂度，
    支持L1、L2和弹性网络正则化。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2;3,4;5,6"
        y: 目标标签，逗号分隔，如 "0,1,0,1"
        C: 正则化强度的倒数，值越小正则化越强，默认 1.0
        penalty: 正则化类型，可选 "l1"/"l2"/"elasticnet"/"none"，默认 "l2"
        solver: 优化算法，可选 "lbfgs"/"liblinear"/"newton-cg"/"sag"/"saga"，默认 "lbfgs"
        max_iter: 最大迭代次数，默认 1000
        X_test: 可选的测试数据特征矩阵，提供则输出预测结果

    返回:
        Markdown 格式字符串，包含模型参数、训练准确率、系数和预测结果
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y).astype(int)

        model = LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=max_iter
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)

        lines = []
        lines.append("## 逻辑回归分类结果\n")
        lines.append(f"**正则化 (C)**: {C}")
        lines.append(f"**惩罚项**: {penalty}")
        lines.append(f"**求解器**: {solver}")
        lines.append(f"**最大迭代**: {max_iter}")
        lines.append(f"**类别数**: {len(model.classes_)}")
        lines.append(f"**类别标签**: {', '.join(str(c) for c in model.classes_)}")
        lines.append(f"**训练准确率**: {train_acc:.4f}\n")

        lines.append("### 模型系数\n")
        if len(model.classes_) == 2:
            lines.append(format_markdown_table(
                ["特征"] + [f"特征{i}" for i in range(X_train.shape[1])],
                [["系数"] + [format_number(v) for v in model.coef_[0]]],
            ))
        else:
            rows = []
            for i, cls in enumerate(model.classes_):
                rows.append([f"类别{cls}"] + [format_number(v) for v in model.coef_[i]])
            lines.append(format_markdown_table(
                ["类别"] + [f"特征{i}" for i in range(X_train.shape[1])],
                rows,
            ))

        lines.append(f"\n**截距**: {format_array(model.intercept_)}")

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_data)
                lines.append("\n### 预测结果\n")
                lines.append(format_markdown_table(
                    ["样本"] + [f"预测类别"] + [f"概率(类别{c})" for c in model.classes_],
                    [[i, int(predictions[i])] + [format_number(p) for p in proba[i]]
                     for i in range(len(predictions))],
                ))
            else:
                lines.append("\n### 预测结果\n")
                lines.append(format_markdown_table(
                    ["样本", "预测类别"],
                    [[i, int(predictions[i])] for i in range(len(predictions))],
                ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def knn_classifier(
    X: str,
    y: str,
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    p: int = 2,
    X_test: str = "",
) -> str:
    """K近邻分类器。

    基于最近邻投票机制进行分类。通过寻找最近的K个样本来决定新样本的类别。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标标签，逗号分隔
        n_neighbors: 近邻数量K，默认 5
        weights: 权重方式，"uniform"为等权重，"distance"为距离倒数加权，默认 "uniform"
        algorithm: 近邻搜索算法，可选 "auto"/"ball_tree"/"kd_tree"/"brute"，默认 "auto"
        p: 距离度量参数，p=1为曼哈顿距离，p=2为欧氏距离，默认 2
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含模型参数、训练准确率和预测结果
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y).astype(int)

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            p=p,
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)

        lines = []
        lines.append("## K近邻分类结果\n")
        lines.append(f"**近邻数 (K)**: {n_neighbors}")
        lines.append(f"**权重方式**: {weights}")
        lines.append(f"**搜索算法**: {algorithm}")
        lines.append(f"**距离参数 (p)**: {p}")
        lines.append(f"**训练样本数**: {X_train.shape[0]}")
        lines.append(f"**特征数**: {X_train.shape[1]}")
        lines.append(f"**类别数**: {len(model.classes_)}")
        lines.append(f"**类别标签**: {', '.join(str(c) for c in model.classes_)}")
        lines.append(f"**训练准确率**: {train_acc:.4f}\n")

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            proba = model.predict_proba(X_test_data)
            lines.append("### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测类别"] + [f"概率(类别{c})" for c in model.classes_],
                [[i, int(predictions[i])] + [format_number(p) for p in proba[i]]
                 for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def svm_classifier(
    X: str,
    y: str,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    degree: int = 3,
    X_test: str = "",
) -> str:
    """支持向量机分类器。

    使用支持向量机进行分类，支持多种核函数。通过寻找最大间隔超平面实现分类。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标标签，逗号分隔
        C: 正则化参数，控制分类错误的惩罚强度，值越大越严格拟合训练数据，默认 1.0
        kernel: 核函数类型，可选 "linear"/"poly"/"rbf"/"sigmoid"，默认 "rbf"
        gamma: 核系数，"scale"或"auto"或浮点数，默认 "scale"
        degree: 多项式核的阶数（仅当kernel="poly"时有效），默认 3
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含模型参数、训练准确率、支持向量信息和预测结果
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y).astype(int)

        model = SVC(
            C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)

        lines = []
        lines.append("## 支持向量机分类结果\n")
        lines.append(f"**正则化参数 (C)**: {C}")
        lines.append(f"**核函数**: {kernel}")
        lines.append(f"**gamma**: {gamma}")
        lines.append(f"**多项式阶数**: {degree}")
        lines.append(f"**类别数**: {len(model.classes_)}")
        lines.append(f"**类别标签**: {', '.join(str(c) for c in model.classes_)}")
        lines.append(f"**训练准确率**: {train_acc:.4f}")
        lines.append(f"**支持向量总数**: {model.n_support_.sum()}")
        lines.append(f"**各类别支持向量数**: {', '.join(str(n) for n in model.n_support_)}\n")

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            proba = model.predict_proba(X_test_data)
            lines.append("### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测类别"] + [f"概率(类别{c})" for c in model.classes_],
                [[i, int(predictions[i])] + [format_number(p) for p in proba[i]]
                 for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def decision_tree_classifier(
    X: str,
    y: str,
    max_depth: int = 0,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    X_test: str = "",
) -> str:
    """决策树分类器。

    使用决策树算法进行分类，通过递归划分特征空间构建树结构。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标标签，逗号分隔
        max_depth: 树的最大深度，0表示不限制，默认 0
        min_samples_split: 分裂节点所需的最小样本数，默认 2
        min_samples_leaf: 叶节点所需的最小样本数，默认 1
        criterion: 分裂标准，"gini"为基尼系数，"entropy"为信息增益，默认 "gini"
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含树结构信息、特征重要性和预测结果
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y).astype(int)

        actual_max_depth = None if max_depth == 0 else max_depth

        model = DecisionTreeClassifier(
            max_depth=actual_max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)

        lines = []
        lines.append("## 决策树分类结果\n")
        lines.append(f"**分裂标准**: {criterion}")
        lines.append(f"**最大深度**: {'不限' if max_depth == 0 else max_depth}")
        lines.append(f"**最小分裂样本数**: {min_samples_split}")
        lines.append(f"**最小叶节点样本数**: {min_samples_leaf}")
        lines.append(f"**实际树深度**: {model.get_depth()}")
        lines.append(f"**叶节点数**: {model.get_n_leaves()}")
        lines.append(f"**类别数**: {len(model.classes_)}")
        lines.append(f"**类别标签**: {', '.join(str(c) for c in model.classes_)}")
        lines.append(f"**训练准确率**: {train_acc:.4f}\n")

        lines.append("### 特征重要性\n")
        lines.append(format_markdown_table(
            ["特征索引", "重要性"],
            [[i, format_number(model.feature_importances_[i])]
             for i in range(X_train.shape[1])],
        ))

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            proba = model.predict_proba(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测类别"] + [f"概率(类别{c})" for c in model.classes_],
                [[i, int(predictions[i])] + [format_number(p) for p in proba[i]]
                 for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def random_forest_classifier(
    X: str,
    y: str,
    n_estimators: int = 100,
    max_depth: int = 0,
    min_samples_split: int = 2,
    max_features: str = "sqrt",
    X_test: str = "",
) -> str:
    """随机森林分类器。

    使用集成学习方法，构建多棵决策树并集成投票进行分类。具有优良的抗过拟合能力。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标标签，逗号分隔
        n_estimators: 决策树数量，默认 100
        max_depth: 每棵树的最大深度，0表示不限制，默认 0
        min_samples_split: 分裂节点所需的最小样本数，默认 2
        max_features: 每次分裂考虑的最大特征数，可选 "sqrt"/"log2"/整数/浮点数，默认 "sqrt"
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含模型参数、特征重要性和预测结果
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y).astype(int)

        actual_max_depth = None if max_depth == 0 else max_depth

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=actual_max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)

        lines = []
        lines.append("## 随机森林分类结果\n")
        lines.append(f"**决策树数量**: {n_estimators}")
        lines.append(f"**最大深度**: {'不限' if max_depth == 0 else max_depth}")
        lines.append(f"**最小分裂样本数**: {min_samples_split}")
        lines.append(f"**最大特征数**: {max_features}")
        lines.append(f"**类别数**: {len(model.classes_)}")
        lines.append(f"**类别标签**: {', '.join(str(c) for c in model.classes_)}")
        lines.append(f"**训练准确率**: {train_acc:.4f}\n")

        lines.append("### 特征重要性\n")
        lines.append(format_markdown_table(
            ["特征索引", "重要性"],
            [[i, format_number(model.feature_importances_[i])]
             for i in range(X_train.shape[1])],
        ))

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            proba = model.predict_proba(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测类别"] + [f"概率(类别{c})" for c in model.classes_],
                [[i, int(predictions[i])] + [format_number(p) for p in proba[i]]
                 for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def naive_bayes_classifier(
    X: str,
    y: str,
    model_type: str = "gaussian",
    alpha: float = 1.0,
    X_test: str = "",
) -> str:
    """朴素贝叶斯分类器。

    基于贝叶斯定理和特征条件独立假设进行分类。支持高斯、多项式和伯努利三种模型。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标标签，逗号分隔
        model_type: 模型类型，"gaussian"为高斯朴素贝叶斯（连续数据），
                    "multinomial"为多项式朴素贝叶斯（计数数据），
                    "bernoulli"为伯努利朴素贝叶斯（二值数据），默认 "gaussian"
        alpha: 拉普拉斯平滑参数（仅multinomial和bernoulli），默认 1.0
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含模型参数和预测结果
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y).astype(int)

        model_type_key = model_type.strip().lower()

        if model_type_key == "gaussian":
            model = GaussianNB()
        elif model_type_key == "multinomial":
            model = MultinomialNB(alpha=alpha)
        elif model_type_key == "bernoulli":
            model = BernoulliNB(alpha=alpha)
        else:
            return f"**错误**: 不支持的模型类型 '{model_type}'，可选: gaussian, multinomial, bernoulli"

        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)

        lines = []
        lines.append("## 朴素贝叶斯分类结果\n")
        lines.append(f"**模型类型**: {model_type_key}")
        if model_type_key != "gaussian":
            lines.append(f"**平滑参数 (alpha)**: {alpha}")
        lines.append(f"**类别数**: {len(model.classes_)}")
        lines.append(f"**类别标签**: {', '.join(str(c) for c in model.classes_)}")
        lines.append(f"**训练准确率**: {train_acc:.4f}\n")

        if model_type_key == "gaussian":
            lines.append("### 高斯模型参数\n")
            lines.append(format_markdown_table(
                ["类别"] + [f"特征{i}_均值" for i in range(X_train.shape[1])],
                [[str(c)] + [format_number(v) for v in model.theta_[idx]]
                 for idx, c in enumerate(model.classes_)],
            ))
            lines.append("")
            lines.append(format_markdown_table(
                ["类别"] + [f"特征{i}_方差" for i in range(X_train.shape[1])],
                [[str(c)] + [format_number(v) for v in model.var_[idx]]
                 for idx, c in enumerate(model.classes_)],
            ))
        else:
            lines.append("### 特征对数概率\n")
            lines.append(format_markdown_table(
                ["类别"] + [f"特征{i}" for i in range(X_train.shape[1])],
                [[str(c)] + [format_number(v) for v in model.feature_log_prob_[idx]]
                 for idx, c in enumerate(model.classes_)],
            ))

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            proba = model.predict_proba(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测类别"] + [f"概率(类别{c})" for c in model.classes_],
                [[i, int(predictions[i])] + [format_number(p) for p in proba[i]]
                 for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"
