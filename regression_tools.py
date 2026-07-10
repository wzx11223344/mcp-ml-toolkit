"""
回归工具模块

提供6个MCP回归工具：
- linear_regression: 线性回归
- ridge_regression: 岭回归
- lasso_regression: Lasso回归
- polynomial_regression: 多项式回归
- decision_tree_regressor: 决策树回归
- random_forest_regressor: 随机森林回归
"""

import sys
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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


def _format_regression_metrics(model, X_train, y_train, lines):
    """辅助函数：格式化回归评估指标。"""
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mse)

    lines.append(f"**R^2 决定系数**: {r2:.4f}")
    lines.append(f"**均方误差 (MSE)**: {mse:.4f}")
    lines.append(f"**均方根误差 (RMSE)**: {rmse:.4f}")
    lines.append(f"**平均绝对误差 (MAE)**: {mae:.4f}")
    return lines


@mcp.tool()
def linear_regression(X: str, y: str, X_test: str = "") -> str:
    """线性回归模型。

    使用最小二乘法拟合线性回归模型，建立特征与目标之间的线性关系。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2;3,4;5,6"
        y: 目标值向量，逗号分隔，如 "10,20,30"
        X_test: 可选的测试数据特征矩阵，提供则输出预测结果

    返回:
        Markdown 格式字符串，包含回归系数、截距和评估指标
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y)

        model = LinearRegression()
        model.fit(X_train, y_train)

        lines = []
        lines.append("## 线性回归结果\n")
        lines.append(f"**样本数**: {X_train.shape[0]}")
        lines.append(f"**特征数**: {X_train.shape[1]}\n")

        lines.append("### 模型参数\n")
        lines.append(format_markdown_table(
            ["特征"] + [f"特征{i}" for i in range(X_train.shape[1])],
            [["系数"] + [format_number(v) for v in model.coef_]],
        ))
        lines.append(f"\n**截距**: {format_number(model.intercept_)}\n")

        lines.append("### 评估指标\n")
        _format_regression_metrics(model, X_train, y_train, lines)

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测值"],
                [[i, format_number(predictions[i])] for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def ridge_regression(
    X: str,
    y: str,
    alpha: float = 1.0,
    solver: str = "auto",
    X_test: str = "",
) -> str:
    """岭回归（L2正则化）模型。

    在线性回归基础上添加L2正则化项，防止过拟合，适用于特征间存在共线性的场景。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标值向量，逗号分隔
        alpha: L2正则化强度，值越大正则化越强，默认 1.0
        solver: 求解方法，可选 "auto"/"svd"/"cholesky"/"lsqr"/"sparse_cg"/"sag"/"saga"，默认 "auto"
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含回归系数、截距和评估指标
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y)

        model = Ridge(alpha=alpha, solver=solver)
        model.fit(X_train, y_train)

        lines = []
        lines.append("## 岭回归结果\n")
        lines.append(f"**正则化强度 (alpha)**: {alpha}")
        lines.append(f"**求解方法**: {solver}")
        lines.append(f"**样本数**: {X_train.shape[0]}")
        lines.append(f"**特征数**: {X_train.shape[1]}\n")

        lines.append("### 模型参数\n")
        lines.append(format_markdown_table(
            ["特征"] + [f"特征{i}" for i in range(X_train.shape[1])],
            [["系数"] + [format_number(v) for v in model.coef_]],
        ))
        lines.append(f"\n**截距**: {format_number(model.intercept_)}\n")

        lines.append("### 评估指标\n")
        _format_regression_metrics(model, X_train, y_train, lines)

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测值"],
                [[i, format_number(predictions[i])] for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def lasso_regression(
    X: str,
    y: str,
    alpha: float = 1.0,
    max_iter: int = 1000,
    X_test: str = "",
) -> str:
    """Lasso回归（L1正则化）模型。

    在线性回归基础上添加L1正则化项，能够产生稀疏权重，实现特征选择。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标值向量，逗号分隔
        alpha: L1正则化强度，值越大稀疏性越强，默认 1.0
        max_iter: 最大迭代次数，默认 1000
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含回归系数、非零特征数和评估指标
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y)

        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X_train, y_train)

        n_nonzero = np.sum(model.coef_ != 0)

        lines = []
        lines.append("## Lasso回归结果\n")
        lines.append(f"**正则化强度 (alpha)**: {alpha}")
        lines.append(f"**最大迭代次数**: {max_iter}")
        lines.append(f"**样本数**: {X_train.shape[0]}")
        lines.append(f"**特征数**: {X_train.shape[1]}")
        lines.append(f"**非零系数数**: {n_nonzero}")
        lines.append(f"**稀疏率**: {1 - n_nonzero / X_train.shape[1]:.2%}\n")

        lines.append("### 模型参数\n")
        lines.append(format_markdown_table(
            ["特征索引", "系数", "是否非零"],
            [[i, format_number(model.coef_[i]), "是" if model.coef_[i] != 0 else "否"]
             for i in range(X_train.shape[1])],
        ))
        lines.append(f"\n**截距**: {format_number(model.intercept_)}\n")

        lines.append("### 评估指标\n")
        _format_regression_metrics(model, X_train, y_train, lines)

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测值"],
                [[i, format_number(predictions[i])] for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def polynomial_regression(
    X: str,
    y: str,
    degree: int = 2,
    include_bias: bool = False,
    X_test: str = "",
) -> str:
    """多项式回归模型。

    通过生成多项式特征进行非线性回归。将原始特征进行多项式展开后使用线性回归拟合。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标值向量，逗号分隔
        degree: 多项式阶数，默认 2
        include_bias: 是否包含偏置项（常数项），默认 False
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含多项式特征数、系数和评估指标
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y)

        poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly_features.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_poly, y_train)

        lines = []
        lines.append("## 多项式回归结果\n")
        lines.append(f"**多项式阶数**: {degree}")
        lines.append(f"**包含偏置**: {include_bias}")
        lines.append(f"**原始特征数**: {X_train.shape[1]}")
        lines.append(f"**展开后特征数**: {X_poly.shape[1]}")
        lines.append(f"**样本数**: {X_train.shape[0]}\n")

        lines.append("### 模型系数\n")
        feature_names = poly_features.get_feature_names_out()
        lines.append(format_markdown_table(
            ["特征名称", "系数"],
            [[name, format_number(coef)] for name, coef in zip(feature_names, model.coef_)],
        ))
        lines.append(f"\n**截距**: {format_number(model.intercept_)}\n")

        lines.append("### 评估指标\n")
        y_pred = model.predict(X_poly)
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        lines.append(f"**R^2 决定系数**: {r2:.4f}")
        lines.append(f"**均方误差 (MSE)**: {mse:.4f}")
        lines.append(f"**均方根误差 (RMSE)**: {np.sqrt(mse):.4f}")
        lines.append(f"**平均绝对误差 (MAE)**: {mae:.4f}")

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            X_test_poly = poly_features.transform(X_test_data)
            predictions = model.predict(X_test_poly)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测值"],
                [[i, format_number(predictions[i])] for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def decision_tree_regressor(
    X: str,
    y: str,
    max_depth: int = 0,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    X_test: str = "",
) -> str:
    """决策树回归模型。

    使用决策树算法进行回归预测，通过递归划分特征空间构建回归树。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标值向量，逗号分隔
        max_depth: 树的最大深度，0表示不限制，默认 0
        min_samples_split: 分裂节点所需的最小样本数，默认 2
        min_samples_leaf: 叶节点所需的最小样本数，默认 1
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含树结构信息、特征重要性和评估指标
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y)

        actual_max_depth = None if max_depth == 0 else max_depth

        model = DecisionTreeRegressor(
            max_depth=actual_max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        model.fit(X_train, y_train)

        lines = []
        lines.append("## 决策树回归结果\n")
        lines.append(f"**最大深度**: {'不限' if max_depth == 0 else max_depth}")
        lines.append(f"**最小分裂样本数**: {min_samples_split}")
        lines.append(f"**最小叶节点样本数**: {min_samples_leaf}")
        lines.append(f"**实际树深度**: {model.get_depth()}")
        lines.append(f"**叶节点数**: {model.get_n_leaves()}")
        lines.append(f"**样本数**: {X_train.shape[0]}")
        lines.append(f"**特征数**: {X_train.shape[1]}\n")

        lines.append("### 特征重要性\n")
        lines.append(format_markdown_table(
            ["特征索引", "重要性"],
            [[i, format_number(model.feature_importances_[i])]
             for i in range(X_train.shape[1])],
        ))
        lines.append("")

        lines.append("### 评估指标\n")
        _format_regression_metrics(model, X_train, y_train, lines)

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测值"],
                [[i, format_number(predictions[i])] for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def random_forest_regressor(
    X: str,
    y: str,
    n_estimators: int = 100,
    max_depth: int = 0,
    min_samples_split: int = 2,
    max_features: str = "sqrt",
    X_test: str = "",
) -> str:
    """随机森林回归模型。

    使用集成学习方法，构建多棵决策树进行回归预测。具有优良的抗过拟合能力。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        y: 目标值向量，逗号分隔
        n_estimators: 决策树数量，默认 100
        max_depth: 每棵树的最大深度，0表示不限制，默认 0
        min_samples_split: 分裂节点所需的最小样本数，默认 2
        max_features: 每次分裂考虑的最大特征数，可选 "sqrt"/"log2"/整数/浮点数，默认 "sqrt"
        X_test: 可选的测试数据特征矩阵

    返回:
        Markdown 格式字符串，包含模型参数、特征重要性和评估指标
    """
    try:
        X_train = parse_matrix(X)
        y_train = parse_array(y)

        actual_max_depth = None if max_depth == 0 else max_depth

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=actual_max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        model.fit(X_train, y_train)

        lines = []
        lines.append("## 随机森林回归结果\n")
        lines.append(f"**决策树数量**: {n_estimators}")
        lines.append(f"**最大深度**: {'不限' if max_depth == 0 else max_depth}")
        lines.append(f"**最小分裂样本数**: {min_samples_split}")
        lines.append(f"**最大特征数**: {max_features}")
        lines.append(f"**样本数**: {X_train.shape[0]}")
        lines.append(f"**特征数**: {X_train.shape[1]}\n")

        lines.append("### 特征重要性\n")
        lines.append(format_markdown_table(
            ["特征索引", "重要性"],
            [[i, format_number(model.feature_importances_[i])]
             for i in range(X_train.shape[1])],
        ))
        lines.append("")

        lines.append("### 评估指标\n")
        _format_regression_metrics(model, X_train, y_train, lines)

        if X_test.strip():
            X_test_data = parse_matrix(X_test)
            predictions = model.predict(X_test_data)
            lines.append("\n### 预测结果\n")
            lines.append(format_markdown_table(
                ["样本", "预测值"],
                [[i, format_number(predictions[i])] for i in range(len(predictions))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"
