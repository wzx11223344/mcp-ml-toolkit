"""
辅助函数和数据预处理工具模块

提供数据解析、格式化等辅助函数，以及4个MCP工具：
- preprocess_data: 数据预处理（标准化/归一化）
- encode_categorical: 分类变量编码
- handle_missing_values: 缺失值处理
- feature_selection: 特征选择
"""

import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.linear_model import LogisticRegression

# 兼容直接运行（python server.py）和模块导入两种方式获取 mcp 实例
if "server" in sys.modules:
    from server import mcp
else:
    from __main__ import mcp


# ==================== 辅助函数 ====================


def parse_array(s):
    """将逗号分隔的字符串解析为 numpy 一维数组。

    例如: "1.0,2.0,3.0" -> array([1.0, 2.0, 3.0])

    参数:
        s: 逗号分隔的数字字符串

    返回:
        numpy.ndarray 一维数组
    """
    values = [x.strip() for x in s.split(",") if x.strip() != ""]
    return np.array([float(x) for x in values])


def parse_matrix(s):
    """将分号分隔的字符串解析为 numpy 二维数组。

    例如: "1,2;3,4" -> array([[1., 2.], [3., 4.]])

    参数:
        s: 分号分隔行、逗号分隔列的字符串

    返回:
        numpy.ndarray 二维数组
    """
    rows = [r.strip() for r in s.split(";") if r.strip() != ""]
    result = []
    for row in rows:
        cols = [c.strip() for c in row.split(",") if c.strip() != ""]
        result.append([float(c) for c in cols])
    return np.array(result)


def parse_matrix_with_nan(s):
    """将分号分隔的字符串解析为 numpy 二维数组，支持 NaN 值。

    字符串中的 "nan" 或 "NaN" 会被解析为 numpy.nan。

    参数:
        s: 分号分隔行、逗号分隔列的字符串

    返回:
        numpy.ndarray 二维数组（可能含 NaN）
    """
    rows = [r.strip() for r in s.split(";") if r.strip() != ""]
    result = []
    for row in rows:
        cols = [c.strip() for c in row.split(",") if c.strip() != ""]
        row_vals = []
        for c in cols:
            if c.lower() == "nan":
                row_vals.append(np.nan)
            else:
                row_vals.append(float(c))
        result.append(row_vals)
    return np.array(result)


def format_number(x, decimals=4):
    """格式化数字为字符串，支持 numpy 标量类型。

    参数:
        x: 要格式化的数字
        decimals: 保留小数位数

    返回:
        格式化后的字符串
    """
    if x is None:
        return "N/A"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        val = float(x)
        if np.isnan(val):
            return "NaN"
        if np.isinf(val):
            return "Inf"
        return f"{val:.{decimals}f}"
    return str(x)


def format_array(arr, decimals=4):
    """将 numpy 数组格式化为逗号分隔的字符串。

    参数:
        arr: numpy 数组
        decimals: 保留小数位数

    返回:
        逗号分隔的字符串
    """
    return ", ".join(format_number(x, decimals) for x in np.ravel(arr))


def format_matrix(mat, decimals=4):
    """将 numpy 二维数组格式化为 Markdown 表格。

    参数:
        mat: numpy 二维数组
        decimals: 保留小数位数

    返回:
        Markdown 格式表格字符串
    """
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    n_cols = mat.shape[1]
    headers = [f"特征{i}" for i in range(n_cols)]
    rows = []
    for i, row in enumerate(mat):
        rows.append([f"样本{i}"] + [format_number(v, decimals) for v in row])
    return format_markdown_table([""] + headers, rows)


def format_markdown_table(headers, rows):
    """生成 Markdown 格式表格。

    参数:
        headers: 表头列表
        rows: 行数据列表，每行是一个列表

    返回:
        Markdown 格式表格字符串
    """
    lines = []
    # 表头行
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    # 分隔行
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    # 数据行
    for row in rows:
        formatted = []
        for v in row:
            if isinstance(v, (int, float, np.integer, np.floating)):
                formatted.append(format_number(v))
            else:
                formatted.append(str(v))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


# ==================== MCP 工具 ====================


@mcp.tool()
def preprocess_data(X: str, method: str = "standard") -> str:
    """对数据进行预处理（标准化、归一化等缩放操作）。

    支持的预处理方法：
    - standard: 标准化（均值为0，方差为1）
    - minmax: 归一化到 [0, 1] 范围
    - maxabs: 最大绝对值缩放到 [-1, 1]
    - robust: 稳健缩放（基于中位数和四分位数）

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2;3,4;5,6"
        method: 预处理方法，可选 "standard"/"minmax"/"maxabs"/"robust"，默认 "standard"

    返回:
        Markdown 格式字符串，包含变换后的数据和缩放器参数
    """
    try:
        data = parse_matrix(X)

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }

        method_key = method.strip().lower()
        if method_key not in scalers:
            return f"**错误**: 不支持的预处理方法 '{method}'，可选: {', '.join(scalers.keys())}"

        scaler = scalers[method_key]()
        transformed = scaler.fit_transform(data)

        lines = []
        lines.append(f"## 数据预处理结果（{method_key}）\n")
        lines.append(f"**原始数据形状**: {data.shape[0]} 行 x {data.shape[1]} 列\n")
        lines.append("### 变换后数据\n")
        lines.append(format_matrix(transformed))

        if hasattr(scaler, "mean_") and scaler.mean_ is not None:
            lines.append("\n### 缩放器参数\n")
            lines.append(format_markdown_table(
                ["参数"] + [f"特征{i}" for i in range(data.shape[1])],
                [["均值"] + [format_number(v) for v in scaler.mean_],
                 ["标准差/缩放"] + [format_number(v) for v in scaler.scale_]],
            ))
        elif hasattr(scaler, "data_min_"):
            lines.append("\n### 缩放器参数\n")
            lines.append(format_markdown_table(
                ["参数"] + [f"特征{i}" for i in range(data.shape[1])],
                [["最小值"] + [format_number(v) for v in scaler.data_min_],
                 ["最大值"] + [format_number(v) for v in scaler.data_max_],
                 ["缩放比例"] + [format_number(v) for v in scaler.scale_]],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def encode_categorical(data: str, method: str = "label") -> str:
    """对分类变量进行编码。

    支持的编码方法：
    - label: 标签编码（将类别映射为整数）
    - onehot: 独热编码（每个类别一个二进制列）
    - ordinal: 序数编码（将类别映射为有序整数）

    参数:
        data: 待编码数据。label 方法使用逗号分隔（如 "cat,dog,bird,cat"）；
              onehot/ordinal 方法使用分号分隔行（如 "cat,dog;bird,cat"）
        method: 编码方法，可选 "label"/"onehot"/"ordinal"，默认 "label"

    返回:
        Markdown 格式字符串，包含编码后的数据和编码映射
    """
    try:
        method_key = method.strip().lower()

        if method_key == "label":
            # 标签编码：一维数据
            values = [x.strip() for x in data.split(",") if x.strip() != ""]
            arr = np.array(values).reshape(-1, 1)

            encoder = LabelEncoder()
            encoded = encoder.fit_transform(values)

            lines = []
            lines.append("## 标签编码结果\n")
            lines.append(f"**编码类别数**: {len(encoder.classes_)}\n")
            lines.append("### 类别映射\n")
            lines.append(format_markdown_table(
                ["原始类别", "编码值"],
                [[cls, code] for cls, code in zip(encoder.classes_, range(len(encoder.classes_)))],
            ))
            lines.append("\n### 编码结果\n")
            lines.append(format_markdown_table(
                ["索引", "原始值", "编码值"],
                [[i, values[i], int(encoded[i])] for i in range(len(values))],
            ))
            return "\n".join(lines)

        elif method_key == "onehot":
            # 独热编码：二维数据
            mat = parse_matrix_str(data)
            n_rows, n_cols = mat.shape

            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(mat)

            lines = []
            lines.append("## 独热编码结果\n")
            lines.append(f"**原始形状**: {n_rows} x {n_cols}")
            lines.append(f"**编码后形状**: {encoded.shape[0]} x {encoded.shape[1]}\n")
            lines.append("### 编码后的数据\n")
            lines.append(format_matrix(encoded))
            lines.append("\n### 特征名称\n")
            for i, name in enumerate(encoder.get_feature_names_out()):
                lines.append(f"- {name}")
            return "\n".join(lines)

        elif method_key == "ordinal":
            # 序数编码：二维数据
            mat = parse_matrix_str(data)
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(mat)

            lines = []
            lines.append("## 序数编码结果\n")
            lines.append(f"**编码形状**: {encoded.shape}\n")
            lines.append("### 编码后的数据\n")
            lines.append(format_matrix(encoded))
            lines.append("\n### 各列类别映射\n")
            for col_idx, categories in enumerate(encoder.categories_):
                lines.append(f"**列 {col_idx}**:")
                lines.append(format_markdown_table(
                    ["原始类别", "编码值"],
                    [[cat, i] for i, cat in enumerate(categories)],
                ))
            return "\n".join(lines)

        else:
            return f"**错误**: 不支持的编码方法 '{method}'，可选: label, onehot, ordinal"

    except Exception as e:
        return f"**错误**: {str(e)}"


def parse_matrix_str(s):
    """将分号分隔的字符串解析为字符串二维数组（用于分类编码）。"""
    rows = [r.strip() for r in s.split(";") if r.strip() != ""]
    result = []
    for row in rows:
        cols = [c.strip() for c in row.split(",") if c.strip() != ""]
        result.append(cols)
    return np.array(result)


@mcp.tool()
def handle_missing_values(data: str, strategy: str = "mean", fill_value: float = 0.0) -> str:
    """处理数据中的缺失值。

    支持的填充策略：
    - mean: 用列均值填充（仅数值型）
    - median: 用列中位数填充
    - most_frequent: 用最频繁值填充
    - constant: 用指定常量填充
    - knn: 用 KNN 近邻值填充

    缺失值在输入中使用 "nan" 表示。

    参数:
        data: 数据矩阵，分号分隔行，逗号分隔列，缺失值用 "nan" 表示，
              如 "1,2,nan;4,nan,6;7,8,9"
        strategy: 填充策略，可选 "mean"/"median"/"most_frequent"/"constant"/"knn"，默认 "mean"
        fill_value: 当 strategy="constant" 时使用的填充值，默认 0.0

    返回:
        Markdown 格式字符串，包含填充后的数据和统计信息
    """
    try:
        mat = parse_matrix_with_nan(data)
        n_missing = int(np.isnan(mat).sum())
        n_total = mat.size

        strategy_key = strategy.strip().lower()

        if strategy_key == "knn":
            imputer = KNNImputer(n_neighbors=5)
            imputed = imputer.fit_transform(mat)
        elif strategy_key in ("mean", "median", "most_frequent", "constant"):
            imputer = SimpleImputer(strategy=strategy_key, fill_value=fill_value)
            imputed = imputer.fit_transform(mat)
        else:
            return f"**错误**: 不支持的策略 '{strategy}'，可选: mean, median, most_frequent, constant, knn"

        lines = []
        lines.append(f"## 缺失值处理结果（策略: {strategy_key}）\n")
        lines.append(f"**数据形状**: {mat.shape[0]} 行 x {mat.shape[1]} 列")
        lines.append(f"**缺失值数量**: {n_missing} / {n_total} ({n_missing/n_total*100:.1f}%)\n")
        lines.append("### 处理后的数据\n")
        lines.append(format_matrix(imputed))

        if hasattr(imputer, "statistics_") and imputer.statistics_ is not None:
            lines.append("\n### 填充值统计\n")
            lines.append(format_markdown_table(
                ["特征"] + [f"特征{i}" for i in range(mat.shape[1])],
                [["填充值"] + [format_number(v) for v in imputer.statistics_]],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def feature_selection(X: str, y: str, k: int = 5, method: str = "f_classif") -> str:
    """进行特征选择，筛选最相关的特征。

    支持的特征选择方法：
    - f_classif: 方差分析 F 检验（适用于分类）
    - mutual_info_classif: 互信息（适用于分类）
    - f_regression: F 检验（适用于回归）
    - mutual_info_regression: 互信息（适用于回归）
    - rfe: 递归特征消除（基于逻辑回归）

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2,3;4,5,6;7,8,9"
        y: 目标向量，逗号分隔，如 "0,1,0"
        k: 选择的特征数量，默认 5
        method: 选择方法，可选 "f_classif"/"mutual_info_classif"/"f_regression"/"mutual_info_regression"/"rfe"

    返回:
        Markdown 格式字符串，包含各特征得分和选中的特征
    """
    try:
        X_data = parse_matrix(X)
        y_data = parse_array(y)
        n_features = X_data.shape[1]
        k = min(k, n_features)

        method_key = method.strip().lower()
        score_funcs = {
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif,
            "f_regression": f_regression,
            "mutual_info_regression": mutual_info_regression,
        }

        lines = []
        lines.append(f"## 特征选择结果（方法: {method_key}）\n")
        lines.append(f"**原始特征数**: {n_features}")
        lines.append(f"**选择特征数**: {k}\n")

        if method_key in score_funcs:
            selector = SelectKBest(score_func=score_funcs[method_key], k=k)
            selector.fit(X_data, y_data)
            scores = selector.scores_
            selected_mask = selector.get_support()

            lines.append("### 各特征得分\n")
            lines.append(format_markdown_table(
                ["特征索引", "得分", "是否选中"],
                [[i, format_number(scores[i]), "是" if selected_mask[i] else "否"]
                 for i in range(n_features)],
            ))

            selected_indices = [i for i in range(n_features) if selected_mask[i]]
            lines.append(f"\n### 选中的特征索引: {selected_indices}")

            # 变换后的数据
            X_selected = selector.transform(X_data)
            lines.append("\n### 变换后数据\n")
            lines.append(format_matrix(X_selected))

        elif method_key == "rfe":
            estimator = LogisticRegression(max_iter=1000)
            selector = RFE(estimator, n_features_to_select=k)
            selector.fit(X_data, y_data)
            ranking = selector.ranking_
            selected_mask = selector.get_support()

            lines.append("### RFE 特征排序\n")
            lines.append(format_markdown_table(
                ["特征索引", "排名", "是否选中"],
                [[i, int(ranking[i]), "是" if selected_mask[i] else "否"]
                 for i in range(n_features)],
            ))

            selected_indices = [i for i in range(n_features) if selected_mask[i]]
            lines.append(f"\n### 选中的特征索引: {selected_indices}")

        else:
            return f"**错误**: 不支持的方法 '{method}'，可选: {', '.join(list(score_funcs.keys()) + ['rfe'])}"

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"
