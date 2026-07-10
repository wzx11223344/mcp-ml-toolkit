"""
聚类与降维工具模块

提供6个MCP聚类和降维工具：
- kmeans_clustering: K均值聚类
- dbscan_clustering: DBSCAN密度聚类
- hierarchical_clustering: 层次聚类
- pca_reduction: PCA主成分降维
- tsne_reduction: t-SNE非线性降维
- silhouette_analysis: 轮廓系数分析
"""

import sys
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples

# 兼容直接运行和模块导入两种方式获取 mcp 实例
if "server" in sys.modules:
    from server import mcp
else:
    from __main__ import mcp

from utils import (
    parse_matrix,
    format_number,
    format_array,
    format_markdown_table,
    format_matrix,
)


@mcp.tool()
def kmeans_clustering(
    X: str,
    n_clusters: int = 3,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
) -> str:
    """K均值聚类算法。

    将数据划分为K个簇，通过迭代最小化簇内平方误差和。
    每个簇的中心是该簇所有点的均值。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2;3,4;5,6;7,8"
        n_clusters: 聚类簇数K，默认 3
        init: 初始化方法，"k-means++"或"random"，默认 "k-means++"
        n_init: 不同初始值运行次数，默认 10
        max_iter: 单次运行最大迭代次数，默认 300

    返回:
        Markdown 格式字符串，包含聚类中心、簇标签和评估指标
    """
    try:
        data = parse_matrix(X)

        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
        )
        model.fit(data)

        labels = model.labels_
        centers = model.cluster_centers_
        inertia = model.inertia_

        # 计算轮廓系数（需要至少2个簇且每个簇至少2个样本）
        if n_clusters > 1 and len(set(labels)) > 1:
            sil_score = silhouette_score(data, labels)
        else:
            sil_score = float("nan")

        lines = []
        lines.append("## K均值聚类结果\n")
        lines.append(f"**聚类簇数**: {n_clusters}")
        lines.append(f"**初始化方法**: {init}")
        lines.append(f"**迭代次数**: {n_init}")
        lines.append(f"**样本数**: {data.shape[0]}")
        lines.append(f"**特征数**: {data.shape[1]}")
        lines.append(f"**惯性 (WCSS)**: {inertia:.4f}")
        lines.append(f"**轮廓系数**: {format_number(sil_score)}\n")

        lines.append("### 聚类中心\n")
        lines.append(format_matrix(centers))

        lines.append("\n### 各簇样本统计\n")
        lines.append(format_markdown_table(
            ["簇编号", "样本数", "占比"],
            [[i, int(np.sum(labels == i)), f"{np.sum(labels == i) / len(labels) * 100:.1f}%"]
             for i in range(n_clusters)],
        ))

        lines.append("\n### 各样本聚类标签\n")
        lines.append(format_markdown_table(
            ["样本索引", "簇编号"],
            [[i, int(labels[i])] for i in range(len(labels))],
        ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def dbscan_clustering(
    X: str,
    eps: float = 0.5,
    min_samples: int = 5,
) -> str:
    """DBSCAN密度聚类算法。

    基于密度的空间聚类算法，能够发现任意形状的簇并识别噪声点。
    不需要预先指定簇数。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        eps: 邻域半径，默认 0.5
        min_samples: 核心点的最小邻域样本数，默认 5

    返回:
        Markdown 格式字符串，包含簇标签、噪声点和评估指标
    """
    try:
        data = parse_matrix(X)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        # 计算轮廓系数（排除噪声点，需要至少2个簇）
        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                sil_score = silhouette_score(data[mask], labels[mask])
            else:
                sil_score = float("nan")
        else:
            sil_score = float("nan")

        lines = []
        lines.append("## DBSCAN聚类结果\n")
        lines.append(f"**邻域半径 (eps)**: {eps}")
        lines.append(f"**最小样本数**: {min_samples}")
        lines.append(f"**样本数**: {data.shape[0]}")
        lines.append(f"**特征数**: {data.shape[1]}")
        lines.append(f"**发现簇数**: {n_clusters}")
        lines.append(f"**噪声点数**: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")
        lines.append(f"**轮廓系数**: {format_number(sil_score)}\n")

        lines.append("### 各簇样本统计\n")
        unique_labels = sorted(set(labels))
        lines.append(format_markdown_table(
            ["标签", "说明", "样本数", "占比"],
            [[int(lab), "噪声" if lab == -1 else f"簇{lab}",
              int(np.sum(labels == lab)),
              f"{np.sum(labels == lab) / len(labels) * 100:.1f}%"]
             for lab in unique_labels],
        ))

        lines.append("\n### 各样本聚类标签\n")
        lines.append(format_markdown_table(
            ["样本索引", "簇编号", "是否噪声"],
            [[i, int(labels[i]), "是" if labels[i] == -1 else "否"]
             for i in range(len(labels))],
        ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def hierarchical_clustering(
    X: str,
    n_clusters: int = 2,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> str:
    """层次聚类算法。

    自底向上的凝聚层次聚类，通过逐步合并最相似的簇来构建聚类层次。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        n_clusters: 目标聚类数，默认 2
        linkage: 连接方式，可选 "ward"/"complete"/"average"/"single"，默认 "ward"
        metric: 距离度量（linkage="ward"时必须为"euclidean"），默认 "euclidean"

    返回:
        Markdown 格式字符串，包含簇标签和评估指标
    """
    try:
        data = parse_matrix(X)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )
        labels = model.fit_predict(data)

        # 计算轮廓系数
        if n_clusters > 1 and len(set(labels)) > 1:
            sil_score = silhouette_score(data, labels)
        else:
            sil_score = float("nan")

        lines = []
        lines.append("## 层次聚类结果\n")
        lines.append(f"**聚类数**: {n_clusters}")
        lines.append(f"**连接方式**: {linkage}")
        lines.append(f"**距离度量**: {metric}")
        lines.append(f"**样本数**: {data.shape[0]}")
        lines.append(f"**特征数**: {data.shape[1]}")
        lines.append(f"**实际簇数**: {len(set(labels))}")
        lines.append(f"**轮廓系数**: {format_number(sil_score)}\n")

        lines.append("### 各簇样本统计\n")
        lines.append(format_markdown_table(
            ["簇编号", "样本数", "占比"],
            [[i, int(np.sum(labels == i)),
              f"{np.sum(labels == i) / len(labels) * 100:.1f}%"]
             for i in range(n_clusters)],
        ))

        lines.append("\n### 各样本聚类标签\n")
        lines.append(format_markdown_table(
            ["样本索引", "簇编号"],
            [[i, int(labels[i])] for i in range(len(labels))],
        ))

        # 层次聚类的连接距离
        if hasattr(model, "distances_") and model.distances_ is not None:
            n_merges = len(model.distances_)
            lines.append(f"\n### 合并距离信息（共{n_merges}次合并）\n")
            lines.append(format_markdown_table(
                ["合并序号", "合并距离"],
                [[i + 1, format_number(model.distances_[i])]
                 for i in range(min(n_merges, 20))],
            ))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def pca_reduction(
    X: str,
    n_components: int = 2,
) -> str:
    """PCA主成分分析降维。

    通过线性变换将高维数据投影到低维空间，保留最大方差方向。
    适用于特征压缩、可视化和去噪。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列，如 "1,2,3;4,5,6;7,8,9;10,11,12"
        n_components: 降维后的维度数，默认 2

    返回:
        Markdown 格式字符串，包含主成分、方差解释比和降维后数据
    """
    try:
        data = parse_matrix(X)
        n_features = data.shape[1]
        n_components = min(n_components, n_features)

        model = PCA(n_components=n_components)
        transformed = model.fit_transform(data)

        lines = []
        lines.append("## PCA降维结果\n")
        lines.append(f"**原始维度**: {n_features}")
        lines.append(f"**降维后维度**: {n_components}")
        lines.append(f"**样本数**: {data.shape[0]}\n")

        lines.append("### 方差解释\n")
        lines.append(format_markdown_table(
            ["主成分", "特征值", "方差解释比", "累计方差解释比"],
            [[i + 1, format_number(model.explained_variance_[i]),
              f"{model.explained_variance_ratio_[i] * 100:.2f}%",
              f"{model.explained_variance_ratio_[:i + 1].sum() * 100:.2f}%"]
             for i in range(n_components)],
        ))

        lines.append("\n### 主成分载荷（特征向量）\n")
        lines.append(format_markdown_table(
            ["原始特征"] + [f"PC{i + 1}" for i in range(n_components)],
            [[f"特征{i}"] + [format_number(model.components_[j, i])
              for j in range(n_components)]
             for i in range(n_features)],
        ))

        lines.append("\n### 降维后数据\n")
        lines.append(format_matrix(transformed))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def tsne_reduction(
    X: str,
    n_components: int = 2,
    perplexity: float = 5.0,
    learning_rate: float = 200.0,
    max_iter: int = 1000,
) -> str:
    """t-SNE非线性降维。

    t-分布随机邻域嵌入，特别适合高维数据的可视化。
    通过保持局部结构将高维数据映射到低维空间。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        n_components: 降维后维度，通常2或3，默认 2
        perplexity: 困惑度，影响局部/全局结构的平衡（建议5-50），默认 5.0
        learning_rate: 学习率，默认 200.0
        max_iter: 最大迭代次数，默认 1000

    返回:
        Markdown 格式字符串，包含降维后数据和KL散度
    """
    try:
        data = parse_matrix(X)

        # perplexity 不能超过样本数-1
        perplexity = min(perplexity, max(data.shape[0] - 1, 1))

        # 兼容不同版本 sklearn 的参数名
        try:
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                init="random",
            )
        except TypeError:
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=max_iter,
                init="random",
            )

        transformed = model.fit_transform(data)
        kl_div = model.kl_divergence_

        lines = []
        lines.append("## t-SNE降维结果\n")
        lines.append(f"**原始维度**: {data.shape[1]}")
        lines.append(f"**降维后维度**: {n_components}")
        lines.append(f"**样本数**: {data.shape[0]}")
        lines.append(f"**困惑度**: {perplexity}")
        lines.append(f"**学习率**: {learning_rate}")
        lines.append(f"**最大迭代**: {max_iter}")
        lines.append(f"**KL散度**: {kl_div:.4f}\n")

        lines.append("### 降维后数据\n")
        lines.append(format_matrix(transformed))

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"


@mcp.tool()
def silhouette_analysis(
    X: str,
    max_clusters: int = 10,
    min_clusters: int = 2,
) -> str:
    """轮廓系数分析，确定最佳聚类数。

    对不同K值（从min_clusters到max_clusters）进行K均值聚类，
    计算轮廓系数来评估聚类质量，辅助选择最佳聚类数。

    参数:
        X: 特征矩阵，分号分隔行，逗号分隔列
        max_clusters: 最大测试聚类数，默认 10
        min_clusters: 最小测试聚类数，默认 2

    返回:
        Markdown 格式字符串，包含各K值的轮廓系数和最佳聚类数建议
    """
    try:
        data = parse_matrix(X)
        n_samples = data.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)

        if min_clusters < 2:
            min_clusters = 2
        if max_clusters < min_clusters:
            return f"**错误**: max_clusters({max_clusters}) 必须大于等于 min_clusters({min_clusters})"

        results = []
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(data, labels)
            else:
                sil_score = float("nan")
            results.append((k, sil_score, kmeans.inertia_))

        # 找最佳K
        best_k = max(results, key=lambda x: x[1] if not np.isnan(x[1]) else -1)

        lines = []
        lines.append("## 轮廓系数分析结果\n")
        lines.append(f"**样本数**: {n_samples}")
        lines.append(f"**特征数**: {data.shape[1]}")
        lines.append(f"**测试范围**: {min_clusters} ~ {max_clusters}")
        lines.append(f"**最佳聚类数**: {best_k[0]}")
        lines.append(f"**最佳轮廓系数**: {best_k[1]:.4f}\n")

        lines.append("### 各K值轮廓系数\n")
        lines.append(format_markdown_table(
            ["聚类数K", "轮廓系数", "WCSS(惯性)", "评价"],
            [[k, format_number(sil), format_number(inertia),
              "*** 最佳 ***" if k == best_k[0] else ""]
             for k, sil, inertia in results],
        ))

        lines.append("\n### 轮廓系数解读\n")
        lines.append("- 轮廓系数范围: [-1, 1]")
        lines.append("- 接近 1: 样本聚类合理，簇内紧凑、簇间分离")
        lines.append("- 接近 0: 样本在簇边界上，聚类不明确")
        lines.append("- 接近 -1: 样本可能被分配到错误的簇")
        lines.append(f"\n**建议**: 最佳聚类数为 {best_k[0]}，轮廓系数 {best_k[1]:.4f}")

        return "\n".join(lines)
    except Exception as e:
        return f"**错误**: {str(e)}"
