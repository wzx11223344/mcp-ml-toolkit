---
name: mcp-ml-toolkit-zx
displayName: 机器学习MCP服务器
version: 1.0.1
summary: 30个MCP工具：逻辑回归/KNN/SVM/决策树/随机森林/朴素贝叶斯/线性回归/Ridge/Lasso/K-Means/DBSCAN/PCA/t-SNE/交叉验证/ROC/网格搜索
tags: [mcp, machine-learning, scikit-learn, classification]
license: MIT
---

# 机器学习 MCP 服务器

基于 FastMCP 框架构建的机器学习工具包服务器，提供 30 个 MCP 工具，覆盖分类、回归、聚类、降维、模型评估和数据预处理等核心机器学习任务。

## 工具总览

### 分类工具 (6个)
- `logistic_regression` - 逻辑回归分类
- `knn_classifier` - K近邻分类
- `svm_classifier` - 支持向量机分类
- `decision_tree_classifier` - 决策树分类
- `random_forest_classifier` - 随机森林分类
- `naive_bayes_classifier` - 朴素贝叶斯分类

### 回归工具 (6个)
- `linear_regression` - 线性回归
- `ridge_regression` - 岭回归
- `lasso_regression` - Lasso回归
- `polynomial_regression` - 多项式回归
- `decision_tree_regressor` - 决策树回归
- `random_forest_regressor` - 随机森林回归

### 聚类与降维工具 (6个)
- `kmeans_clustering` - K均值聚类
- `dbscan_clustering` - DBSCAN密度聚类
- `hierarchical_clustering` - 层次聚类
- `pca_reduction` - PCA主成分降维
- `tsne_reduction` - t-SNE非线性降维
- `silhouette_analysis` - 轮廓系数分析

### 模型评估工具 (8个)
- `train_test_split_data` - 数据集分割
- `cross_validate_model` - 交叉验证
- `classification_report_tool` - 分类报告
- `confusion_matrix_tool` - 混淆矩阵
- `roc_curve_analysis` - ROC曲线分析
- `feature_importance_analysis` - 特征重要性分析
- `hyperparameter_grid_search` - 网格搜索调参
- `learning_curve_analysis` - 学习曲线分析

### 数据预处理工具 (4个)
- `preprocess_data` - 数据标准化/归一化
- `encode_categorical` - 分类变量编码
- `handle_missing_values` - 缺失值处理
- `feature_selection` - 特征选择

## 参数格式约定

- 一维数组: 逗号分隔，如 `"1.0,2.0,3.0"`
- 二维矩阵: 分号分隔行，逗号分隔列，如 `"1,2;3,4;5,6"`
- 所有工具返回 Markdown 格式字符串

## 使用方式

```bash
python server.py
```

或配置到 MCP 客户端中运行。
