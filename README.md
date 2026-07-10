# MCP 机器学习工具包

基于 [FastMCP](https://github.com/jlowin/fastmcp) 框架和 [scikit-learn](https://scikit-learn.org/) 构建的机器学习 MCP 服务器，提供 **30 个 MCP 工具**，覆盖分类、回归、聚类、降维、模型评估和数据预处理等核心机器学习任务。

## 特性

- **30 个 MCP 工具**：涵盖机器学习全流程
- **FastMCP 框架**：标准 MCP 协议，兼容主流 MCP 客户端
- **scikit-learn 引擎**：业界标准机器学习库
- **Markdown 输出**：所有工具返回格式化的 Markdown 字符串
- **灵活参数传递**：数组用逗号分隔，矩阵用分号分隔，方便通过 MCP 协议调用
- **错误处理**：每个工具内置异常捕获，返回友好的错误信息

## 安装

```bash
# 克隆项目
git clone <repository_url>
cd mcp-ml-toolkit

# 安装依赖
pip install -r requirements.txt
```

## 使用

### 直接运行

```bash
python server.py
```

### 配置到 MCP 客户端

在 MCP 客户端配置文件中添加：

```json
{
  "mcpServers": {
    "ml-toolkit": {
      "command": "python",
      "args": ["path/to/server.py"]
    }
  }
}
```

### 使用示例

所有工具通过 MCP 协议调用，参数以字符串形式传递：

**逻辑回归分类：**
```
X = "1,2;3,4;5,6;7,8;9,10;11,12"
y = "0,0,1,1,0,1"
调用: logistic_regression(X, y, C=1.0, penalty="l2")
```

**K均值聚类：**
```
X = "1,2;1.5,1.8;5,8;8,8;1,0.6;9,11"
调用: kmeans_clustering(X, n_clusters=2)
```

**PCA降维：**
```
X = "2.5,2.4,2.1;0.5,0.7,4.2;2.2,2.9,1.5;1.9,2.2,3.3;3.1,3.0,2.7"
调用: pca_reduction(X, n_components=2)
```

**交叉验证：**
```
X = "1,2;3,4;5,6;7,8"
y = "0,0,1,1"
调用: cross_validate_model(X, y, model_type="svm", cv=5)
```

**数据预处理：**
```
X = "1,2,3;4,5,6;7,8,9"
调用: preprocess_data(X, method="standard")
```

## 参数格式约定

| 数据类型 | 格式 | 示例 |
|---------|------|------|
| 一维数组 | 逗号分隔 | `"1.0,2.0,3.0"` |
| 二维矩阵 | 分号分隔行，逗号分隔列 | `"1,2;3,4;5,6"` |
| 缺失值 | 使用 `nan` 表示 | `"1,nan,3;4,5,nan"` |
| 参数网格 | 参数名:值1,值2;参数名2:值1,值2 | `"C:0.1,1,10;kernel:rbf,linear"` |

## 工具列表

### 分类工具 (6个)

| 工具 | 说明 | 算法 |
|------|------|------|
| `logistic_regression` | 逻辑回归分类 | L1/L2正则化，多求解器 |
| `knn_classifier` | K近邻分类 | 支持加权、多种距离度量 |
| `svm_classifier` | 支持向量机分类 | linear/poly/rbf/sigmoid核 |
| `decision_tree_classifier` | 决策树分类 | gini/entropy分裂标准 |
| `random_forest_classifier` | 随机森林分类 | 集成学习，特征重要性 |
| `naive_bayes_classifier` | 朴素贝叶斯分类 | Gaussian/Multinomial/Bernoulli |

### 回归工具 (6个)

| 工具 | 说明 | 算法 |
|------|------|------|
| `linear_regression` | 线性回归 | 最小二乘法 |
| `ridge_regression` | 岭回归 | L2正则化 |
| `lasso_regression` | Lasso回归 | L1正则化，特征选择 |
| `polynomial_regression` | 多项式回归 | 非线性特征展开 |
| `decision_tree_regressor` | 决策树回归 | 递归划分 |
| `random_forest_regressor` | 随机森林回归 | 集成学习 |

### 聚类与降维工具 (6个)

| 工具 | 说明 | 算法 |
|------|------|------|
| `kmeans_clustering` | K均值聚类 | 迭代优化 |
| `dbscan_clustering` | DBSCAN密度聚类 | 基于密度，自动识别噪声 |
| `hierarchical_clustering` | 层次聚类 | 凝聚式层次聚类 |
| `pca_reduction` | PCA主成分降维 | 线性降维 |
| `tsne_reduction` | t-SNE降维 | 非线性降维，可视化 |
| `silhouette_analysis` | 轮廓系数分析 | 最佳聚类数选择 |

### 模型评估工具 (8个)

| 工具 | 说明 | 功能 |
|------|------|------|
| `train_test_split_data` | 数据集分割 | 支持分层抽样 |
| `cross_validate_model` | 交叉验证 | K折交叉验证 |
| `classification_report_tool` | 分类报告 | 精确率/召回率/F1 |
| `confusion_matrix_tool` | 混淆矩阵 | 含二分类详细指标 |
| `roc_curve_analysis` | ROC曲线分析 | AUC/最佳阈值 |
| `feature_importance_analysis` | 特征重要性 | 排名与选择建议 |
| `hyperparameter_grid_search` | 网格搜索调参 | 自动搜索最佳参数 |
| `learning_curve_analysis` | 学习曲线 | 欠拟合/过拟合诊断 |

### 数据预处理工具 (4个)

| 工具 | 说明 | 功能 |
|------|------|------|
| `preprocess_data` | 数据缩放 | Standard/MinMax/MaxAbs/Robust |
| `encode_categorical` | 分类编码 | Label/OneHot/Ordinal |
| `handle_missing_values` | 缺失值处理 | Mean/Median/MostFrequent/Constant/KNN |
| `feature_selection` | 特征选择 | F检验/互信息/RFE |

## 项目结构

```
mcp-ml-toolkit/
├── server.py                  # 主入口，FastMCP服务器
├── classification_tools.py    # 分类工具 (6个)
├── regression_tools.py        # 回归工具 (6个)
├── clustering_tools.py        # 聚类降维工具 (6个)
├── evaluation_tools.py        # 模型评估工具 (8个)
├── utils.py                   # 辅助函数 + 数据预处理工具 (4个)
├── requirements.txt           # Python依赖
├── README.md                  # 项目文档
└── SKILL.md                   # SkillHub技能描述
```

## 技术栈

- **[MCP](https://modelcontextprotocol.io/)** - Model Context Protocol，AI模型上下文协议
- **[FastMCP](https://github.com/jlowin/fastmcp)** - MCP Python SDK，快速构建MCP服务器
- **[scikit-learn](https://scikit-learn.org/)** - 机器学习算法库
- **[NumPy](https://numpy.org/)** - 数值计算
- **[Pandas](https://pandas.pydata.org/)** - 数据处理
- **[SciPy](https://scipy.org/)** - 科学计算

## 许可证

MIT License
