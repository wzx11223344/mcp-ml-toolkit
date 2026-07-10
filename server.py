"""
MCP 机器学习工具包服务器

主入口文件，使用 FastMCP 框架创建 MCP 服务器。
提供 30 个机器学习相关的 MCP 工具，覆盖分类、回归、聚类、降维、模型评估和数据预处理。

使用方式:
    python server.py

工具总览:
    - 分类工具 (6): logistic_regression, knn_classifier, svm_classifier,
      decision_tree_classifier, random_forest_classifier, naive_bayes_classifier
    - 回归工具 (6): linear_regression, ridge_regression, lasso_regression,
      polynomial_regression, decision_tree_regressor, random_forest_regressor
    - 聚类降维工具 (6): kmeans_clustering, dbscan_clustering, hierarchical_clustering,
      pca_reduction, tsne_reduction, silhouette_analysis
    - 模型评估工具 (8): train_test_split_data, cross_validate_model, classification_report_tool,
      confusion_matrix_tool, roc_curve_analysis, feature_importance_analysis,
      hyperparameter_grid_search, learning_curve_analysis
    - 数据预处理工具 (4): preprocess_data, encode_categorical, handle_missing_values,
      feature_selection
"""

from mcp.server.fastmcp import FastMCP

# 创建 FastMCP 服务器实例
mcp = FastMCP("ml-toolkit")

# 导入工具模块以注册所有 MCP 工具
# utils 必须最先导入，因为其他模块依赖其辅助函数
import utils
import classification_tools
import regression_tools
import clustering_tools
import evaluation_tools

# 模块导入时会自动通过 @mcp.tool() 装饰器注册工具
# 共注册 30 个 MCP 工具

if __name__ == "__main__":
    mcp.run()
