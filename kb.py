from sentence_transformers import SentenceTransformer
import chromadb

# 🔹 Step 1: Your ML Knowledge Base (10 docs)
DOCUMENTS = [
    # ================= CORE ALGORITHMS =================
    {"id": "doc_001", "topic": "XGBoost", "text": "XGBoost is a highly efficient implementation of gradient boosting. It builds decision trees sequentially and minimizes a loss function using gradient descent. It includes regularization (L1 and L2), supports parallel processing, handles missing values automatically, and is known for high performance in structured data tasks."},

    {"id": "doc_002", "topic": "Gradient Boosting", "text": "Gradient Boosting is an ensemble technique where models are built sequentially, and each new model corrects errors made by previous ones. It optimizes a loss function using gradient descent. It is powerful but can overfit if not properly regularized."},

    {"id": "doc_003", "topic": "Random Forest", "text": "Random Forest is an ensemble learning method that constructs multiple decision trees using random subsets of data and features. Predictions are made by averaging or voting. It reduces overfitting and improves generalization compared to a single decision tree."},

    {"id": "doc_004", "topic": "KNN", "text": "K-Nearest Neighbors (KNN) is a non-parametric algorithm that classifies data points based on the majority label of their nearest neighbors. It relies on distance metrics like Euclidean distance and is simple but computationally expensive for large datasets."},

    {"id": "doc_005", "topic": "SVM", "text": "Support Vector Machine (SVM) is a supervised learning algorithm that finds the optimal hyperplane separating classes with maximum margin. It can handle non-linear data using kernel functions such as RBF and polynomial kernels."},

    {"id": "doc_006", "topic": "Decision Trees", "text": "Decision Trees split data based on feature values to make predictions. They are easy to interpret but prone to overfitting. Techniques like pruning and ensemble methods help improve their performance."},

    # ================= DEEP LEARNING =================
    {"id": "doc_007", "topic": "Neural Networks", "text": "Neural Networks consist of layers of interconnected neurons that learn patterns in data through weights and activation functions. They are widely used in deep learning for tasks like image recognition and NLP."},

    {"id": "doc_008", "topic": "CNN", "text": "Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data such as images. They use convolutional layers to extract spatial features and are widely used in computer vision."},

    {"id": "doc_009", "topic": "RNN", "text": "Recurrent Neural Networks (RNNs) are designed for sequential data. They maintain memory of previous inputs, making them useful for time series and natural language processing tasks."},

    {"id": "doc_010", "topic": "Transformers", "text": "Transformers are deep learning models based on self-attention mechanisms. They process entire sequences in parallel and are widely used in NLP tasks like translation, summarization, and chatbots."},

    # ================= EXPLAINABILITY =================
    {"id": "doc_011", "topic": "SHAP", "text": "SHAP (SHapley Additive exPlanations) explains model predictions by assigning each feature an importance value based on cooperative game theory. It provides both global and local interpretability."},

    {"id": "doc_012", "topic": "LIME", "text": "LIME explains individual predictions by approximating a complex model locally with a simpler interpretable model. It perturbs input data and observes output changes."},

    # ================= DATA PROCESSING =================
    {"id": "doc_013", "topic": "Feature Engineering", "text": "Feature Engineering involves transforming raw data into meaningful inputs for models. Techniques include encoding categorical variables, scaling, normalization, feature selection, and creating interaction features."},

    {"id": "doc_014", "topic": "Data Preprocessing", "text": "Data preprocessing includes cleaning data, handling missing values, removing duplicates, normalizing features, and preparing data for machine learning models."},

    {"id": "doc_015", "topic": "Dimensionality Reduction", "text": "Dimensionality reduction reduces the number of features while preserving important information. Techniques include PCA, t-SNE, and UMAP."},

    # ================= EVALUATION =================
    {"id": "doc_016", "topic": "Evaluation Metrics", "text": "Model evaluation uses metrics like accuracy, precision, recall, F1-score, and ROC-AUC. For imbalanced datasets, precision-recall and F1-score are more informative than accuracy."},

    {"id": "doc_017", "topic": "Cross Validation", "text": "Cross-validation splits data into multiple folds to evaluate model performance more reliably. It reduces overfitting and ensures better generalization."},

    # ================= ML CONCEPTS =================
    {"id": "doc_018", "topic": "Overfitting", "text": "Overfitting occurs when a model learns noise instead of patterns, performing well on training data but poorly on unseen data."},

    {"id": "doc_019", "topic": "Underfitting", "text": "Underfitting occurs when a model is too simple to capture patterns in data, resulting in poor performance on both training and test data."},

    {"id": "doc_020", "topic": "Bias-Variance Tradeoff", "text": "The bias-variance tradeoff balances model simplicity and complexity. High bias leads to underfitting, while high variance leads to overfitting."},

    {"id": "doc_021", "topic": "Regularization", "text": "Regularization techniques like L1 and L2 add penalties to model complexity to prevent overfitting."},

    # ================= ADVANCED =================
    {"id": "doc_022", "topic": "Hyperparameter Tuning", "text": "Hyperparameter tuning involves selecting optimal parameters using techniques like Grid Search, Random Search, or Bayesian Optimization."},

    {"id": "doc_023", "topic": "Ensemble Learning", "text": "Ensemble learning combines multiple models to improve performance. Common methods include bagging, boosting, and stacking."},

    {"id": "doc_024", "topic": "Clustering", "text": "Clustering is an unsupervised learning technique that groups similar data points together. Common algorithms include K-Means and DBSCAN."},

    {"id": "doc_025", "topic": "Anomaly Detection", "text": "Anomaly detection identifies rare or unusual patterns in data. It is used in fraud detection, system monitoring, and cybersecurity."},

    {"id": "doc_026", "topic": "Model Comparison", "text": "Model comparison is the process of evaluating multiple machine learning models to determine which performs best for a given task. It requires using the same dataset split, evaluation metrics, and validation strategy for all models to ensure fairness. Common metrics include accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation is often used to obtain reliable performance estimates. Model comparison also considers factors like training time, interpretability, scalability, and robustness, not just accuracy."}
]