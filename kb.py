from sentence_transformers import SentenceTransformer
import chromadb

# 🔹 Step 1: Your ML Knowledge Base (10 docs)
documents = [
    {
        "id": "doc_001",
        "topic": "XGBoost",
        "text": "XGBoost is an optimized implementation of gradient boosting. It builds trees sequentially and uses regularization to reduce overfitting. It is fast, efficient, and widely used in machine learning competitions."
    },
    {
        "id": "doc_002",
        "topic": "Gradient Boosting",
        "text": "Gradient Boosting builds models sequentially, where each new model corrects the errors of previous ones. It uses gradient descent to minimize loss and can achieve high accuracy but may be slow."
    },
    {
        "id": "doc_003",
        "topic": "Random Forest",
        "text": "Random Forest is an ensemble method that builds multiple decision trees independently and combines their outputs. It reduces overfitting and improves stability."
    },
    {
        "id": "doc_004",
        "topic": "KNN",
        "text": "K-Nearest Neighbors is a simple algorithm that classifies data based on the closest neighbors. It is easy to understand but can be slow for large datasets."
    },
    {
        "id": "doc_005",
        "topic": "SVM",
        "text": "Support Vector Machine finds the optimal hyperplane to separate classes. It works well for high-dimensional data and can use kernel functions for complex boundaries."
    },
    {
        "id": "doc_006",
        "topic": "SHAP",
        "text": "SHAP explains model predictions using Shapley values from game theory. It provides both global and local explanations for model outputs."
    },
    {
        "id": "doc_007",
        "topic": "LIME",
        "text": "LIME explains individual predictions by approximating the model locally. It perturbs input data and observes changes in output."
    },
    {
        "id": "doc_008",
        "topic": "Evaluation Metrics",
        "text": "Classification models use metrics like accuracy, precision, recall, and F1-score. These metrics help evaluate performance, especially in imbalanced datasets."
    },
    {
        "id": "doc_009",
        "topic": "Feature Engineering",
        "text": "Feature engineering involves transforming raw data into useful features. It includes encoding categorical variables, handling missing values, and scaling."
    },
    {
        "id": "doc_010",
        "topic": "Model Comparison",
        "text": "Comparing models involves evaluating them on the same dataset using consistent metrics. XGBoost and Gradient Boosting often outperform simpler models."
    }
]

# 🔹 Step 2: Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Step 3: Create ChromaDB
client = chromadb.Client()
collection = client.create_collection("ml_papers")

# 🔹 Step 4: Convert to embeddings
texts = [d["text"] for d in documents]
embeddings = embedder.encode(texts).tolist()

# 🔹 Step 5: Store in DB
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[d["id"] for d in documents],
    metadatas=[{"topic": d["topic"]} for d in documents]
)

# 🔹 Step 6: Test retrieval
query = "Difference between XGBoost and Gradient Boosting"
q_emb = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=q_emb, n_results=3)

print("\nRetrieved Topics:")
for m in results["metadatas"][0]:
    print("-", m["topic"])