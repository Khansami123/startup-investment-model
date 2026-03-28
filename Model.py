print("Step 1: Importing libraries...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans

import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

print("Step 1 complete.\n")

print("Step 2: Loading dataset...")

dataset = pd.read_csv("/kaggle/input/datasets/zusmani/pakistan-startup-census/Pakistan Startup Census.csv")

print("Dataset preview:")
print(dataset.head())
print("Step 2 complete.\n")

print("Step 3: Cleaning and preprocessing...")

dataset["Description"] = dataset["Description"].fillna("")
dataset["Category"] = dataset["Category"].fillna("Unspecified")
dataset["Tagline"] = dataset["Tagline"].fillna("")

def extract_year(value):
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group()) if match else np.nan

dataset["Founding_Year"] = dataset["Founded"].apply(extract_year)
dataset["Founding_Year"] = dataset["Founding_Year"].fillna(dataset["Founding_Year"].median())

dataset["Company_Age"] = 2026 - dataset["Founding_Year"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in stopwords.words("english")]
    return " ".join(words)

dataset["Description_Clean"] = dataset["Description"].apply(clean_text)
dataset["Description_Length"] = dataset["Description_Clean"].apply(lambda x: len(x.split()))

print(dataset[["Description_Clean", "Description_Length"]].head())
print("Step 3 complete.\n")

print("Step 4: Creating target label...")

keywords = ["fintech", "ai", "iot", "health", "automation", "cloud"]

def compute_label(row):
    score = 0
    text = row["Description_Clean"]
    for key in keywords:
        if key in text:
            score += 1
    if row["Description_Length"] > 40:
        score += 1
    return 1 if score >= 2 else 0

dataset["Investment_Label"] = dataset.apply(compute_label, axis=1)

print("Label counts:")
print(dataset["Investment_Label"].value_counts())
print("Step 4 complete.\n")
print("Step 5: Splitting dataset & computing class weights...")

features = dataset[["Description_Clean", "Category", "Company_Age", "Description_Length"]]
targets = dataset["Investment_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    targets,
    test_size=0.2,
    random_state=42,
    stratify=targets
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {cls: w for cls, w in zip(np.unique(y_train), weights)}

print("Class Weights:", class_weights)
print("Step 5 complete.\n")
print("Step 6: Setting up transformers...")

text_col = "Description_Clean"
cat_col = ["Category"]
num_cols = ["Company_Age", "Description_Length"]

preprocessor = ColumnTransformer(
    transformers=[
        ("text_features", TfidfVectorizer(max_features=2000), text_col),
        ("cat_features", OneHotEncoder(handle_unknown="ignore"), cat_col),
        ("num_features", "passthrough", num_cols)
    ]
)

print("Step 6 complete.\n")
print("Step 7: Comparing models...")

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=300, class_weight=class_weights))
    ]),
    "Perceptron": Pipeline([
        ("prep", preprocessor),
        ("model", Perceptron(class_weight=class_weights))
    ])
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    results[name] = (acc, prec, rec)
    print(f"{name} -> Accuracy: {acc}, Precision: {prec}, Recall: {rec}\n")

print("Step 7 complete.\n")

print("Step 8: Confusion Matrix...")

best_model = models["Logistic Regression"]
best_model.fit(X_train, y_train)
y_best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_best_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

print("Step 8 complete.\n")

print("Step 9: Feature importance...")

tfidf_names = best_model.named_steps["prep"].named_transformers_["text_features"].get_feature_names_out()
cat_names = best_model.named_steps["prep"].named_transformers_["cat_features"].get_feature_names_out()
all_features = np.concatenate([tfidf_names, cat_names, num_cols])

coef = best_model.named_steps["model"].coef_[0]

feat_df = pd.DataFrame({"Feature": all_features, "Importance": coef})
feat_df["Abs"] = feat_df["Importance"].abs()

top_feats = feat_df.sort_values("Abs", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_feats,
    x="Abs",
    y="Feature",
    hue="Feature",
    legend=False,
    palette="viridis"
)
plt.title("Top 20 Most Important Features")
plt.show()

print("Step 9 complete.\n")

print("Step 10: Performing clustering with meaningful naming...")

from collections import Counter

tfidf_cluster = TfidfVectorizer(max_features=600, stop_words="english")
cluster_matrix = tfidf_cluster.fit_transform(dataset["Description_Clean"])

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
dataset["Cluster_Group"] = kmeans.fit_predict(cluster_matrix)

print("\nExtracting top keywords per cluster...")

cluster_keywords = {}
terms = tfidf_cluster.get_feature_names_out()

for cluster_id in range(5):
    idx = np.where(dataset["Cluster_Group"] == cluster_id)[0]
    tfidf_sums = np.array(cluster_matrix[idx].sum(axis=0)).flatten()
    top_indices = tfidf_sums.argsort()[::-1][:10]
    top_terms = [terms[i] for i in top_indices]
    cluster_keywords[cluster_id] = top_terms
    print(f"Cluster {cluster_id}: {top_terms}")

cluster_names = {}

for cid, keywords in cluster_keywords.items():
    text = " ".join(keywords)

    if any(word in text for word in ["payment", "fintech", "wallet", "bank", "transaction"]):
        cluster_names[cid] = "FinTech & Digital Payments"
    elif any(word in text for word in ["health", "doctor", "clinic", "hospital", "medical"]):
        cluster_names[cid] = "HealthTech & Medical Services"
    elif any(word in text for word in ["shop", "store", "buy", "ecommerce", "market"]):
        cluster_names[cid] = "E‑Commerce & Marketplaces"
    elif any(word in text for word in ["data", "software", "analytics", "cloud", "ai"]):
        cluster_names[cid] = "SaaS / AI / Software Solutions"
    elif any(word in text for word in ["delivery", "service", "logistics", "on-demand"]):
        cluster_names[cid] = "Logistics & On‑Demand Services"
    else:
        cluster_names[cid] = "General Tech / Mixed Category"

dataset["Cluster_Name"] = dataset["Cluster_Group"].map(cluster_names)

print("\nCluster Names Assigned:")
for cid, name in cluster_names.items():
    print(f"Cluster {cid} → {name}")

print("Step 10 complete.\n")


print("Step 11: Generating visual insights...")

plt.figure(figsize=(20, 10))
sns.countplot(
    data=dataset,
    x="Cluster_Name",
    hue="Cluster_Name",       
    legend=False,              
    palette="viridis"
)

plt.title("Startup Clusters Based on Description Similarity (Named Groups)")
plt.xlabel("Cluster Name")
plt.ylabel("Number of Startups")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

print("Updated cluster visual generated with proper naming.")
