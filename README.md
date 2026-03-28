#  Startup Investment Prediction & Clustering (Pakistan)

This project uses machine learning and natural language processing (NLP) to analyze startup data from Pakistan and predict whether a startup is likely to attract investment. It also performs clustering to group startups into meaningful categories.

##  Project Overview

The goal of this project is to:

* Predict **investment potential** of startups based on their descriptions
* Extract insights using **text analysis (TF-IDF)**
* Compare multiple machine learning models
* Group startups into **intelligent clusters** using KMeans


##  Key Features

*  Data cleaning and preprocessing
*  NLP-based feature engineering
*  TF-IDF vectorization for text
*  Model comparison:

* Logistic Regression
 * Perceptron
*  Class imbalance handling using class weights
*  Model evaluation:

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix
*  Feature importance visualization
*  Unsupervised clustering (KMeans)
*  Automatic cluster naming using keyword extraction

## 📁 Dataset

* **Name:** Pakistan Startup Census
* **Source:** Kaggle
* Contains startup information such as:

  * Description
  * Category
  * Founding year
  * Tagline

## ⚙️ Tech Stack

* Python
* pandas, numpy
* scikit-learn
* nltk
* matplotlib, seaborn


## 🔍 How It Works

### 1. Data Preprocessing

* Missing values handled
* Text cleaned (lowercase, stopword removal)
* Feature engineering:

  * Company age
  * Description length

### 2. Label Creation

A custom scoring system assigns:

* `1` → High investment potential
* `0` → Low investment potential

Based on:

* Keywords (fintech, AI, cloud, etc.)
* Description richness

### 3. Model Training

Two models are trained and compared:

* Logistic Regression
* Perceptron

Class imbalance is handled using **balanced class weights**.

### 4. Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

### 5. Feature Importance

Top contributing features are extracted from the trained model.

### 6. Clustering

* KMeans groups startups into 5 clusters
* TF-IDF used for vectorization
* Each cluster is automatically labeled (e.g., FinTech, HealthTech)


## 📈 Sample Outputs

* Confusion Matrix heatmap
* Top 20 important features
* Cluster distribution chart

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/startup-investment-model.git
cd startup-investment-model
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the script

```
python model.py
```

## 📂 Project Structure

```
startup-investment-model/
│
├── model.py
├── Presentation
├── README.md
├── requirements.txt
└── data/
```

##  Future Improvements

*  Hyperparameter tuning
*  Use advanced models (Random Forest, XGBoost)
*  Deploy as a web app (Streamlit)
*  Use real investment data instead of rule-based labels

##  Contributing

Feel free to fork this repository and submit pull requests.

##  Contact

abdulsimple878@gmail.com

##  Acknowledgments

* Dataset from Kaggle
* Inspired by real-world startup investment analysis
