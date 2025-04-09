# 📊 Customer Churn Prediction using Apache Spark MLlib

## 📌 Objective

Implementing a complete machine learning pipeline with Apache Spark's MLlib in order to forecast customer attrition is the aim of this project. We cover feature engineering, model training, feature selection, hyperparameter adjustment, and data preprocessing.

---

## Task-wise Breakdown with Sample Outputs

---

### Task 1: Data Preprocessing and Feature Engineering

**Description:**
- Filled missing values in `TotalCharges` with `0.0`
- Applied `StringIndexer` and `OneHotEncoder` to categorical columns (`gender`, `PhoneService`, `InternetService`)
- Combined features using `VectorAssembler`

**Sample Output:**
=== Data Preprocessing === Sample processed rows (features and label): [Row(features=SparseVector(10, {0: 1.0, 3: 1.0, 5: 1.0, 7: 0.0, 9: 29.85}), label=0.0), Row(features=SparseVector(10, {1: 1.0, 3: 1.0, 6: 1.0, 8: 0.0, 9: 1889.5}), label=0.0), Row(features=SparseVector(10, {0: 1.0, 4: 1.0, 5: 1.0, 7: 0.0, 9: 108.15}), label=1.0)]


---

### Task 2: Train and Evaluate Logistic Regression Model

**Description:**
- Split the dataset (80% train / 20% test)
- Trained a `LogisticRegression` model
- Evaluated model using AUC score

**Sample Output:**
=== Logistic Regression === AUC: 0.7772


---

### Task 3: Feature Selection using Chi-Square Test

**Description:**
- Used `ChiSqSelector` to choose top 5 most relevant features

**Sample Output:**
=== Feature Selection (Chi-Square) === Top 5 selected features (first 5 rows): [Row(selectedFeatures=SparseVector(5, {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 29.85}), label=0.0), Row(selectedFeatures=SparseVector(5, {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 1889.5}), label=0.0), Row(selectedFeatures=SparseVector(5, {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 108.15}), label=1.0)]


---

### Task 4: Hyperparameter Tuning and Model Comparison

**Description:**
- Tuned hyperparameters using `CrossValidator` (5-fold cross-validation)
- Compared 4 classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosted Trees
- Evaluated models using AUC

**Sample Output:**
=== Model Tuning and Comparison === LogisticRegression AUC: 0.7730 DecisionTree AUC: 0.7290 RandomForest AUC: 0.8448 GBTClassifier AUC: 0.7620 Best model: RandomForest with AUC = 0.8448


---

## Steps to Run

### Step 1: Install Dependencies
```bash
pip install pyspark
```
Step 2: (Optional) Generate Dataset
```bash
python dataset-generator.py
```
Step 3: Check Python Version
```bash
python --version
```
Step 4: Run the Analysis Script
```bash
spark-submit customer-churn-analysis.py
```
📂 Output File

All task outputs are saved to a file:

model_outputs.txt
It includes:

Feature vectors
AUC scores
Selected features
Model comparison results
