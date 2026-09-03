# ============================================================
# EMPLOYEE TURNOVER ANALYTICS PROJECT
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# STEP 1: DATA QUALITY CHECK
# ============================================================

print("\n========================================")
print("STEP 1: DATA QUALITY CHECK")
print("========================================")

df = pd.read_csv("HR_comma_sep.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nEmployee Turnover:")
print(df["left"].value_counts())

print("\nTurnover Percentage:")
print(df["left"].value_counts(normalize=True) * 100)


# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("\nShape After Removing Duplicates:")
print(df.shape)


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n========================================")
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("========================================")


# 1. Employee Turnover
plt.figure(figsize=(6, 4))
df["left"].value_counts().sort_index().plot(kind="bar")
plt.title("Employee Turnover")
plt.xlabel("Employee Left (0 = No, 1 = Yes)")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Salary vs Turnover
plt.figure(figsize=(6, 4))
df.groupby("salary")["left"].mean().plot(kind="bar")
plt.title("Turnover Rate by Salary")
plt.xlabel("Salary")
plt.ylabel("Turnover Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. Department vs Turnover
plt.figure(figsize=(8, 4))
df.groupby("sales")["left"].mean().sort_values().plot(kind="bar")
plt.title("Turnover Rate by Department")
plt.xlabel("Department")
plt.ylabel("Turnover Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 4. Monthly Hours
plt.figure(figsize=(6, 4))
df.groupby("left")["average_montly_hours"].mean().plot(kind="bar")
plt.title("Average Monthly Hours by Turnover")
plt.xlabel("Employee Left (0 = No, 1 = Yes)")
plt.ylabel("Average Monthly Hours")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 5. Satisfaction Level
plt.figure(figsize=(7, 4))
df.groupby("left")["satisfaction_level"].mean().plot(kind="bar")
plt.title("Average Satisfaction Level by Turnover")
plt.xlabel("Employee Left (0 = No, 1 = Yes)")
plt.ylabel("Average Satisfaction")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 3: CLUSTERING
# ============================================================

print("\n========================================")
print("STEP 3: EMPLOYEE CLUSTERING")
print("========================================")

cluster_features = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company"
]

X_cluster = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nEmployee Count in Each Cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nCluster Summary:")
print(df.groupby("cluster")[cluster_features].mean())

print("\nTurnover Rate by Cluster:")
print(df.groupby("cluster")["left"].mean())


# ============================================================
# STEP 4: PREPARE DATA + SMOTE
# ============================================================

print("\n========================================")
print("STEP 4: SMOTE FOR CLASS IMBALANCE")
print("========================================")

# Remove cluster because it is an unsupervised analysis result
ml_df = df.drop(columns=["cluster"])

# Convert categorical columns into numerical values
ml_df = pd.get_dummies(
    ml_df,
    columns=["sales", "salary"],
    drop_first=True
)

X = ml_df.drop("left", axis=1)
y = ml_df["left"]

print("\nOriginal Class Distribution:")
print(y.value_counts())

# Train-test split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# ============================================================
# STEP 5: 5-FOLD CROSS VALIDATION
# ============================================================

print("\n========================================")
print("STEP 5: 5-FOLD CROSS VALIDATION")
print("========================================")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "SVM": SVC()
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = {}

for name, model in models.items():

    # SMOTE inside pipeline prevents data leakage during CV
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    cv_results[name] = scores.mean()

    print("\n" + name)
    print("Fold Scores:", scores)
    print("Mean Accuracy:", round(scores.mean(), 4))


# ============================================================
# STEP 6: BEST MODEL SELECTION
# ============================================================

print("\n========================================")
print("STEP 6: BEST MODEL SELECTION")
print("========================================")

best_model_name = max(
    cv_results,
    key=cv_results.get
)

best_accuracy = cv_results[best_model_name]

print("\nBest Model:")
print(best_model_name)

print("Best Cross Validation Accuracy:")
print(round(best_accuracy, 4))


# Create best model
best_model = models[best_model_name]

best_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", best_model)
])

# Train best model
best_pipeline.fit(X_train, y_train)

# Prediction
y_pred = best_pipeline.predict(X_test)

# Test accuracy
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(round(test_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# STEP 7: EMPLOYEE RETENTION STRATEGIES
# ============================================================

print("\n========================================")
print("STEP 7: EMPLOYEE RETENTION STRATEGIES")
print("========================================")

print("""
1. Improve Employee Satisfaction
   - Conduct regular employee feedback surveys.
   - Identify reasons for low satisfaction.
   - Improve workplace environment.

2. Manage Excessive Workload
   - Monitor employees with very high monthly working hours.
   - Distribute workload fairly.
   - Encourage work-life balance.

3. Review Salary Structure
   - Employees with lower salaries may have higher turnover.
   - Provide competitive salaries and performance incentives.

4. Career Growth Opportunities
   - Provide training and skill-development programs.
   - Create clear promotion paths.

5. Reduce Work Overload
   - Monitor employees handling too many projects.
   - Assign projects according to employee capacity.

6. Recognize Employee Performance
   - Introduce rewards and recognition programs.
   - Appreciate high-performing employees.

7. Focus on High-Risk Employees
   - Use the prediction model to identify employees
     who have a higher probability of leaving.
   - HR can take preventive retention actions.
""")


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n========================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nDataset Size After Cleaning:")
print(df.shape)

print("\nBest Model:")
print(best_model_name)

print("\nBest 5-Fold CV Accuracy:")
print(round(best_accuracy * 100, 2), "%")

print("\nTest Accuracy:")
print(round(test_accuracy * 100, 2), "%")

print("\nAll 7 Tasks Completed Successfully!")