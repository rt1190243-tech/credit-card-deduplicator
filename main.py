import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read Dataset
df = pd.read_csv("creditcard.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DESCRIPTIVE ANALYSIS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== BEFORE REMOVING DUPLICATES =====")
print(df.shape)

df = df.drop_duplicates()

print("\n===== AFTER REMOVING DUPLICATES =====")
print(df.shape)

# Feature Engineering
df["Amount_Log"] = df["Amount"] + 1

# Class Distribution
print("\n===== CLASS DISTRIBUTION =====")
print(df["Class"].value_counts())

# Bar Chart
df["Class"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Model Building
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n===== DECISION TREE =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))

