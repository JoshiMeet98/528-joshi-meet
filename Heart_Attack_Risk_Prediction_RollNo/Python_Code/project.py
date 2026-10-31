import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# loading dataset from Dataset folder
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_file = os.path.join(base_dir, "..", "Dataset", "Medicaldataset.csv")

plots_folder = os.path.join(base_dir, "..", "Reports", "plots")
os.makedirs(plots_folder, exist_ok=True)

df = pd.read_csv(dataset_file)

# outlier handling using IQR
num_cols = ["Age", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]
df_clean = df.copy()

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    
    # capping extreme outliers
    p1 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df_clean[col] = df_clean[col].clip(p1, p99)

# creating eda plots
# 1. histogram
plt.figure(figsize=(10, 6))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2, 3, i)
    plt.hist(df[col], bins=20, color="skyblue", edgecolor="black")
    plt.title(col)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "histogram.png"))
plt.close()

# 2. box plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.boxplot(x="Result", y="Troponin", data=df)
plt.title("Troponin")

plt.subplot(1, 3, 2)
sns.boxplot(x="Result", y="CK-MB", data=df)
plt.title("CK-MB")

plt.subplot(1, 3, 3)
sns.boxplot(x="Result", y="Blood sugar", data=df)
plt.title("Blood Sugar")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "boxplot.png"))
plt.close()

# 3. pie chart
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
df["Result"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["tomato", "lightgreen"], startangle=90)
plt.title("Heart Attack (Positive vs Negative)")
plt.ylabel("")

plt.subplot(1, 2, 2)
df["Gender"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["skyblue", "orange"], startangle=90)
plt.title("Gender (1=Male, 0=Female)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "pie_chart.png"))
plt.close()

# 4. bar chart
plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", hue="Result", data=df)
plt.title("Heart Attack Cases by Gender")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "bar_chart.png"))
plt.close()

# 5. correlation heatmap
plt.figure(figsize=(8, 6))
corr_data = df_clean.copy()
corr_data["Result"] = (corr_data["Result"] == "positive").astype(int)
sns.heatmap(corr_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "correlation_matrix.png"))
plt.close()

# splitting dataset into features and target
x = df_clean.drop(columns=["Result"])
y = (df_clean["Result"] == "positive").astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# scaling features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# training logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(x_train_scaled, y_train)

# evaluate model
y_pred = model.predict(x_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=5, scoring="accuracy")

# user input for patient prediction
print("=== Heart Attack Risk Prediction ===")
print("Enter patient medical details to analyze risk based on dataset:\n")

try:
    age = float(input("Enter Age: "))
    gender = int(input("Enter Gender (1 for Male, 0 for Female): "))
    hr = float(input("Enter Heart Rate (bpm): "))
    sbp = float(input("Enter Systolic Blood Pressure: "))
    dbp = float(input("Enter Diastolic Blood Pressure: "))
    bs = float(input("Enter Blood Sugar (mg/dL): "))
    ckmb = float(input("Enter CK-MB cardiac marker: "))
    troponin = float(input("Enter Troponin level: "))

    # user input dataframe
    user_df = pd.DataFrame([[age, gender, hr, sbp, dbp, bs, ckmb, troponin]], columns=x.columns)
    user_scaled = scaler.transform(user_df)

    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1] * 100

    print("\n--- Prediction Result ---")
    if pred == 1:
        print(f"Prediction: Heart Attack / At Risk")
        print(f"Risk Chance: {prob:.1f}%")
    else:
        print(f"Prediction: No Heart Attack / Not At Risk")
        print(f"Risk Chance: {prob:.1f}%")

    # analyzing user input based on dataset stats
    print("\n--- Data Analysis based on Dataset ---")
    pos_data = df[df["Result"] == "positive"]
    neg_data = df[df["Result"] == "negative"]

    if troponin > neg_data["Troponin"].mean():
        print(f"- Your Troponin is {troponin}. In dataset, safe patients average {neg_data['Troponin'].mean():.3f}, while heart attack cases average {pos_data['Troponin'].mean():.3f}. (High risk)")
    else:
        print(f"- Your Troponin ({troponin}) is within normal safe range.")

    if ckmb > neg_data["CK-MB"].mean():
        print(f"- Your CK-MB is {ckmb}. This is above normal average ({neg_data['CK-MB'].mean():.1f}).")
    else:
        print(f"- Your CK-MB ({ckmb}) is normal.")

    if sbp > df["Systolic blood pressure"].mean():
        print(f"- Systolic BP ({sbp:.0f}) is higher than dataset average ({df['Systolic blood pressure'].mean():.0f} mmHg).")
    else:
        print(f"- Blood pressure ({sbp:.0f}/{dbp:.0f} mmHg) is normal.")

except ValueError:
    print("Invalid input. Please enter valid numbers.")

# showing model evaluation metrics
print("\n--- Model Performance on Dataset ---")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall: {rec * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"5-Fold Cross Validation: {cv_scores.mean() * 100:.2f}%")
