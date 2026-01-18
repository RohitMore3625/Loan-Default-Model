"""
train_model.py
----------------
End-to-end training pipeline for Loan Default Prediction

Steps:
1. Load data
2. Clean & preprocess
3. Encode categorical variables
4. Train Random Forest model
5. Save trained model and scaler
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("../data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

# --------------------------------------------------
# 2. Basic Cleaning
# --------------------------------------------------

# Drop identifier column
if "loan_id" in df.columns:
    df.drop(columns=["loan_id"], inplace=True)

# Strip spaces from all string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()


# --------------------------------------------------
# 3. Target Encoding (SAFE)
# --------------------------------------------------

TARGET_COL = "loan_status"

df[TARGET_COL] = df[TARGET_COL].map({
    "Approved": 0,   # Non-defaulter
    "Rejected": 1    # Potential defaulter
})

# Validate target
if df[TARGET_COL].isna().sum() > 0:
    raise ValueError("Target column contains NaN values after mapping!")


# --------------------------------------------------
# 4. Encode Categorical Features
# --------------------------------------------------

categorical_cols = ["education", "self_employed"]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # stored if needed later


# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# --------------------------------------------------
# 6. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 7. Model Training
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 8. Save Artifacts
# --------------------------------------------------

joblib.dump(model, "loan_default_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete")
print("📦 Saved files:")
print(" - loan_default_model.pkl")
print(" - scaler.pkl")
