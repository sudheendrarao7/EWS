import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv")

# Convert relevant columns to numeric (in case of import issues)
cols_to_numeric = [
    'CRISIL_Score', 
    'Net_Profit_Margin', 
    'ICR', 
    'CRILC_Flag',
    'CRISIL_Sectoral_Index', 
    'Fund_Based_Limits', 
    'TOL/TNW',
    'TOL/Adj_TNW', 
    'Order Book/Net Worth'
]

df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

# Drop NA
df = df.dropna()

# Generate Default_Flag based on same R logic
conditions = (
    (df['CRISIL_Score'] >= 7).astype(int) +
    (df['Net_Profit_Margin'] < 0.03).astype(int) +
    (df['ICR'] < 1.5).astype(int) +
    (df['CRILC_Flag'] == 1).astype(int) +
    (df['CRISIL_Sectoral_Index'] > 3).astype(int) +
    (df['Fund_Based_Limits'] > 90).astype(int) +
    (df['TOL/TNW'] > 4).astype(int) +
    (df['TOL/Adj_TNW'] > 5).astype(int) +
    ((df['Order Book/Net Worth'] > 5) | (df['Order Book/Net Worth'] < 1)).astype(int)
)


df['Default_Flag'] = np.where(conditions >= 3, 'Default', 'NonDefault')

# Encode labels
le = LabelEncoder()
df['Default_Flag'] = le.fit_transform(df['Default_Flag'])  # Default = 1, NonDefault = 0

# Features and labels
X = df[cols_to_numeric]
y = df['Default_Flag']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=123)

# Model training
rf = RandomForestClassifier(n_estimators=500, random_state=123)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png")  # saves image
plt.show()

# Cross-validated predictions
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
cv_preds = cross_val_predict(rf, X, y, cv=cv, method='predict')
cv_score = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:, 1]
print("\n5-Fold CV Classification Report:\n", classification_report(y, cv_preds))

# Feature Importance
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
print("\nFeature Importances:\n", feat_df)

# Save model
joblib.dump(rf, 'rf_model.pkl')
