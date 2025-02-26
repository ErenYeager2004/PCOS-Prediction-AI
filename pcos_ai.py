import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

file_path = "pcos_final_dataset_fixed.json"  
df = pd.read_json(file_path)

df.drop(columns=["Sl. No", "Patient File No.", ""], inplace=True, errors="ignore")

df["Marraige Status (Yrs)"] = pd.to_numeric(df["Marraige Status (Yrs)"], errors="coerce")
df["II    beta-HCG(mIU/mL)"] = pd.to_numeric(df["II    beta-HCG(mIU/mL)"], errors="coerce")
df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors="coerce")

yn_columns = ["Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
              "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"]
df[yn_columns] = df[yn_columns].replace({"Y": 1, "N": 0, "Yes": 1, "No": 0}).astype(int)

df.fillna(df.median(), inplace=True)

top_features = ["Follicle No. (R)", "Follicle No. (L)", "Skin darkening (Y/N)", "hair growth(Y/N)",
                "Weight gain(Y/N)", "Cycle(R/I)", "Fast food (Y/N)", "Pimples(Y/N)", "AMH(ng/mL)", "Weight (Kg)"]

X = df[top_features]
y = df["PCOS (Y/N)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

log_reg_preds = log_reg.predict(X_test)
rf_preds = rf_model.predict(X_test)

print("\n🔹 Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_preds))
print("\n📊 Logistic Regression Report:\n", classification_report(y_test, log_reg_preds))
print("\n🔹 Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\n📊 Random Forest Report:\n", classification_report(y_test, rf_preds))

def predict_pcos(user_data):
    user_df = pd.DataFrame([user_data], columns=top_features)
    prediction_prob = log_reg.predict_proba(user_df)[0][1]  # Probability of having PCOS
    print(f"🔍 PCOS Probability: {prediction_prob:.2f}")  # Debugging output
    
    threshold_upper = 0.6  # Edge case threshold
    threshold_lower = 0.75  # High-risk threshold
    
    if prediction_prob >= threshold_lower:
        return "PCOS Detected"
    elif prediction_prob >= threshold_upper:
        return "Borderline - At Risk"
    else:
        return "No PCOS"

new_patient = {
    "Follicle No. (R)": int(input("Follicle No. (R): ")),
    "Follicle No. (L)": int(input("Follicle No. (L): ")),
    "Skin darkening (Y/N)": int(input("Skin darkening (0-No, 1-Yes): ")),
    "hair growth(Y/N)": int(input("Hair Growth (0-No, 1-Yes): ")),
    "Weight gain(Y/N)": int(input("Weight Gain (0-No, 1-Yes): ")),
    "Cycle(R/I)": int(input("Cycle Regularity (2-Regular, 4-Irregular): ")),
    "Fast food (Y/N)": int(input("Fast Food Consumption (0-No, 1-Yes): ")),
    "Pimples(Y/N)": int(input("Pimples (0-No, 1-Yes): ")),
    "AMH(ng/mL)": float(input("AMH Level (ng/mL): ")),
    "Weight (Kg)": float(input("Weight (Kg): "))
}

result = predict_pcos(new_patient)
print("\n🩺 Diagnosis Result:", result)
