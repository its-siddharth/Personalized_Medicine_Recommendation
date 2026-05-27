import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TF_AVAILABLE = True
except Exception as e:
    print(f"Warning: TensorFlow could not be loaded ({type(e).__name__}: {e}). Deep learning steps will be skipped.")

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    pass

import joblib

# 2. Dataset Handling
print("Generating Dataset...")
np.random.seed(42)
num_samples = 2000

age = np.random.randint(18, 85, num_samples)
systolic_bp = np.random.randint(90, 180, num_samples)
diastolic_bp = systolic_bp - np.random.randint(30, 60, num_samples)
glucose_level = np.random.randint(70, 250, num_samples)
heart_rate = np.random.randint(60, 120, num_samples)

diagnoses = []
symptoms_list = []
medicines = []

disease_profiles = {
    "Hypertension": {"sym": ["headache", "shortness of breath", "nosebleeds", "dizziness"], "meds": ["Lisinopril", "Amlodipine", "Losartan"]},
    "Diabetes": {"sym": ["frequent urination", "increased thirst", "fatigue", "blurred vision"], "meds": ["Metformin", "Insulin", "Glipizide"]},
    "Asthma": {"sym": ["wheezing", "shortness of breath", "chest tightness", "coughing"], "meds": ["Albuterol", "Fluticasone", "Montelukast"]},
    "Migraine": {"sym": ["severe headache", "nausea", "sensitivity to light", "visual aura"], "meds": ["Sumatriptan", "Ibuprofen", "Propranolol"]},
    "Healthy": {"sym": ["none"], "meds": ["None", "Multivitamins"]}
}

for i in range(num_samples):
    sys = systolic_bp[i]
    gluc = glucose_level[i]
    if sys > 140:
        diag = "Hypertension"
    elif gluc > 126:
        diag = "Diabetes"
    else:
        diag = np.random.choice(["Asthma", "Migraine", "Healthy"], p=[0.2, 0.2, 0.6])
        
    diagnoses.append(diag)
    symptoms = np.random.choice(disease_profiles[diag]["sym"], size=np.random.randint(1, min(3, len(disease_profiles[diag]["sym"]) + 1)), replace=False)
    symptoms_list.append(", ".join(symptoms))
    medicines.append(np.random.choice(disease_profiles[diag]["meds"]))

df = pd.DataFrame({
    'patient_id': range(1, num_samples + 1),
    'age': age,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'glucose_level': glucose_level,
    'heart_rate': heart_rate,
    'symptoms': symptoms_list,
    'diagnosis': diagnoses,
    'medicine': medicines
})

print(f"Dataset Shape: {df.shape}")

# 3. Preprocessing
print("\\nPreprocessing data...")
tfidf = TfidfVectorizer(max_features=10)
symptoms_tfidf = tfidf.fit_transform(df['symptoms']).toarray()
symptoms_df = pd.DataFrame(symptoms_tfidf, columns=[f"sym_{i}" for i in range(symptoms_tfidf.shape[1])])

features = df[['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'heart_rate']].copy()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

X = pd.concat([features_scaled_df, symptoms_df], axis=1)
le_diag = LabelEncoder()
y = le_diag.fit_transform(df['diagnosis'])
le_med = LabelEncoder()
df['medicine_encoded'] = le_med.fit_transform(df['medicine'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Disease Prediction
print("\\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, y_pred))

# 5. Recommendation System
medicine_descriptions = {
    "Lisinopril": "ACE inhibitor used to treat high blood pressure (hypertension).",
    "Amlodipine": "Calcium channel blocker for hypertension and angina.",
    "Losartan": "Angiotensin II receptor blocker for hypertension.",
    "Metformin": "Improves blood sugar control in people with type 2 diabetes.",
    "Insulin": "Hormone used to control blood sugar in diabetes.",
    "Glipizide": "Stimulates the pancreas to release insulin for diabetes.",
    "Albuterol": "Bronchodilator that relaxes muscles in the airways for asthma.",
    "Fluticasone": "Steroid that prevents the release of substances that cause inflammation in asthma.",
    "Montelukast": "Leukotriene inhibitor used to prevent asthma attacks.",
    "Sumatriptan": "Used to treat acute migraine headaches.",
    "Ibuprofen": "Nonsteroidal anti-inflammatory drug (NSAID) for pain like migraine.",
    "Propranolol": "Beta blocker used to prevent migraines and treat hypertension.",
    "Multivitamins": "Dietary supplement for general health.",
    "None": "No medication required."
}

med_df = pd.DataFrame(list(medicine_descriptions.items()), columns=['medicine', 'description'])

def recommend_content_based(disease):
    subset = df[df['diagnosis'] == disease]
    if len(subset) == 0: return ["None"]
    return subset['medicine'].value_counts().head(3).index.tolist()

ratings = []
for i in range(1000):
    user = np.random.randint(1, 500)
    med_idx = np.random.randint(0, len(med_df))
    med = med_df['medicine'].iloc[med_idx]
    user_disease = df.iloc[user]['diagnosis'] if user < len(df) else "Unknown"
    
    if med in disease_profiles.get(user_disease, {}).get("meds", []):
        rating = np.random.randint(4, 6)
    else:
        rating = np.random.randint(1, 4)
    ratings.append((user, med, rating))

ratings_df = pd.DataFrame(ratings, columns=['user_id', 'medicine', 'rating'])

# Train SVD using scikit-learn
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='medicine', values='rating').fillna(0)
user_ids = user_item_matrix.index.tolist()
med_cols = user_item_matrix.columns.tolist()

svd = TruncatedSVD(n_components=10, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_

predicted_ratings = np.dot(user_factors, item_factors)
pred_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=med_cols)

def recommend_collaborative(user_id, n=3):
    if user_id not in pred_df.index:
        return []
    
    user_predictions = pred_df.loc[user_id]
    user_meds = ratings_df[ratings_df['user_id'] == user_id]['medicine'].tolist()
    
    unseen_predictions = user_predictions.drop(labels=[m for m in user_meds if m in user_predictions.index], errors='ignore')
    top_meds = unseen_predictions.sort_values(ascending=False).head(n).index.tolist()
    return top_meds

def recommend_hybrid(user_id, disease):
    cb_recs = recommend_content_based(disease)
    cf_recs = recommend_collaborative(user_id, n=5)
    final_recs = list(dict.fromkeys(cb_recs + cf_recs)) 
    if 'None' in final_recs and len(final_recs) > 1: final_recs.remove('None')
    return final_recs[:3]

# 6. Deep Learning
print("\nTraining Deep Learning Model...")
if TF_AVAILABLE:
    tf.random.set_seed(42)
    dl_model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(le_diag.classes_), activation='softmax')
    ])
    dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dl_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = dl_model.evaluate(X_test, y_test, verbose=0)
    print(f"Deep Learning Test Accuracy: {acc:.4f}")
else:
    print("Skipping Deep Learning training because TensorFlow is not available.")

# 7. Test API Simulation
print("\nTesting API Simulation:")
def predict_and_recommend(age, sys_bp, dia_bp, glucose, heart_rate, symptoms_text):
    sym_vec = tfidf.transform([symptoms_text]).toarray()
    sym_df = pd.DataFrame(sym_vec, columns=[f"sym_{i}" for i in range(sym_vec.shape[1])])
    features_input = pd.DataFrame({'age': [age], 'systolic_bp': [sys_bp], 'diastolic_bp': [dia_bp], 'glucose_level': [glucose], 'heart_rate': [heart_rate]})
    feat_scaled = scaler.transform(features_input)
    feat_df = pd.DataFrame(feat_scaled, columns=features_input.columns)
    input_df = pd.concat([feat_df, sym_df], axis=1)
    
    pred_idx = rf_model.predict(input_df)[0]
    predicted_disease = le_diag.inverse_transform([pred_idx])[0]
    recs = recommend_hybrid(999, predicted_disease)
    return predicted_disease, recs

disease, meds = predict_and_recommend(45, 150, 95, 110, 85, "headache, dizziness")
print(f"Test Input: Age=45, BP=150/95, Glucose=110, Symptoms='headache, dizziness'")
print(f"Predicted Disease: {disease}")
print(f"Recommended Medicines: {meds}")

# 8. Save
print("\\nSaving models...")
joblib.dump(rf_model, 'disease_predictor_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(tfidf, 'tfidf_symptoms.pkl')
joblib.dump(le_diag, 'label_encoder_diag.pkl')
print("Models saved successfully to .pkl files!")
