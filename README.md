# Personalized_Medicine_Recommendation — Diagnostic & Therapeutics Platform

Personalized_Medicine_Recommendation is a highly polished, end-to-end clinical machine learning pipeline and interactive clinic dashboard. It leverages vitals-based diagnostic modeling and a hybrid recommendation engine to predict patient diagnoses and suggest personalized therapeutics.

The system is designed with a premium, responsive glassmorphic clinical interface featuring instant diagnostic predictions, custom interactive elements, and a seamless light/dark mode theme controller.

---

## 🌟 Key Features

*   **Clinical Vitals Diagnostic Inference**: Dynamic inputs for key patient parameters (**Age**, **Systolic/Diastolic Blood Pressure**, **Glucose Levels**, and **Heart Rate**) alongside present symptom selections.
*   **Intelligent Hybrid Recommendation Engine**: Suggests patient-specific therapeutics using an advanced hybrid algorithm:
    *   **Content-Based Filtering**: Matches medications based on typical diagnostic indicators and profile suitability.
    *   **Collaborative SVD Matrix Factorization**: Factorizes simulated clinical rating patterns using `scikit-learn`'s `TruncatedSVD` to offer highly relevant suggestions.
*   **Explainable ML (SHAP Visualizer)**: Animates contribution weights dynamically using custom SVG charts to represent SHAP feature contributions (e.g., highlighting Systolic BP contribution if Hypertension is diagnosed).
*   **Dual Aesthetic Modes**: A responsive, floating theme controller switches the dashboard between a futuristic dark holographic clinic space (`#0a0e17`) and a high-contrast standard light layout (`#f8fafc`).
*   **Clean, Complete Notebook**: Contains a fully documented Jupyter Notebook (`Personalized_Medicine_Recommendation.ipynb`) outlining preprocessing, training benchmarks, and explainability plots.

---

## 📂 Project Structure

```text
Personalized_Medicine_Recommendation/
├── Personalized_Medicine_Recommendation.ipynb  # Primary diagnostic & recommendation notebook
├── run_pipeline.py                             # Clean, robust production ML training pipeline
├── index.html                                  # Futuristic patient dashboard (Light/Dark glassmorphism)
├── medical_banner.png                          # Tech-art clinical banner asset
├── README.md                                   # Project documentation
├── disease_predictor_rf.pkl                    # Trained Random Forest classifier
├── label_encoder_diag.pkl                      # Diagnosis label encoder
├── scaler.pkl                                  # Vitals standard scaler
└── tfidf_symptoms.pkl                          # Symptom TF-IDF text vectorizer
```

---

## 🛠️ Setup & Installation

Follow these steps to set up the project locally on your system:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Personalized_Medicine_Recommendation.git
cd Personalized_Medicine_Recommendation
```

### 2. Activate Virtual Environment
Activate your local project virtual environment to access all pre-configured packages:
*   **Windows**:
    ```powershell
    .venv\Scripts\activate
    ```
*   **macOS/Linux**:
    ```bash
    source .venv/bin/activate
    ```

### 3. Install Dependencies
If creating a new environment, install the necessary clinical machine learning stack:
```bash
pip install pandas numpy scikit-learn tensorflow shap matplotlib seaborn
```

### 4. Execute ML Production Pipeline
Train the estimators, evaluate diagnostics accuracy, compile outputs, and generate model binary pickle assets:
```bash
python run_pipeline.py
```

### 5. Launch Clinical Web Dashboard
Serve the front-end dashboard locally using Python's built-in lightweight server:
```bash
python -m http.server 8000
```
Open your browser and navigate to: 👉 **[http://localhost:8000](http://localhost:8000)**

Or access the live clinical web application online at: 🌐 **[https://its-siddharth.github.io/Personalized_Medicine_Recommendation/](https://its-siddharth.github.io/Personalized_Medicine_Recommendation/)**

---

## 🧠 Core Technology Stack

*   **Machine Learning Models**: Random Forest Classifier (`scikit-learn`), Deep Learning Sequential Neural Network (`TensorFlow`/`Keras`).
*   **Therapeutics Recommender**: Truncated SVD Matrix Factorization, Content-Based TF-IDF Cosine Similarity.
*   **Model Explainability**: SHAP (SHapley Additive exPlanations) TreeExplainer.
*   **Dashboard Frontend**: Semantic HTML5, Vanilla JavaScript (ES6+), Modern CSS3 Variables & Grid layout.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
