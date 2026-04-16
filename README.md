# Explainable Heart Disease Prediction Using Ensemble-Quantum Machine Learning

## Aim
Build a reliable and explainable heart disease risk estimator that pairs quantum machine learning classifiers with a Bagging ensemble so clinicians can spot at-risk patients earlier and understand the prediction drivers..

## Abstract
This project trains a Bagging ensemble around a Quantum Support Vector Classifier (QSVC) using the Cleveland heart disease dataset, compares it against other quantum (QNN, VQC) and classical (SVC, ANN) classifiers, and explains every prediction through SHapley Additive exPlanations (SHAP).

## Real-time benefits
- **Early clinical alerts:** QSVC-powered scoring highlights high-risk patients ahead of acute symptoms so care teams can triage more effectively.
- **Explainability for stakeholders:** SHAP visualizations translate quantum outputs into familiar feature impacts (age, cholesterol, chest pain type, etc.), supporting regulatory review.
- **Robust deployment:** Bagging-QSVC stabilizes noisy quantum measurements, making the model easier to maintain across departments.

## How to use / run
1. **Prepare the environment:**
   ```powershell
   python -m venv .venv
   .venv\\Scripts\\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```powershell
   python train_model.py
   ```
   Outputs: trained model in models/, ROC/confusion PNGs under execution_outputs/, and SHAP explanations.
3. **Run the interactive demo:**
   ```powershell
   python app.py
   # or streamlit run app.py
   ```
   Enter the nine clinical inputs to view the predicted risk level, explanation, and gauge.
4. **Inspect quantum experiments (optional):**
   ```powershell
   python quantum_circuits_execution.py --mode full
   ```
   The script logs circuit diagrams, confusion matrices, and ROC curves under execution_outputs/.

## Data source
Place the Cleveland heart disease CSV inside data/ or at the project root before running training.
