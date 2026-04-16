from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap          # ← upgraded from ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).resolve().parent
SOURCE_DIR   = PROJECT_DIR / "Explainable_Heart_Disease_Prediction_Using_Ensemble-Quantum_ML-main"
DATASET_PATH = SOURCE_DIR / "Cleveland Dataset.csv"
MODEL_DIR    = PROJECT_DIR / "models"
MODEL_PATH   = MODEL_DIR / "bagging_qsvc_quantum.joblib"

# ── Feature set ────────────────────────────────────────────────────────────
# Core clinical features — must match what the Streamlit app sends.
# Interaction features (cp_exang, oldpeak_slope, ca_thal) are added by the
# app's prepare_model_values() before calling the model, so list them here
# only if you want the model to receive them.
FEATURES = ["ca", "cp", "thal", "exang", "slope"]
TARGET   = "target"

# ── Model builder ──────────────────────────────────────────────────────────
def build_model(feature_dimension: int) -> BaggingClassifier:
    """
    Construct a Bagging ensemble of QSVC pipelines.

    ZZFeatureMap encodes pairwise feature interactions via ZZ-entanglement,
    giving a richer (higher-expressibility) Hilbert-space embedding than the
    single-qubit ZFeatureMap.  reps=2 keeps circuit depth manageable on a
    statevector simulator while still capturing second-order correlations.
    """
    algorithm_globals.random_seed = 12345

    backend = QuantumInstance(
        BasicAer.get_backend("statevector_simulator"),
        seed_simulator=algorithm_globals.random_seed,
        seed_transpiler=algorithm_globals.random_seed,
    )

    feature_map    = ZZFeatureMap(feature_dimension=feature_dimension, reps=2, entanglement="full")
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
    qsvc           = QSVC(quantum_kernel=quantum_kernel)
    pipeline       = make_pipeline(MinMaxScaler(), qsvc)

    return BaggingClassifier(
        estimator=pipeline,       # 'base_estimator' is deprecated; use 'estimator'
        n_estimators=3,
        random_state=1,
        n_jobs=1,
    )


# ── Training entry point ───────────────────────────────────────────────────
def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)

    # Sanity check
    missing = [f for f in FEATURES + [TARGET] if f not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X_df = data[FEATURES].copy()
    y    = data[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y,
    )

    print(f"Dataset rows  : {len(X_df)}")
    print(f"Feature set   : {FEATURES}")
    print(f"Feature dim   : {len(FEATURES)}")
    print("Kernel        : ZZFeatureMap (reps=2, full entanglement)")
    print("Model         : Bagging-QSVC (n_estimators=3)\n")

    # ── Hold-out evaluation ──
    print("Training evaluation model on train split…")
    eval_model = build_model(feature_dimension=len(FEATURES))
    eval_model.fit(X_train, y_train)
    test_preds    = eval_model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, test_preds))

    print(f"Test accuracy : {test_accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, test_preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_preds))

    # ── Final model trained on full dataset ──
    print("\nTraining final model on full dataset…")
    final_model = build_model(feature_dimension=len(FEATURES))
    final_model.fit(X_df, y)
    train_preds    = final_model.predict(X_df)
    train_accuracy = float(accuracy_score(y, train_preds))

    # ── Save artifact ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":        final_model,
        "model_name":   "Bagging-QSVC (ZZFeatureMap, reps=2, full entanglement)",
        "features":     FEATURES,
        "target":       TARGET,
        "dataset_path": str(DATASET_PATH),
        "train_accuracy": train_accuracy,
        "test_accuracy":  test_accuracy,
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"\nTrain accuracy (full data) : {train_accuracy * 100:.2f}%")
    print(f"Saved model artifact to    : {MODEL_PATH}")


if __name__ == "__main__":
    main()