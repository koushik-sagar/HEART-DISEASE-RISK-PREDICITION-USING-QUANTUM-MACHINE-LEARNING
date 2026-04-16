"""
=============================================================================
  QUANTUM CIRCUIT EXECUTION CODE
  Models: QSVC | QNN | VQC | Bagging-QSVC
  Dataset: Cleveland Heart Disease Dataset
  Compatible with: Qiskit 0.45.x, qiskit-machine-learning 0.7.x
=============================================================================

INSTALLATION (run once in your terminal):
  pip install qiskit==0.45.2
  pip install qiskit-aer==0.13.3
  pip install qiskit-machine-learning==0.7.2
  pip install scikit-learn matplotlib seaborn pandas numpy
=============================================================================
"""

import numpy as np
import pandas as pd
import argparse
import builtins
import multiprocessing as mp
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')   # change to 'TkAgg' or remove this line if running interactively
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ── Qiskit core ──────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import SPSA, L_BFGS_B
from qiskit.primitives import Sampler

# ── Qiskit Aer backend ───────────────────────────────────────────────────────
from qiskit_aer import AerSimulator

# ── Qiskit Machine Learning ──────────────────────────────────────────────────
from qiskit_machine_learning.algorithms import QSVC, VQC, NeuralNetworkClassifier
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.neural_networks import SamplerQNN

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "execution_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_NOTEBOOK = clear_output is not None and "ipykernel" in sys.modules

TARGET_COLUMN = "target"
COMMON_FEATURES = ["ca", "cp", "thal", "exang", "slope"]
FEATURE_MAP_REPS = 2
FEATURE_MAP_ENTANGLEMENT = "linear"
ANSATZ_REPS = 2
ANSATZ_ENTANGLEMENT = "linear"


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum heart-disease model runner")
    parser.add_argument(
        "--mode",
        choices=["circuits", "quick", "balanced", "full"],
        default="circuits",
        help="Use 'circuits' to export diagrams only, 'quick' for verification, 'balanced' for a middle ground, or 'full' for the heaviest run.",
    )
    return parser.parse_args()


ARGS = parse_args()
RUN_MODE = ARGS.mode
if RUN_MODE == "circuits":
    QNN_CV_SPLITS = 0
    QNN_OPTIMIZER_MAXITER = 0
    VQC_OPTIMIZER_MAXITER = 0
    BAGGING_N_ESTIMATORS = 0
elif RUN_MODE == "quick":
    QNN_CV_SPLITS = 3
    QNN_OPTIMIZER_MAXITER = 25
    VQC_OPTIMIZER_MAXITER = 30
    BAGGING_N_ESTIMATORS = 5
elif RUN_MODE == "balanced":
    QNN_CV_SPLITS = 5
    QNN_OPTIMIZER_MAXITER = 60
    VQC_OPTIMIZER_MAXITER = 60
    BAGGING_N_ESTIMATORS = 10
else:
    QNN_CV_SPLITS = 8
    QNN_OPTIMIZER_MAXITER = 150
    VQC_OPTIMIZER_MAXITER = 100
    BAGGING_N_ESTIMATORS = 10


def safe_print(*args, **kwargs):
    """Fall back to ASCII when the active terminal cannot render Unicode."""
    try:
        return builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        converted = [str(arg).encode("ascii", "replace").decode("ascii") for arg in args]
        return builtins.print(*converted, **kwargs)


print = safe_print


def sanitise_filename(text):
    """Create a filesystem-safe filename stem from a display title."""
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_")


def format_circuit_title(title):
    """Render document-friendly circuit headings."""
    return title.replace("_", " ").upper()


def resolve_dataset_path():
    """Find the Cleveland dataset regardless of the current working directory."""
    candidates = [
        BASE_DIR / "Cleveland_Dataset.csv",
        BASE_DIR / "Cleveland Dataset.csv",
        BASE_DIR / "Explainable_Heart_Disease_Prediction_Using_Ensemble-Quantum_ML-main" / "Cleveland Dataset.csv",
        BASE_DIR / "data" / "Cleveland_Dataset.csv",
        BASE_DIR / "data" / "Cleveland Dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cleveland dataset not found. Put 'Cleveland_Dataset.csv' in the project "
        "folder or keep the original dataset under "
        "'Explainable_Heart_Disease_Prediction_Using_Ensemble-Quantum_ML-main'."
    )


def build_zz_feature_map(
    feature_dimension,
    reps=FEATURE_MAP_REPS,
    entanglement=FEATURE_MAP_ENTANGLEMENT,
):
    """Prefer the newer zz_feature_map builder when available."""
    try:
        from qiskit.circuit.library import zz_feature_map

        return zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix="x",
            name="ZZFeatureMap",
        )
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap

        return ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix="x",
        )


def build_efficient_su2_ansatz(
    num_qubits,
    reps=ANSATZ_REPS,
    entanglement=ANSATZ_ENTANGLEMENT,
):
    """Prefer the newer efficient_su2 builder when available."""
    try:
        from qiskit.circuit.library import efficient_su2

        return efficient_su2(
            num_qubits=num_qubits,
            su2_gates=["ry", "rz"],
            entanglement=entanglement,
            reps=reps,
            parameter_prefix="theta",
            name="EfficientSU2",
        )
    except ImportError:
        from qiskit.circuit.library import EfficientSU2

        return EfficientSU2(
            num_qubits=num_qubits,
            su2_gates=["ry", "rz"],
            entanglement=entanglement,
            reps=reps,
            parameter_prefix="theta",
            flatten=True,
        )


def build_qnn_classifier_circuit(feature_map, ansatz, num_qubits):
    """Keep the QNN structure as feature_map -> ansatz -> measurement."""
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc.measure_all()
    return qc


def get_common_feature_frame(df):
    """Use one shared raw feature set across all models."""
    return df[COMMON_FEATURES].copy()


def build_text_circuit_figure(drawing_text, title):
    """Fallback renderer when the Matplotlib circuit drawer is unavailable."""
    lines = [line.rstrip() for line in drawing_text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        lines = [drawing_text]
    drawing_text = "\n".join(lines)

    max_chars = max(len(line) for line in lines)
    fig_width = min(max(10, max_chars * 0.11), 44)
    fig_height = min(max(1.8, len(lines) * 0.28 + 0.8), 18)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=0.86)
    fig.text(
        0.01,
        0.97,
        format_circuit_title(title),
        ha='left',
        va='top',
        fontsize=14,
        fontweight='bold',
        fontfamily='Times New Roman',
    )
    ax.text(
        0.0,
        0.98,
        drawing_text,
        family='DejaVu Sans Mono',
        fontsize=8,
        va='top',
        ha='left',
        transform=ax.transAxes,
    )
    return fig


def build_circuit_figure(circuit, title, drawing_text=None, fold=-1):
    """Create a tightly cropped circuit figure for PDF/PNG export."""
    try:
        fig = circuit.decompose().draw(
            output='mpl',
            fold=fold,
            idle_wires=False,
        )
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.04, top=0.84)
        fig.suptitle(
            format_circuit_title(title),
            x=0.01,
            y=0.97,
            fontsize=14,
            fontweight='bold',
            fontfamily='Times New Roman',
            ha='left',
        )
        return fig
    except Exception:
        if drawing_text is None:
            drawing_text = str(circuit.decompose().draw(output='text', fold=fold))
        return build_text_circuit_figure(drawing_text, title)


def build_qnn_components(num_inputs):
    """Build the QNN feature map, ansatz, and classifier circuit."""
    feature_map_qnn = build_zz_feature_map(
        feature_dimension=num_inputs,
        reps=FEATURE_MAP_REPS,
        entanglement=FEATURE_MAP_ENTANGLEMENT,
    )
    ansatz_qnn = build_efficient_su2_ansatz(
        num_qubits=num_inputs,
        reps=ANSATZ_REPS,
        entanglement=ANSATZ_ENTANGLEMENT,
    )
    qc_qnn = build_qnn_classifier_circuit(feature_map_qnn, ansatz_qnn, num_inputs)
    return feature_map_qnn, ansatz_qnn, qc_qnn


def build_vqc_components(feature_dim):
    """Build the VQC feature map and ansatz."""
    feature_map_vqc = build_zz_feature_map(
        feature_dimension=feature_dim,
        reps=FEATURE_MAP_REPS,
        entanglement=FEATURE_MAP_ENTANGLEMENT,
    )
    ansatz_vqc = build_efficient_su2_ansatz(
        num_qubits=feature_dim,
        reps=ANSATZ_REPS,
        entanglement=ANSATZ_ENTANGLEMENT,
    )
    return feature_map_vqc, ansatz_vqc


def save_circuit_text(circuit, title, fold=-1):
    """Save unwrapped circuit text plus document-safe PNG/PDF renders."""
    decomposed = circuit.decompose()
    drawing_text = str(decomposed.draw(output='text', fold=fold))
    text_output_path = OUTPUT_DIR / f"circuit_{sanitise_filename(title)}.txt"
    text_output_path.write_text(drawing_text, encoding="utf-8")
    try:
        print(drawing_text)
    except UnicodeEncodeError:
        ascii_fallback = drawing_text.encode("ascii", "replace").decode("ascii")
        print(ascii_fallback)
    print(f"  [saved {text_output_path}]")

    fig = build_circuit_figure(circuit, title, drawing_text=drawing_text, fold=fold)

    png_output_path = OUTPUT_DIR / f"circuit_{sanitise_filename(title)}.png"
    pdf_output_path = OUTPUT_DIR / f"circuit_{sanitise_filename(title)}.pdf"
    fig.savefig(png_output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(pdf_output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  [saved {png_output_path}]")
    print(f"  [saved {pdf_output_path}]")
    return drawing_text


def save_combined_circuit_pdf(circuit_pages, filename="all_quantum_circuits.pdf"):
    """Save all circuit diagrams into one titled multi-page PDF."""
    pdf_output_path = OUTPUT_DIR / filename
    with PdfPages(pdf_output_path) as pdf:
        for title, circuit, drawing_text in circuit_pages:
            fig = build_circuit_figure(circuit, title, drawing_text=drawing_text)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
    print(f"  [saved {pdf_output_path}]")
    return pdf_output_path


def save_objective_plot(values, title, filename):
    """Persist a loss curve without opening a GUI backend."""
    if not values:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.plot(range(len(values)), values)
    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  [saved {output_path}]")


def shutdown_runtime():
    """Best-effort cleanup so the script returns to the shell promptly."""
    plt.close('all')
    for child in mp.active_children():
        child.terminate()
        child.join(timeout=2)
    sys.stdout.flush()
    sys.stderr.flush()


# =============================================================================
#  HELPER UTILITIES
# =============================================================================

def cm_analysis(y_true, y_pred, labels, title="", figsize=(5, 4)):
    """Pretty-print a labelled confusion matrix heat-map."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c, p, s = cm[i, j], cm_perc[i, j], cm_sum[i][0]
            annot[i, j] = ('0.0%%\n0/%d' % s) if c == 0 else ('%.1f%%\n%d/%d' % (p, c, s))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_cm, annot=annot, fmt='', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"confusion_matrix_{sanitise_filename(title)}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  [saved {output_path}]")


def plot_roc(y_true, y_score, title="ROC Curve"):
    """Plot and save a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"roc_{sanitise_filename(title)}.png"
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  [saved {output_path}]")


# =============================================================================
#  LOAD & PREVIEW DATA
# =============================================================================

print("=" * 70)
print("  LOADING CLEVELAND HEART DISEASE DATASET")
print("=" * 70)
print(f"Run mode: {RUN_MODE}")

dataset_path = resolve_dataset_path()
print(f"Dataset path: {dataset_path}")
data = pd.read_csv(dataset_path)
print(data.head())
print("\nShape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())

X_common_df = get_common_feature_frame(data)
y_all = data[TARGET_COLUMN].values
shared_feature_dimension = X_common_df.shape[1]

print("\nShared raw features used by all models:", COMMON_FEATURES)
print(f"Shared feature dimension: {shared_feature_dimension}")
print("QNN and VQC now use the same five features directly (no PCA reduction).")

shared_feature_map_qsvc_bg = build_zz_feature_map(
    feature_dimension=shared_feature_dimension,
    reps=FEATURE_MAP_REPS,
    entanglement=FEATURE_MAP_ENTANGLEMENT,
)

if RUN_MODE == "circuits":
    print("\n" + "=" * 70)
    print("  CIRCUITS ONLY MODE")
    print("=" * 70)
    circuit_pages = []

    print(
        f"\n[QSVC] Shared ZZ feature map "
        f"({shared_feature_dimension} qubits, reps={FEATURE_MAP_REPS}, "
        f"entanglement='{FEATURE_MAP_ENTANGLEMENT}'):"
    )
    qsvc_feature_map_text = save_circuit_text(shared_feature_map_qsvc_bg, "QSVC_feature_map")
    circuit_pages.append(
        ("QSVC Feature Map", shared_feature_map_qsvc_bg, qsvc_feature_map_text)
    )

    qnn_feature_map_only, qnn_ansatz_only, qnn_circuit_only = build_qnn_components(
        shared_feature_dimension
    )
    print(
        f"\n[QNN] Quantum circuit (feature_map + ansatz + measure, "
        f"{shared_feature_dimension} qubits):"
    )
    qnn_circuit_text = save_circuit_text(qnn_circuit_only, "QNN_circuit")
    circuit_pages.append(
        ("QNN Circuit", qnn_circuit_only, qnn_circuit_text)
    )

    vqc_feature_map_only, vqc_ansatz_only = build_vqc_components(shared_feature_dimension)
    print("\n[VQC] Feature Map Circuit:")
    vqc_feature_map_text = save_circuit_text(vqc_feature_map_only, "VQC_feature_map")
    circuit_pages.append(
        ("VQC Feature Map", vqc_feature_map_only, vqc_feature_map_text)
    )

    print("\n[VQC] Ansatz Circuit:")
    vqc_ansatz_text = save_circuit_text(vqc_ansatz_only, "VQC_ansatz")
    circuit_pages.append(
        ("VQC Ansatz", vqc_ansatz_only, vqc_ansatz_text)
    )

    print(
        f"\n[Bagging-QSVC] Shared ZZ feature map "
        f"({shared_feature_dimension} qubits, reps={FEATURE_MAP_REPS}, "
        f"entanglement='{FEATURE_MAP_ENTANGLEMENT}'):"
    )
    bagging_feature_map_text = save_circuit_text(
        shared_feature_map_qsvc_bg,
        "Bagging_QSVC_feature_map",
    )
    circuit_pages.append(
        (
            "Bagging-QSVC Feature Map",
            shared_feature_map_qsvc_bg,
            bagging_feature_map_text,
        )
    )
    save_combined_circuit_pdf(circuit_pages)

    print(f"\nAll circuit files were saved to: {OUTPUT_DIR}")
    shutdown_runtime()
    raise SystemExit(0)


# =============================================================================
#  MODEL 1 – QSVC (Quantum Support Vector Classifier)
# =============================================================================

print("\n" + "=" * 70)
print("  MODEL 1 – QSVC")
print("=" * 70)

# ── 1a. Data preparation ─────────────────────────────────────────────────────
X_q = X_common_df.values
y_q = y_all

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
    X_q, y_q, test_size=0.20, random_state=42
)

# Scale to [0, 1] — required for quantum encoding
samples = np.vstack([X_train_q, X_test_q])
mm_scaler = MinMaxScaler((0, 1)).fit(samples)
X_train_q = mm_scaler.transform(X_train_q)
X_test_q  = mm_scaler.transform(X_test_q)

C_val      = 1000     # SVM regularisation parameter

# ── 1b. Build quantum kernel ─────────────────────────────────────────────────
#  FIX: use AerSimulator instead of deprecated QuantumInstance/BasicAer
algorithm_globals.random_seed = 12345

backend_qsvc = AerSimulator(method="statevector")

qkernel = QuantumKernel(
    feature_map=shared_feature_map_qsvc_bg,
    quantum_instance=backend_qsvc
)

# ── 1c. Visualise the feature-map circuit ────────────────────────────────────
print(
    f"\n[QSVC] Shared ZZ feature map "
    f"({shared_feature_dimension} qubits, reps={FEATURE_MAP_REPS}, "
    f"entanglement='{FEATURE_MAP_ENTANGLEMENT}'):"
)
save_circuit_text(shared_feature_map_qsvc_bg, "QSVC_feature_map")

# ── 1d. Train & evaluate ─────────────────────────────────────────────────────
qsvc = QSVC(quantum_kernel=qkernel, C=C_val)
print("\n[QSVC] Training …")
qsvc.fit(X_train_q, y_train_q)

y_pred_qsvc = qsvc.predict(X_test_q)
acc_qsvc = accuracy_score(y_test_q, y_pred_qsvc)

print(f"\n[QSVC] Test Accuracy : {acc_qsvc * 100:.2f}%")
print("[QSVC] Classification Report:\n",
      classification_report(y_test_q, y_pred_qsvc))

cm_analysis(y_test_q, y_pred_qsvc, labels=[0, 1], title="QSVC")
plot_roc(y_test_q, y_pred_qsvc, title="QSVC ROC Curve")


# =============================================================================
#  MODEL 2 – QNN (Quantum Neural Network via SamplerQNN)
# =============================================================================

print("\n" + "=" * 70)
print("  MODEL 2 – QNN (CircuitQNN → SamplerQNN)")
print("=" * 70)

algorithm_globals.random_seed = 42

# ── 2a. Data preparation ─────────────────────────────────────────────────────
X_qnn_raw = X_common_df.values
y_qnn     = y_all

std_scaler = StandardScaler().fit(X_qnn_raw)
X_std      = std_scaler.transform(X_qnn_raw)

mm_qnn    = MinMaxScaler((0, 1)).fit(X_std)
X_mm_qnn  = mm_qnn.transform(X_std)

num_inputs = X_mm_qnn.shape[1]

# ── 2b. Build the quantum circuit ─────────────────────────────────────────────
feature_map_qnn, ansatz_qnn, qc_qnn = build_qnn_components(num_inputs)

print(
    f"\n[QNN] Quantum circuit (feature_map + ansatz + measure, "
    f"{num_inputs} qubits):"
)
print(f"[QNN] Optimizer: L_BFGS_B(maxiter={QNN_OPTIMIZER_MAXITER})")
save_circuit_text(qc_qnn, "QNN_circuit")

# ── 2c. Parity interpret function ─────────────────────────────────────────────
def parity(x):
    """Map bit-string integer → {0, 1} by parity of set bits."""
    return int("{:b}".format(x).count("1")) % 2

# ── 2d. Build SamplerQNN ─────────────────────────────────────────────────────
#  FIX: CircuitQNN (deprecated) → SamplerQNN (modern API)
sampler_qnn = Sampler()

qnn = SamplerQNN(
    circuit=qc_qnn,
    input_params=feature_map_qnn.parameters,
    weight_params=ansatz_qnn.parameters,
    interpret=parity,
    output_shape=2,
    sampler=sampler_qnn,
)

# ── 2e. Callback for live loss plot ───────────────────────────────────────────
objective_func_vals = []

def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    if not INTERACTIVE_NOTEBOOK:
        return
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("QNN - Objective Function vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.plot(range(len(objective_func_vals)), objective_func_vals)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# ── 2f. Wrap in NeuralNetworkClassifier & cross-validate ─────────────────────
X_df_qnn = pd.DataFrame(X_mm_qnn, columns=[f"f{i}" for i in range(num_inputs)])
y_s      = pd.Series(y_qnn)

kf = KFold(n_splits=QNN_CV_SPLITS, shuffle=True, random_state=1)
qnn_scores = []

print(f"\n[QNN] Starting {QNN_CV_SPLITS}-fold cross-validation ...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_df_qnn), 1):
    x_tr, x_te = X_df_qnn.iloc[train_idx].values, X_df_qnn.iloc[test_idx].values
    y_tr, y_te = y_s.iloc[train_idx].values,       y_s.iloc[test_idx].values
    print(f"  Fold {fold}/{QNN_CV_SPLITS}: training ...")

    circuit_classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=L_BFGS_B(maxiter=QNN_OPTIMIZER_MAXITER),
        loss='absolute_error',
        callback=callback_graph,
    )
    circuit_classifier.fit(x_tr, y_tr)
    y_pred_qnn = circuit_classifier.predict(x_te)
    fold_acc   = accuracy_score(y_te, y_pred_qnn)
    qnn_scores.append(fold_acc)
    print(f"  Fold {fold}: accuracy = {fold_acc * 100:.2f}%")

print(f"\n[QNN] Mean CV Accuracy : {np.mean(qnn_scores) * 100:.2f}%")
print(f"[QNN] Classification Report (last fold):\n",
      classification_report(y_te, y_pred_qnn))

save_objective_plot(
    objective_func_vals,
    title="QNN - Objective Function vs Iteration",
    filename="qnn_objective_history.png",
)
cm_analysis(y_te, y_pred_qnn, labels=[0, 1], title="QNN")
plot_roc(y_te, y_pred_qnn, title="QNN ROC Curve")


# =============================================================================
#  MODEL 3 – VQC (Variational Quantum Classifier)
# =============================================================================

print("\n" + "=" * 70)
print("  MODEL 3 – VQC")
print("=" * 70)

# ── 3a. Data preparation ─────────────────────────────────────────────────────
X_vqc_raw = X_common_df.values
y_vqc     = y_all

X_tr_v, X_te_v, y_tr_v, y_te_v = train_test_split(
    X_vqc_raw, y_vqc, test_size=0.20, random_state=42, stratify=y_vqc
)

scaler_v   = StandardScaler()
X_tr_v     = scaler_v.fit_transform(X_tr_v)
X_te_v     = scaler_v.transform(X_te_v)

mm_v       = MinMaxScaler((0, 1)).fit(np.vstack([X_tr_v, X_te_v]))
X_tr_v     = mm_v.transform(X_tr_v).astype(float)
X_te_v     = mm_v.transform(X_te_v).astype(float)
y_tr_v     = np.asarray(y_tr_v, dtype=int).ravel()
y_te_v     = np.asarray(y_te_v, dtype=int).ravel()

feature_dim_v = X_tr_v.shape[1]

print(f"\n[VQC] Train: {X_tr_v.shape}, Test: {X_te_v.shape}")
print(f"[VQC] Unique labels: {np.unique(y_tr_v)}")
print(
    f"[VQC] ZZ feature map qubits/features: {feature_dim_v}, "
    f"ansatz: EfficientSU2(reps={ANSATZ_REPS})"
)
print(f"[VQC] SPSA maxiter: {VQC_OPTIMIZER_MAXITER}")

# ── 3b. Build VQC circuit components ─────────────────────────────────────────
algorithm_globals.random_seed = 42

feature_map_vqc, ansatz_vqc = build_vqc_components(feature_dim_v)
optimizer_vqc   = SPSA(maxiter=VQC_OPTIMIZER_MAXITER)
sampler_vqc     = Sampler()

print("\n[VQC] Feature Map Circuit:")
save_circuit_text(feature_map_vqc, "VQC_feature_map")

print("\n[VQC] Ansatz Circuit:")
save_circuit_text(ansatz_vqc, "VQC_ansatz")

# ── 3c. Train & evaluate ─────────────────────────────────────────────────────
vqc = VQC(
    sampler=sampler_vqc,
    feature_map=feature_map_vqc,
    ansatz=ansatz_vqc,
    optimizer=optimizer_vqc,
)

print("\n[VQC] Training …")
vqc.fit(X_tr_v, y_tr_v)

y_pred_vqc = vqc.predict(X_te_v)
acc_vqc    = accuracy_score(y_te_v, y_pred_vqc)

print(f"\n[VQC] Test Accuracy : {acc_vqc * 100:.2f}%")
print("[VQC] Classification Report:\n",
      classification_report(y_te_v, y_pred_vqc))

cm_analysis(y_te_v, y_pred_vqc, labels=[0, 1], title="VQC")
plot_roc(y_te_v, y_pred_vqc, title="VQC ROC Curve")


# =============================================================================
#  MODEL 4 – BAGGING-QSVC (Proposed Model)
# =============================================================================

print("\n" + "=" * 70)
print("  MODEL 4 – BAGGING-QSVC (Proposed Model)")
print("=" * 70)

# ── 4a. Data preparation ─────────────────────────────────────────────────────
X_bg = X_common_df.values
y_bg = y_all

X_tr_bg, X_te_bg, y_tr_bg, y_te_bg = train_test_split(
    X_bg, y_bg, test_size=0.20, random_state=42
)

mm_bg    = MinMaxScaler((0, 1)).fit(np.vstack([X_tr_bg, X_te_bg]))
X_tr_bg  = mm_bg.transform(X_tr_bg)
X_te_bg  = mm_bg.transform(X_te_bg)

C_bg          = 1000

# ── 4b. Build QSVC base estimator ─────────────────────────────────────────────
#  FIX: QuantumInstance + BasicAer removed → use AerSimulator directly
algorithm_globals.random_seed = 12345

backend_bg = AerSimulator(method="statevector")

print(
    f"\n[Bagging-QSVC] Shared ZZ feature map "
    f"({shared_feature_dimension} qubits, reps={FEATURE_MAP_REPS}, "
    f"entanglement='{FEATURE_MAP_ENTANGLEMENT}'):"
)
save_circuit_text(shared_feature_map_qsvc_bg, "Bagging_QSVC_feature_map")

qkernel_bg = QuantumKernel(
    feature_map=shared_feature_map_qsvc_bg,
    quantum_instance=backend_bg
)

qsvc_bg = QSVC(quantum_kernel=qkernel_bg, C=C_bg)

# ── 4c. First train the base QSVC alone ──────────────────────────────────────
print("\n[Bagging-QSVC] Training base QSVC …")
qsvc_bg.fit(X_tr_bg, y_tr_bg)
base_score = qsvc_bg.score(X_te_bg, y_te_bg)
print(f"  Base QSVC score : {base_score * 100:.2f}%")

# ── 4d. Wrap in a pipeline + BaggingClassifier ────────────────────────────────
#  FIX: 'base_estimator' renamed to 'estimator' in sklearn ≥ 1.2
#       Use try/except for backwards compatibility

pipeline_bg = make_pipeline(MinMaxScaler(), qsvc_bg)

try:
    # sklearn ≥ 1.2
    bg_clf = BaggingClassifier(
        estimator=pipeline_bg,
        n_estimators=BAGGING_N_ESTIMATORS,
        random_state=1,
        n_jobs=1,             # quantum simulators are not thread-safe; keep n_jobs=1
    )
except TypeError:
    # sklearn < 1.2
    bg_clf = BaggingClassifier(
        base_estimator=pipeline_bg,
        n_estimators=BAGGING_N_ESTIMATORS,
        random_state=1,
        n_jobs=1,
    )

print(f"\n[Bagging-QSVC] Training ensemble (n_estimators={BAGGING_N_ESTIMATORS}) ...")
bg_clf.fit(X_tr_bg, y_tr_bg)

y_pred_bg  = bg_clf.predict(X_te_bg)
acc_bg     = accuracy_score(y_te_bg, y_pred_bg)
train_acc  = bg_clf.score(X_tr_bg, y_tr_bg)

print(f"\n[Bagging-QSVC] Test  Accuracy : {acc_bg   * 100:.2f}%")
print(f"[Bagging-QSVC] Train Accuracy : {train_acc * 100:.2f}%")
print("[Bagging-QSVC] Classification Report:\n",
      classification_report(y_te_bg, y_pred_bg))

cm_analysis(y_te_bg, y_pred_bg, labels=[0, 1], title="Bagging-QSVC")
plot_roc(y_te_bg, y_pred_bg, title="Bagging-QSVC ROC Curve")


# =============================================================================
#  SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"  {'Model':<20} {'Test Accuracy':>14}")
print(f"  {'-'*20}  {'-'*14}")
print(f"  {'QSVC':<20} {acc_qsvc*100:>13.2f}%")
print(f"  {'QNN (mean CV)':<20} {np.mean(qnn_scores)*100:>13.2f}%")
print(f"  {'VQC':<20} {acc_vqc*100:>13.2f}%")
print(f"  {'Bagging-QSVC':<20} {acc_bg*100:>13.2f}%")
print("=" * 70)
print(f"\nAll circuit text files and plots were saved to: {OUTPUT_DIR}")
shutdown_runtime()
