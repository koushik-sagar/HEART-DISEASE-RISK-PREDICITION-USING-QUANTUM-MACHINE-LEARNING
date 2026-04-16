# Auto-generated from the notebook copy.
# Source notebook: QSVC, SVM, QNN, ANN, VQC, and Bagging-QSVC.ipynb

import matplotlib
matplotlib.use("Agg")

# %% cell 0
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

# =========================
# Qiskit (updated syntax)
# =========================

from qiskit import QuantumCircuit
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import EfficientSU2, zz_feature_map
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms import QSVC, NeuralNetworkClassifier, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import SPSA, L_BFGS_B, COBYLA
from qiskit_machine_learning.state_fidelities import ComputeUncompute

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    def clear_output(*args, **kwargs):
        return None

import warnings
warnings.filterwarnings('ignore')

# %% cell 1
import sys
print(sys.executable)


# %% cell 2
data = pd.read_csv('Cleveland Dataset.csv')
df = data.copy()
selected_features = ['ca', 'cp', 'thal', 'exang', 'slope']


def build_shared_feature_map(feature_dimension):
    return zz_feature_map(
        feature_dimension=feature_dimension,
        reps=2,
        entanglement='linear'
    )


def build_shared_ansatz(num_qubits):
    return EfficientSU2(num_qubits=num_qubits, reps=2)


print(df.head())

# %% cell 3
print(data.isnull().sum())

# %% cell 4
data.describe()

# %% [markdown] cell 5
# ## Heart Disease Distribution

# %% cell 6
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='target',data=data)
plt.title("Heart Disease Distribution")
plt.show()

# %% [markdown] cell 7
# ## Feature Correlation Heatmap

# %% cell 8
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# %% [markdown] cell 9
# ## Feature selection

# %% cell 10
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

X = data.drop(['target'] ,axis="columns")
y = data['target']

estimator = SVC(kernel="linear")
selector = RFE(estimator, n_features_to_select=6, step=1)
selector = selector.fit(X, y)

# %% cell 11
from operator import itemgetter
features = X.columns.to_list()
for x, y in (sorted(zip(selector.ranking_ , features), key=itemgetter(0))):
    print(x, y)

# %% cell 12
features =['age', 'chol', 'thalach','trestbps','cp','exang','oldpeak','fbs', 'thal']
data[features].hist(figsize=(18,10), bins=15)
plt.suptitle("Feature Distribution")
plt.show

# %% cell 13
corr_with_target=data.corr()['target'].sort_values(ascending=False)
print(corr_with_target)

# %% [markdown] cell 14
# # Classical classifiers 

# %% [markdown] cell 15
# ## SVM

# %% cell 16
# =========================
# Support vector machine (SVM)
# =========================

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = df[selected_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Underfitted SVM
ClassifierSVM = SVC(
    kernel='rbf',
    C=0.1,
    gamma=0.5,
    class_weight='balanced'
)
ClassifierSVM.fit(X_train, y_train)
y_pred_svm = ClassifierSVM.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Classical SVC Accuracy: {svm_accuracy*100:.2f}%")

# %% [markdown] cell 17
# ### SVM performance

# %% cell 18
# classification report of SVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

expected_y  = y_test
predicted_y = ClassifierSVM.predict(X_test)

svm_acc = accuracy_score(expected_y, predicted_y)


# print accuracy, classification report and confusion matrix for svm classifier
print(f"Classical SVC Accuracy: {svm_acc*100:.2f}%")
print("Classification report: \n", metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))

# %% [markdown] cell 19
# ### SVM--Confusion Matrix

# %% cell 20
# confusion matrix of SVM
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_sum = np.sum(cm, axis=1, keepdims=True) 
    cm_perc = cm / cm_sum.astype(float) * 100 

    annot = np.empty_like(cm).astype(str) 
    nrows, ncols = cm.shape 

    for i in range(nrows): 
        for j in range(ncols): 
            c = cm[i, j] 
            p = cm_perc[i, j] 
            s = cm_sum[i][0]

            if c == 0:
                annot[i, j] = '0.0%%\n0/%d' % s
            else:
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

    cm = pd.DataFrame(cm, index=labels, columns=labels) 
    cm.index.name = 'Actual' 
    cm.columns.name = 'Predicted' 

    fig, ax = plt.subplots(figsize=figsize) 
    sns.heatmap(cm, annot=annot, fmt='', ax=ax) 

cm_analysis(y_test, predicted_y, labels=[0,1], ymap=None, figsize=(5,4)) 

# %% [markdown] cell 21
# ### SVM--ROC Curve

# %% cell 22
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get decision scores instead of class labels
svm_scores = ClassifierSVM.decision_function(X_test)

# ROC values
fper, tper, thresholds = roc_curve(expected_y, svm_scores)
roc_auc = auc(fper, tper)

# Plot ROC
plt.figure(figsize=(6,4))
plt.plot(fper, tper, label=f'SVM ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of SVM')
plt.legend()
plt.show()

# %% [markdown] cell 23
# ## ANN

# %% cell 24
# Also known as classical QNN
X = df[selected_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = MLPClassifier(max_iter=1000, random_state=5)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Classical ANN Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# %% [markdown] cell 25
# ### ANN Performance

# %% cell 26
# classification report of ANN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

expected_y  = y_test
predicted_y = model.predict(X_test) 

ann_acc = accuracy_score(expected_y, predicted_y)
print(f"Classical ANN Accuracy: {ann_acc*100:.2f}%")
 

# print classification report and confusion matrix for the classifier
print("Classification report: \n", metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))

# %% [markdown] cell 27
# ### ANN--Confusion Matrix

# %% cell 28
# confusion matrix of ANN
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_sum = np.sum(cm, axis=1, keepdims=True) 
    cm_perc = cm / cm_sum.astype(float) * 100 
    annot = np.empty_like(cm).astype(str) 
    nrows, ncols = cm.shape 
    for i in range(nrows): 
        for j in range(ncols): 
            c = cm[i, j] 
            p = cm_perc[i, j] 
            if i == j: 
                s = cm_sum[i][0]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s) 
            elif c == 0: 
                annot[i, j] = '' 
            else: 
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels) 
    cm.index.name = 'Actual' 
    cm.columns.name = 'Predicted' 
    fig, ax = plt.subplots(figsize=figsize) 
    sns.heatmap(cm, annot=annot, fmt='', ax=ax) 
    
cm_analysis(y_test, predicted_y, labels=[0,1], ymap=None, figsize=(5,4)) 

# %% [markdown] cell 29
# ### ANN--ROC Curve

# %% cell 30
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities for positive class
y_prob = model.predict_proba(X_test)[:, 1]

# ROC values
fper, tper, thresholds = roc_curve(expected_y, y_prob)
roc_auc = auc(fper, tper)

# Plot ROC
plt.figure(figsize=(6,4))
plt.plot(fper, tper, label='ANN ROC ')
plt.plot([0, 1], [0, 1], linestyle='--',color='orange', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of ANN')
plt.legend()
plt.show()

# %% [markdown] cell 31
# # Quantum classifiers 

# %% [markdown] cell 32
#    ## QSVC

# %% cell 33
X = df[selected_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

samples = np.append(X_train, X_test, axis=0)
minmax_scaler = MinMaxScaler((0, 1)).fit(samples)
X_train = minmax_scaler.transform(X_train)
X_test = minmax_scaler.transform(X_test)

# %% cell 34
# number of qubits is equal to the number of features
num_qubits = X_train.shape[1]
# regularization parameter
C = 1000

# %% cell 35
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

algorithm_globals.random_seed = 12345

feature_map = build_shared_feature_map(num_qubits)
sampler = StatevectorSampler(seed=algorithm_globals.random_seed)
fidelity = ComputeUncompute(sampler=sampler)
qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

qsvc = QSVC(quantum_kernel=qkernel, C=C)

# %% cell 36
# training
qsvc.fit(X_train,y_train)

# testing
qsvc_score = qsvc.score(X_test, y_test)
print(f"QSVC classification test score: {qsvc_score*100:.2f}%")

# %% [markdown] cell 37
# ### QSVC performance

# %% cell 38
# classification report of QSVC
expected_y  = y_test
predicted_y = qsvc.predict(X_test) 

# print classification report and confusion matrix for the classifier
print("Classification report: \n", metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))

# %% [markdown] cell 39
# ### QSVC--Confusion Matrix

# %% cell 40
# confusion matrix of QSVC
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_sum = np.sum(cm, axis=1, keepdims=True) 
    cm_perc = cm / cm_sum.astype(float) * 100 
    annot = np.empty_like(cm).astype(str) 
    nrows, ncols = cm.shape 
    for i in range(nrows): 
        for j in range(ncols): 
            c = cm[i, j] 
            p = cm_perc[i, j] 
            if i == j: 
                s = cm_sum[i][0]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s) 
            elif c == 0: 
                annot[i, j] = '' 
            else: 
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels) 
    cm.index.name = 'Actual' 
    cm.columns.name = 'Predicted' 
    fig, ax = plt.subplots(figsize=figsize) 
    sns.heatmap(cm, annot=annot, fmt='', ax=ax) 
    
cm_analysis(y_test, predicted_y, labels=[0,1], ymap=None, figsize=(5,4)) 

# %% [markdown] cell 41
# ### QSVC--ROC Curve

# %% cell 42
# ROC curve of QSVC
import seaborn as sns
sns.set_theme()



def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='blue', label='ROC')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    
fper, tper, thresholds = roc_curve(expected_y, predicted_y)
plot_roc_curve(fper, tper)

# %% [markdown] cell 43
# ## QNN

# %% cell 44
algorithm_globals.random_seed = 42
sampler = StatevectorSampler(seed=algorithm_globals.random_seed)

# %% cell 45
X = df[selected_features]
y = df['target']

std_scaler = StandardScaler().fit(X)
X = std_scaler.transform(X)

minmax_scaler = MinMaxScaler((0, 1)).fit(X)
X = minmax_scaler.transform(X)

# for cross validation
X_df = pd.DataFrame(X, columns=selected_features)

num_qubits = X_df.shape[1]

# %% cell 46
# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

# %% cell 47
# Classification with a SamplerQNN
# construct feature map
feature_map = build_shared_feature_map(num_qubits)

# construct ansatz
ansatz = build_shared_ansatz(num_qubits)

# construct quantum circuit
qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)
qc.measure_all()
qc.decompose()

# %% cell 48
# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2


output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping

# %% cell 49
# =========================
# Construct QNN
# =========================

from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# Backend sampler
sampler = StatevectorSampler(seed=42)

# Parity function
def parity(x):
    return f"{x:b}".count("1") % 2

output_shape = 2  # binary classification

# Construct QNN
circuit_qnn = SamplerQNN(
    circuit=qc,
    sampler=sampler,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=output_shape
)

print("QNN constructed successfully")

# %% cell 50
# construct classifier
circuit_classifier = NeuralNetworkClassifier(
            neural_network=circuit_qnn, optimizer= L_BFGS_B(), loss='absolute_error', callback=callback_graph
)

# %% cell 51
# cross validation
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

kf = KFold(n_splits = 8, shuffle = True, random_state= 1)
scores = []
for i in range(8):
    result = next(kf.split(X_df), None)
    x_train = X_df.iloc[result[0]]
    x_test = X_df.iloc[result[1]]
    y_train = y.iloc[result[0]]
    y_test = y.iloc[result[1]]
    circuit_classifier.fit(x_train,y_train)
    y_pred = circuit_classifier.predict(x_test)
    
plt.rcParams["figure.figsize"] = (6, 4)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))

# %% [markdown] cell 52
# ### QNN performance

# %% cell 53
# classification report of QNN
expected_y  = y_test
predicted_y = circuit_classifier.predict(x_test) 

# print accuracy, classification report and confusion matrix for the classifier
print(f"QNN Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification report: \n", metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))

# %% [markdown] cell 54
# ### QNN--Confusion Matrix

# %% cell 55
# confusion matrix of QNN
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_sum = np.sum(cm, axis=1, keepdims=True) 
    cm_perc = cm / cm_sum.astype(float) * 100 
    annot = np.empty_like(cm).astype(str) 
    nrows, ncols = cm.shape 
    for i in range(nrows): 
        for j in range(ncols): 
            c = cm[i, j] 
            p = cm_perc[i, j] 
            if i == j: 
                s = cm_sum[i][0]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s) 
            elif c == 0: 
                annot[i, j] = '' 
            else: 
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels) 
    cm.index.name = 'Actual' 
    cm.columns.name = 'Predicted' 
    fig, ax = plt.subplots(figsize=figsize) 
    sns.heatmap(cm, annot=annot, fmt='', ax=ax) 
    
cm_analysis(y_test, predicted_y, labels=[0,1], ymap=None, figsize=(5,4)) 

# %% [markdown] cell 56
# ### QNN--ROC Curve

# %% cell 57
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ROC curve of QNN
def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of QNN")
    plt.legend()
    plt.grid(True)
    plt.show()

# expected_y = true labels
# predicted_y = predicted probabilities or labels

fpr, tpr, thresholds = roc_curve(expected_y, predicted_y)
plot_roc_curve(fpr, tpr)

# %% [markdown] cell 58
# ## VQC

# %% cell 59
X = df[selected_features]
y = df['target']

# %% cell 60
# ================================
# DATA PREPROCESSING
# ================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Min-Max Scaling to [0,1] (important for quantum encoding)
samples = np.vstack((X_train, X_test))
minmax_scaler = MinMaxScaler((0, 1))
minmax_scaler.fit(samples)

X_train = minmax_scaler.transform(X_train)
X_test = minmax_scaler.transform(X_test)

print('Data preprocessing complete')
print('Train shape:', X_train.shape)
print('Test shape :', X_test.shape)

# %% cell 61
# ================================
# Ensure NumPy arrays
# ================================

import numpy as np

X_train = np.asarray(X_train, dtype=float)
X_test  = np.asarray(X_test, dtype=float)

y_train = np.asarray(y_train, dtype=int).ravel()
y_test  = np.asarray(y_test, dtype=int).ravel()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Unique labels:", np.unique(y_train))

# %% cell 62
# ================================
# VQC
# ================================

from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.metrics import accuracy_score

# -------------------------
# Parameters
# -------------------------
num_qubits = X_train.shape[1]
algorithm_globals.random_seed = 42

# -------------------------
# Sampler (modern)
# -------------------------
sampler = StatevectorSampler(seed=algorithm_globals.random_seed)

# -------------------------
# Feature map & Ansatz
# -------------------------
feature_map = build_shared_feature_map(num_qubits)
ansatz = build_shared_ansatz(num_qubits)

# -------------------------
# Optimizer
# -------------------------
optimizer = SPSA(maxiter=100)

# -------------------------
# VQC Model
# -------------------------
vqc = VQC(
    sampler=sampler,
    num_qubits=num_qubits,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

# -------------------------
# Train
# -------------------------
vqc.fit(X_train, y_train)

# -------------------------
# Test
# -------------------------
y_pred_vqc = vqc.predict(X_test)

acc = accuracy_score(y_test, y_pred_vqc)
print(f"VQC Test Accuracy: {acc*100:.2f}%")

# %% [markdown] cell 63
# ### VOC Performance

# %% cell 64
#result = vqc.run(quantum_instance)
#print("Quantum accuracy on test set: {0}%".format(round(result['testing_accuracy']*100, 2)))
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

y_pred_vqc = vqc.predict(X_test)
acc = accuracy_score(y_test, y_pred_vqc)

print(f"VQC Accuracy: {acc*100:.2f}%")


print("Classification Report:\n")
print(classification_report(y_test, y_pred_vqc))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_vqc))

# %% cell 65
print(f"y_test shape:      {y_test.shape}")
print(f"predicted_y shape: {len(predicted_y)}")
print(f"X_test shape:      {X_test.shape}")

# %% [markdown] cell 66
# ### VQC--Confusion Matrix

# %% cell 67
# confusion matrix of VQC
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_sum = np.sum(cm, axis=1, keepdims=True) 
    cm_perc = cm / cm_sum.astype(float) * 100 
    annot = np.empty_like(cm).astype(str) 
    nrows, ncols = cm.shape 
    for i in range(nrows): 
        for j in range(ncols): 
            c = cm[i, j] 
            p = cm_perc[i, j] 
            if i == j: 
                s = cm_sum[i][0]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s) 
            elif c == 0: 
                annot[i, j] = '' 
            else: 
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels) 
    cm.index.name = 'Actual' 
    cm.columns.name = 'Predicted' 
    fig, ax = plt.subplots(figsize=figsize) 
    sns.heatmap(cm, annot=annot, fmt='', ax=ax) 
    
cm_analysis(y_test, y_pred_vqc, labels=[0,1], ymap=None, figsize=(5,4)) 

# %% [markdown] cell 68
# ### VQC--ROC Curve

# %% cell 69
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme (modern & safe)
sns.set_theme(style="whitegrid")

def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of VQC")
    plt.legend()
    plt.show()

# Compute ROC
fpr, tpr, thresholds = roc_curve(expected_y, predicted_y)

# Plot
plot_roc_curve(fpr, tpr)

# %% [markdown] cell 70
# # Proposed model 

# %% [markdown] cell 71
# ## Bagging-QSVC

# %% cell 72
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier

# qiskit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

import warnings
warnings.filterwarnings('ignore')

# %% cell 73
data = pd.read_csv('Cleveland Dataset.csv')
df = data.copy()
selected_features = ['ca', 'cp', 'thal', 'exang', 'slope']


def build_shared_feature_map(feature_dimension):
    return zz_feature_map(
        feature_dimension=feature_dimension,
        reps=2,
        entanglement='linear'
    )

# %% cell 74
X = df[selected_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

samples = np.append(X_train, X_test, axis=0)
minmax_scaler = MinMaxScaler((0, 1)).fit(samples)
X_train = minmax_scaler.transform(X_train)
X_test = minmax_scaler.transform(X_test)

# %% cell 75
# number of qubits is equal to the number of features
num_qubits = X_train.shape[1]
# regularization parameter
C = 1000

# %% cell 76
algorithm_globals.random_seed = 12345

feature_map = build_shared_feature_map(num_qubits)
sampler = StatevectorSampler(seed=algorithm_globals.random_seed)
fidelity = ComputeUncompute(sampler=sampler)
qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

qsvc = QSVC(quantum_kernel=qkernel, C=C)

# %% cell 77
# training
qsvc.fit(X_train,y_train)

# testing
qsvc_score = qsvc.score(X_test, y_test)
print(f"QSVC classification test score: {qsvc_score*100:.2f}%")

# %% cell 78
# Pipeline Estimator
pipeline = make_pipeline(MinMaxScaler(), qsvc)

# Instantiate the bagging classifier
bgclassifier = BaggingClassifier(estimator=pipeline, n_estimators=100,
                                 random_state=1, n_jobs=1)

# Fit the bagging classifier
bgclassifier.fit(X_train, y_train)

# Model scores on test and training data
print(f"Model test Score: {bgclassifier.score(X_test, y_test)*100:.2f}%")
print(f"Model training Score: {bgclassifier.score(X_train, y_train)*100:.2f}%")

# %% [markdown] cell 79
# ### Bagging-QSVC performance

# %% cell 80
# classification report of Bagging-QSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

expected_y  = y_test
predicted_y = bgclassifier.predict(X_test)

# Accuracy
acc = accuracy_score(expected_y, predicted_y)
print(f"Bagging-QSVC Accuracy: {(acc*100):.2f}%")

# Classification report and confusion matrix
print("Classification report:\n", classification_report(expected_y, predicted_y))
print("Confusion matrix:\n", confusion_matrix(expected_y, predicted_y))

# %% [markdown] cell 81
# ### Bagging-QSVC-- Confusion Matrix

# %% cell 82
# confusion matrix of Bagging-QSVC
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,4)): 
    if ymap is not None: 
        y_pred = [ymap[yi] for yi in y_pred] 
        y_true = [ymap[yi] for yi in y_true] 
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_sum = np.sum(cm, axis=1, keepdims=True) 
    cm_perc = cm / cm_sum.astype(float) * 100 
    annot = np.empty_like(cm).astype(str) 
    nrows, ncols = cm.shape 
    for i in range(nrows): 
        for j in range(ncols): 
            c = cm[i, j] 
            p = cm_perc[i, j] 
            if i == j: 
                s = cm_sum[i][0]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s) 
            elif c == 0: 
                annot[i, j] = '' 
            else: 
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels) 
    cm.index.name = 'Actual' 
    cm.columns.name = 'Predicted' 
    fig, ax = plt.subplots(figsize=figsize) 
    sns.heatmap(cm, annot=annot, fmt='', ax=ax) 
    
cm_analysis(y_test, predicted_y, labels=[0,1], ymap=None, figsize=(5,4)) 

# %% [markdown] cell 83
# ### Bagging-QSVC--ROC Curve

# %% cell 84
# ROC curve of Bagging-QSVC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#  Modern seaborn style (safe)
sns.set_theme(style="whitegrid")

def plot_roc_curve(fper, tper):
    plt.figure(figsize=(6, 5))
    plt.plot(fper, tper, label="ROC Curve", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of Bagging-QSVC")
    plt.legend()
    plt.show()

# Compute ROC values
fper, tper, thresholds = roc_curve(expected_y, predicted_y)

# Plot ROC
plot_roc_curve(fper, tper)
