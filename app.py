from __future__ import annotations

from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bagging_qsvc_quantum.joblib"

FEATURE_LABELS = {
    "age":           "Age",
    "sex":           "Sex",
    "cp":            "Chest Pain Type",
    "thalach":       "Maximum Heart Rate Reached",
    "exang":         "Exercise-Induced Angina",
    "oldpeak":       "ST Depression (Exercise vs Rest)",
    "slope":         "Peak Exercise ST Slope",
    "ca":            "Number of Major Vessels",
    "thal":          "Thallium Heart Scan",
    "cp_exang":      "Chest Pain × Exercise Angina",
    "oldpeak_slope": "ST Depression × ST Slope",
    "ca_thal":       "Major Vessels × Thallium",
}

FEATURE_ORDER = ["age", "sex", "cp", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

OPTION_SETS = {
    "sex": [
        {"ui": 0, "model": 0, "label": "Female"},
        {"ui": 1, "model": 1, "label": "Male"},
    ],
    "cp": [
        {"ui": 1, "model": 0, "label": "Typical Angina"},
        {"ui": 2, "model": 1, "label": "Atypical Angina"},
        {"ui": 3, "model": 2, "label": "Non-Anginal Pain"},
        {"ui": 4, "model": 3, "label": "Asymptomatic"},
    ],
    "exang": [
        {"ui": 0, "model": 0, "label": "No"},
        {"ui": 1, "model": 1, "label": "Yes"},
    ],
    "slope": [
        {"ui": 1, "model": 0, "label": "Upsloping"},
        {"ui": 2, "model": 1, "label": "Flat"},
        {"ui": 3, "model": 2, "label": "Downsloping"},
    ],
    "ca": [
        {"ui": 0, "model": 0, "label": "0 vessels"},
        {"ui": 1, "model": 1, "label": "1 vessel"},
        {"ui": 2, "model": 2, "label": "2 vessels"},
        {"ui": 3, "model": 3, "label": "3 vessels"},
    ],
    "thal": [
        {"ui": 1, "model": 2, "label": "Normal"},
        {"ui": 2, "model": 1, "label": "Fixed Defect"},
        {"ui": 3, "model": 3, "label": "Reversible Defect"},
    ],
}

NUMERIC_CONFIG = {
    "age":     {"min": 1,   "max": 120, "step": 1,   "format": "%d",   "default": 55},
    "thalach": {"min": 60,  "max": 220, "step": 1,   "format": "%d",   "default": 150},
    "oldpeak": {"min": 0.0, "max": 6.0, "step": 0.1, "format": "%.1f", "default": 0.0},
}

DISCLAIMER_TEXT = (
    "This application is intended for educational and research purposes only. "
    "It does not constitute medical advice, diagnosis, or treatment. "
    "Users should consult qualified healthcare professionals for any medical concerns."
)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioQ — Heart Disease Risk Prediction",
    layout="wide",
    page_icon="",
)


# ── Model Loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ── Feature Engineering ────────────────────────────────────────────────────
def prepare_model_values(values: dict[str, float | int]) -> dict[str, float]:
    """Add interaction features expected by the model."""
    prepared = {k: float(v) for k, v in values.items()}
    prepared["cp_exang"]      = prepared["cp"]      * prepared["exang"]
    prepared["oldpeak_slope"] = prepared["oldpeak"]  * prepared["slope"]
    prepared["ca_thal"]       = prepared["ca"]       * prepared["thal"]
    return prepared


def make_model_input(values: dict[str, float | int], features: list[str]) -> list[list[float]]:
    prepared = prepare_model_values(values)
    return [[float(prepared[f]) for f in features]]


# ── Validation ─────────────────────────────────────────────────────────────
def validate_inputs(values: dict[str, float | int]) -> tuple[list[str], list[str]]:
    errors:   list[str] = []
    warnings: list[str] = []

    age     = int(values["age"])
    thalach = int(values["thalach"])
    oldpeak = float(values["oldpeak"])

    max_hr = 220 - age
    if thalach > max_hr:
        errors.append(
            f"Heart rate {thalach} bpm exceeds age-adjusted maximum (~{max_hr} bpm for age {age}). "
            "Please verify the value."
        )
    if oldpeak > 5.0:
        warnings.append(f"ST depression of {oldpeak} is unusually high — please confirm.")

    return errors, warnings


# ── Scoring ────────────────────────────────────────────────────────────────
def quantum_risk_score(model, model_input: list[list[float]]) -> float:
    """Sigmoid-normalised average decision function across bagging estimators."""
    scores = [float(est.decision_function(model_input)[0]) for est in model.estimators_]
    avg    = sum(scores) / len(scores)
    return 1.0 / (1.0 + math.exp(-avg))


def clinical_rule_risk(values: dict[str, float | int]) -> tuple[float, list[str], list[str]]:
    score    = 0.0
    reasons:  list[str] = []
    warnings: list[str] = []

    age     = int(values["age"])
    thalach = int(values["thalach"])
    oldpeak = float(values["oldpeak"])
    cp      = int(values["cp"])
    exang   = int(values["exang"])
    slope   = int(values["slope"])
    ca      = int(values["ca"])
    thal    = int(values["thal"])

    age_max_hr = 220 - age
    if thalach > min(220, age_max_hr + 20):
        warnings.append(
            f"Heart rate {thalach} bpm is unusually high for age {age} "
            f"(age-adjusted estimate ≈ {age_max_hr} bpm)."
        )

    if age >= 65:
        score += 1.0;  reasons.append("Age ≥ 65: elevated baseline cardiovascular risk")
    elif age >= 55:
        score += 0.5;  reasons.append("Age 55–64: moderately elevated cardiovascular risk")

    if cp == 3:
        score += 1.5;  reasons.append("Asymptomatic chest pain — strong risk signal")
    elif cp == 2:
        score += 0.25; reasons.append("Non-anginal chest pain adds some uncertainty")

    if exang == 1:
        score += 1.5;  reasons.append("Exercise-induced angina — significant abnormal finding")

    if oldpeak >= 2.0:
        score += 1.5;  reasons.append(f"ST depression {oldpeak} ≥ 2.0: strong abnormal signal")
    elif oldpeak >= 1.0:
        score += 1.0;  reasons.append(f"ST depression {oldpeak} (1.0–2.0): elevated risk")
    elif oldpeak >= 0.5:
        score += 0.25; reasons.append(f"ST depression {oldpeak}: minor contribution")

    if slope == 2:
        score += 1.25; reasons.append("Downsloping ST — clinically concerning")
    elif slope == 1:
        score += 1.0;  reasons.append("Flat ST slope — clinically concerning")

    if ca >= 3:
        score += 2.0;  reasons.append("3 major vessels affected — severe blockage risk")
    elif ca >= 2:
        score += 1.0;  reasons.append("2 major vessels affected — elevated risk")
    elif ca == 1:
        score += 1.0;  reasons.append("1 major vessel affected — non-trivial blockage risk")

    if thal == 3:
        score += 1.5;  reasons.append("Reversible thallium defect — suggests ischemia")
    elif thal == 1:
        score += 0.5;  reasons.append("Fixed thallium defect — adds some risk")

    if thalach < 100:
        score += 0.5;  reasons.append("Low max heart rate — possible poor cardiac response")

    if int(values["sex"]) == 1 and age >= 55:
        score += 0.25; reasons.append("Older male — small additional risk signal")

    return min(score / 8.0, 1.0), reasons, warnings


def low_risk_profile_cap(values: dict[str, float | int]) -> tuple[float | None, list[str]]:
    age     = int(values["age"])
    thalach = int(values["thalach"])
    oldpeak = float(values["oldpeak"])
    cp      = int(values["cp"])
    exang   = int(values["exang"])
    slope   = int(values["slope"])
    ca      = int(values["ca"])
    thal    = int(values["thal"])

    age_max_hr = 220 - age
    # A good stress-test heart rate is >= 85% of age-adjusted max.
    # But the Cleveland dataset records peak exercise HR, not resting HR,
    # so we require at least 100 bpm as a loose sanity floor, and that the
    # reading doesn't fall more than 50 bpm below the age-adjusted maximum
    # (very low peak HR can indicate poor cardiac reserve).
    hr_ok = thalach >= 100 and thalach >= (age_max_hr - 50)
    is_strong_low = (
        age     <= 65
        and cp    in (0, 1)
        and exang == 0
        and oldpeak <= 0.5
        and slope  == 0
        and ca     == 0
        and thal   in (1, 2)    # model values: 2=Normal, 1=Fixed defect
        and hr_ok
    )
    if not is_strong_low:
        return None, []

    reasons = [
        "No major vessel blockage",
        "No exercise-induced angina",
        "Very low ST depression",
        "Upsloping ST slope",
        "Good heart-rate response for age",
        "Typical angina pattern without other abnormalities" if cp == 0
            else "Atypical angina without other abnormalities",
    ]
    if thal == 2:
        reasons.append("Normal thallium scan")
        return 0.18, reasons
    reasons.append("Fixed thallium defect present but overall profile is reassuring")
    return 0.24, reasons


# ── CSS / Styling ──────────────────────────────────────────────────────────
def render_styles() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

        <style>
        /* ── Variables ── */
        :root {
            --bg:       #070b14;
            --surface:  #0d1424;
            --card:     #111827;
            --border:   rgba(255,255,255,0.07);
            --red:      #e63946;
            --teal:     #2dd4bf;
            --text:     #f0f4ff;
            --muted:    #6b7a9b;
            --label:    #94a3c4;
            --amber:    #f59e0b;
            --font-head:'Playfair Display', serif;
            --font-body:'DM Sans', sans-serif;
        }

        /* ── App background ── */
        .stApp {
            background: var(--bg) !important;
            font-family: var(--font-body) !important;
            color: var(--text) !important;
        }
        /* subtle noise texture */
        .stApp::before {
            content: '';
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
        }
        /* ambient glow blobs */
        .stApp::after {
            content: '';
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background:
                radial-gradient(ellipse 600px 600px at -10% -15%, rgba(230,57,70,0.12), transparent 60%),
                radial-gradient(ellipse 500px 500px at 110% 110%, rgba(30,64,175,0.12), transparent 60%);
        }

        /* ── Layout ── */
        .block-container {
            max-width: 980px !important;
            padding-top: 0 !important;
            padding-bottom: 3rem !important;
            background: transparent !important;
            position: relative; z-index: 1;
        }
        html, body, [class*="css"], p, div, span {
            font-family: var(--font-body) !important;
        }
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Dark cards (border containers) ── */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 24px !important;
            box-shadow: 0 24px 60px rgba(0,0,0,0.45) !important;
            padding: 0.5rem !important;
            margin-bottom: 1.2rem !important;
        }

        /* ── Hero ── */
        .hero-card {
            padding: 3rem 2rem 2rem;
            text-align: center;
        }
        .cardioq-badge {
            display: inline-flex; align-items: center; gap: 7px;
            background: rgba(230,57,70,0.12); border: 1px solid rgba(230,57,70,0.3);
            padding: 5px 16px; border-radius: 100px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
            color: var(--red); text-transform: uppercase; margin-bottom: 16px;
        }
        .badge-dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--red); display: inline-block;
            animation: pulse-dot 1.4s ease infinite;
        }
        @keyframes pulse-dot {
            0%,100% { opacity:1; transform:scale(1); }
            50%     { opacity:0.4; transform:scale(0.6); }
        }
        .hero-title {
            margin: 0;
            font-family: var(--font-head) !important;
            font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 700; line-height: 1.15;
            background: linear-gradient(135deg, #fff 20%, #94a3c4 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            color: var(--muted); font-size: 14px; margin-top: 8px; letter-spacing: 0.04em;
        }

        /* ── ECG animation ── */
        .ecg-wrapper {
            width: 100%; max-width: 680px; height: 52px;
            margin: 20px auto 0; overflow: hidden; position: relative;
        }
        .ecg-wrapper svg { width: 200%; height: 100%; animation: ecg-scroll 3s linear infinite; }
        @keyframes ecg-scroll { from { transform:translateX(0); } to { transform:translateX(-50%); } }
        .ecg-wrapper::before, .ecg-wrapper::after {
            content:''; position:absolute; top:0; width:80px; height:100%; z-index:2;
        }
        .ecg-wrapper::before { left:0;  background:linear-gradient(to right, var(--bg), transparent); }
        .ecg-wrapper::after  { right:0; background:linear-gradient(to left,  var(--bg), transparent); }

        /* ── Stats band ── */
        .stats-band {
            display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 24px 0;
        }
        .stat-tile {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; padding: 20px 22px;
        }
        .stat-tile .num {
            font-family: var(--font-head) !important;
            font-size: 1.8rem; color: var(--teal); font-weight: 700;
        }
        .stat-tile .desc { font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.5; }

        /* ── Divider ── */
        .cardioq-divider { width:100%; height:1px; background:var(--border); margin:8px 0 28px; }

        /* ── Section headers ── */
        .section-head {
            display: flex; align-items: center; gap: 10px;
            padding: 20px 28px 4px; margin-bottom: 4px;
        }
        .section-icon-box {
            width: 34px; height: 34px; background: rgba(230,57,70,0.15);
            border: 1px solid rgba(230,57,70,0.3); border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; flex-shrink: 0;
        }
        .section-title {
            margin: 0;
            font-family: var(--font-head) !important;
            font-size: 1.3rem; font-weight: 600; color: var(--text);
        }

        /* ── Input labels ── */
        label[data-testid="stWidgetLabel"] p,
        label[data-testid="stWidgetLabel"] {
            font-size: 11px !important; font-weight: 600 !important;
            letter-spacing: 0.08em !important; color: var(--label) !important;
            text-transform: uppercase !important;
        }

        /* ── Inputs & selects ── */
        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
            font-family: var(--font-body) !important;
            font-size: 14px !important;
            min-height: 48px !important;
            box-shadow: none !important;
        }
        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:hover,
        div[data-testid="stNumberInput"] input:hover {
            border-color: rgba(45,212,191,0.3) !important;
        }
        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stNumberInput"] input:focus {
            border-color: rgba(45,212,191,0.5) !important;
            box-shadow: 0 0 0 3px rgba(45,212,191,0.1) !important;
        }
        [data-baseweb="select"] [data-baseweb="menu"] {
            background: #1a2035 !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stNumberInput"] button {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--label) !important;
        }

        /* ── Predict button ── */
        .stButton > button {
            width: 100% !important;
            border: none !important; border-radius: 14px !important;
            background: linear-gradient(135deg, #e63946, #c81d25) !important;
            color: white !important;
            font-family: var(--font-body) !important;
            font-size: 15px !important; font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            padding: 0.9rem 1.2rem !important;
            box-shadow: 0 6px 24px rgba(230,57,70,0.35) !important;
            transition: transform 0.15s, box-shadow 0.15s !important;
            margin-top: 16px !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 32px rgba(230,57,70,0.45) !important;
        }
        .stButton > button:active { transform: translateY(0) !important; }

        /* ── Result card ── */
        .result-card {
            border-radius: 20px; padding: 28px 32px;
            border: 1px solid; display: flex; align-items: center; gap: 24px;
            margin: 8px 20px 16px;
        }
        .result-card.high     { background: rgba(230,57,70,0.08);  border-color: rgba(230,57,70,0.3); }
        .result-card.moderate { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); }
        .result-card.low      { background: rgba(45,212,191,0.06); border-color: rgba(45,212,191,0.3); }

        .result-label {
            font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; margin-bottom: 4px;
        }
        .result-card.high     .result-label { color: var(--red); }
        .result-card.moderate .result-label { color: var(--amber); }
        .result-card.low      .result-label { color: var(--teal); }

        .result-headline {
            font-family: var(--font-head) !important;
            font-size: clamp(1.4rem, 3vw, 2rem); font-weight: 700;
            line-height: 1.2; color: var(--text);
        }
        .result-sub { font-size: 13px; color: var(--muted); margin-top: 8px; line-height: 1.6; }
        .gauge-wrap { flex-shrink: 0; }

        /* ── Warning card ── */
        .warning-card {
            border-radius: 12px; padding: 12px 16px; margin: 8px 20px;
            border: 1px solid rgba(245,158,11,0.35);
            background: rgba(245,158,11,0.08);
            color: #fcd34d; font-size: 13px; line-height: 1.55;
        }

        /* ── Error card ── */
        .error-card {
            border-radius: 12px; padding: 12px 16px; margin: 8px 20px;
            border: 1px solid rgba(230,57,70,0.4);
            background: rgba(230,57,70,0.1);
            color: #fca5a5; font-size: 13px; line-height: 1.55;
        }

        /* ── Factor items ── */
        .factor-section { padding: 4px 20px 16px; }
        .factors-title {
            font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--label); margin-bottom: 12px;
        }
        .factor-item {
            display: flex; align-items: flex-start; gap: 12px;
            padding: 12px 16px; background: rgba(255,255,255,0.03);
            border: 1px solid var(--border); border-radius: 12px;
            margin-bottom: 8px; font-size: 13px; line-height: 1.5; color: var(--text);
        }
        .factor-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--red); margin-top: 5px; flex-shrink: 0;
        }
        .factor-dot.teal { background: var(--teal); }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {
            border-radius: 14px; overflow: hidden;
            border: 1px solid var(--border) !important;
            margin: 0 20px 16px;
        }
        [data-testid="stDataFrame"] table { background: var(--surface) !important; }
        [data-testid="stDataFrame"] th {
            background: rgba(255,255,255,0.05) !important;
            color: var(--label) !important;
            font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        }
        [data-testid="stDataFrame"] td {
            color: var(--text) !important; font-size: 13px;
            border-color: var(--border) !important;
        }

        /* ── Spinner overlay ── */
        [data-testid="stSpinner"] { color: var(--teal) !important; }

        /* ── Disclaimer ── */
        .disclaimer-card {
            border-radius: 16px; padding: 20px 24px;
            border: 1px solid var(--border); background: var(--card);
            display: flex; gap: 14px; align-items: flex-start; margin-top: 4px;
        }
        .disc-icon { font-size: 20px; margin-top: 1px; flex-shrink: 0; }
        .disclaimer-card p { font-size: 13px; color: var(--muted); line-height: 1.7; margin: 0; }
        .disclaimer-card strong { color: var(--label); }

        /* ── Footer ── */
        .cardioq-footer {
            margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border);
            display: flex; justify-content: space-between; align-items: center;
            font-size: 12px; color: var(--muted); flex-wrap: wrap; gap: 10px;
        }
        .footer-brand {
            font-family: var(--font-head) !important;
            color: var(--label); font-size: 15px;
        }

        /* ── Inner padding ── */
        .input-inner { padding: 4px 20px 20px; }

        /* ── Responsive ── */
        @media (max-width: 640px) {
            .stats-band { grid-template-columns: 1fr; }
            .result-card { flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Gauge SVG ──────────────────────────────────────────────────────────────
def build_gauge_svg(percent: float, risk_label: str) -> str:
    r, cx, cy  = 52, 70, 70
    circ       = math.pi * r
    arc_len    = circ * min(percent / 100, 1)
    color      = {"High Risk": "#e63946", "Low Risk": "#2dd4bf"}.get(risk_label, "#f59e0b")

    angle = (percent / 100) * 180 - 180
    rad   = (angle * math.pi) / 180
    nx    = cx + r * math.cos(rad)
    ny    = cy + r * math.sin(rad)

    return f"""
    <svg width="140" height="85" viewBox="0 0 140 85">
      <path d="M 18 70 A 52 52 0 0 1 122 70"
        fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="10" stroke-linecap="round"/>
      <path d="M 18 70 A 52 52 0 0 1 122 70"
        fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
        stroke-dasharray="{arc_len:.2f} {circ:.2f}"
        style="filter:drop-shadow(0 0 6px {color})"/>
      <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"
            stroke="white" stroke-width="2.5" stroke-linecap="round" opacity="0.9"/>
      <circle cx="{cx}" cy="{cy}" r="5" fill="white" opacity="0.9"/>
      <text x="14"  y="83" font-size="9" fill="#6b7a9b" font-family="DM Sans,sans-serif">LOW</text>
      <text x="106" y="83" font-size="9" fill="#6b7a9b" font-family="DM Sans,sans-serif">HIGH</text>
      <text x="{cx}" y="{cy - 16}" text-anchor="middle" font-size="14" font-weight="700"
            fill="{color}" font-family="DM Sans,sans-serif">{round(percent)}%</text>
    </svg>"""


# ── UI helpers ─────────────────────────────────────────────────────────────
def render_section_header(title: str, icon: str) -> None:
    st.markdown(
        f'<div class="section-head">'
        f'<div class="section-icon-box">{icon}</div>'
        f'<h2 class="section-title">{title}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )


def format_option(feature: str, ui_code: int) -> str:
    for item in OPTION_SETS[feature]:
        if item["ui"] == ui_code:
            return f"{ui_code} — {item['label']}"
    return str(ui_code)


def render_result_card(label: str, risk_percent: float) -> None:
    class_map = {"High Risk": "high", "Low Risk": "low", "Moderate Risk": "moderate"}
    icon_map  = {"High Risk": "⚠", "Low Risk": "✓", "Moderate Risk": "⚡"}
    head_map  = {
        "High Risk":     "Elevated Cardiac Risk",
        "Low Risk":      "Cardiac Profile Appears Healthy",
        "Moderate Risk": "Moderate Cardiac Risk Detected",
    }
    sub_map = {
        "High Risk":     "Multiple risk indicators detected. Clinical consultation is strongly recommended.",
        "Low Risk":      "Parameters fall within generally healthy ranges. Maintain your lifestyle habits.",
        "Moderate Risk": "Clinical review is advisable. Monitor contributing indicators closely.",
    }

    risk_class = class_map.get(label, "moderate")
    gauge_svg  = build_gauge_svg(risk_percent, label)

    st.markdown(
        f'<div class="result-card {risk_class}">'
        f'  <div style="flex:1">'
        f'    <div class="result-label">{icon_map.get(label, "·")} {label}</div>'
        f'    <div class="result-headline">{head_map.get(label, label)}</div>'
        f'    <div class="result-sub">{sub_map.get(label, "")}</div>'
        f'  </div>'
        f'  <div class="gauge-wrap">{gauge_svg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# App Entry
# ══════════════════════════════════════════════════════════════════════════
render_styles()

artifact = load_artifact()
if artifact is None:
    st.error("⚠ Trained model not found at: `models/bagging_qsvc_quantum.joblib`")
    st.info(
        "Run the training script first:\n\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

model    = artifact["model"]
features = artifact["features"]

# ── Hero Header ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-card">
        <div class="cardioq-badge"><span class="badge-dot"></span>Quantum ML · QSVC Bagging Ensemble</div>
        <h1 class="hero-title">Cardiac Risk Intelligence</h1>
        <p class="hero-subtitle">Heart Disease Risk Prediction powered by Quantum Machine Learning</p>
        <div class="ecg-wrapper" aria-hidden="true">
          <svg viewBox="0 0 700 52" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="
              0,26 40,26 55,26 60,10 65,42 70,6 75,46 80,26 90,26
              130,26 145,26 150,10 155,42 160,6 165,46 170,26 180,26
              220,26 235,26 240,10 245,42 250,6 255,46 260,26 270,26
              310,26 325,26 330,10 335,42 340,6 345,46 350,26 360,26
              400,26 415,26 420,10 425,42 430,6 435,46 440,26 450,26
              490,26 505,26 510,10 515,42 520,6 525,46 530,26 540,26
              580,26 595,26 600,10 605,42 610,6 615,46 620,26 630,26
              670,26 685,26 690,10 695,42 700,6"
            stroke="#e63946" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <polyline points="
              700,26 740,26 755,26 760,10 765,42 770,6 775,46 780,26 790,26
              830,26 845,26 850,10 855,42 860,6 865,46 870,26 880,26
              920,26 935,26 940,10 945,42 950,6 955,46 960,26 970,26
              1010,26 1025,26 1030,10 1035,42 1040,6 1045,46 1050,26 1060,26
              1100,26 1115,26 1120,10 1125,42 1130,6 1135,46 1140,26 1150,26
              1190,26 1205,26 1210,10 1215,42 1220,6 1225,46 1230,26 1240,26
              1280,26 1295,26 1300,10 1305,42 1310,6 1315,46 1320,26 1330,26
              1370,26 1385,26 1390,10 1395,42 1400,6"
            stroke="#e63946" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
    </div>
    <div class="cardioq-divider"></div>
    """,
    unsafe_allow_html=True,
)

# ── Stats Band ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="stats-band">
        <div class="stat-tile">
            <div class="num">9</div>
            <div class="desc">Clinical input features analysed per prediction</div>
        </div>
        <div class="stat-tile">
            <div class="num">ZZMap</div>
            <div class="desc">ZZFeatureMap quantum kernel with bagging ensemble</div>
        </div>
        <div class="stat-tile">
            <div class="num">Real-time</div>
            <div class="desc">Instant risk stratification from entered parameters</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Patient Input Form ─────────────────────────────────────────────────────
display_rows:  list[dict[str, object]] = []
input_values:  dict[str, float | int]  = {}

with st.container(border=True):
    render_section_header("Patient Parameters", "🩺")
    st.markdown('<div class="input-inner">', unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="large")
    columns = [left_col, right_col]

    for idx, feature in enumerate(FEATURE_ORDER):
        col = columns[idx % 2]
        with col:
            if feature in OPTION_SETS:
                ui_codes = [item["ui"] for item in OPTION_SETS[feature]]
                sel = st.selectbox(
                    FEATURE_LABELS[feature],
                    options=ui_codes,
                    format_func=lambda code, f=feature: format_option(f, code),
                    key=f"ui_{feature}",
                )
                chosen = next(item for item in OPTION_SETS[feature] if item["ui"] == sel)
                input_values[feature] = chosen["model"]
                display_rows.append({
                    "Feature": FEATURE_LABELS[feature],
                    "Value":   chosen["ui"],
                    "Meaning": chosen["label"],
                })
                continue

            cfg = NUMERIC_CONFIG[feature]
            if cfg["format"] == "%d":
                val = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=int(cfg["min"]), max_value=int(cfg["max"]),
                    value=int(cfg["default"]), step=int(cfg["step"]),
                    format=cfg["format"], key=f"ui_{feature}",
                )
                parsed: int | float = int(val)
            else:
                val = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                    value=float(cfg["default"]), step=float(cfg["step"]),
                    format=cfg["format"], key=f"ui_{feature}",
                )
                parsed = float(val)

            input_values[feature] = parsed
            display_rows.append({
                "Feature": FEATURE_LABELS[feature],
                "Value":   parsed,
                "Meaning": "—",
            })

    btn_cols = st.columns([1.5, 1, 1.5])
    with btn_cols[1]:
        predict_clicked = st.button("Analyse Cardiac Risk →")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────
if predict_clicked:
    input_errors, input_warnings = validate_inputs(input_values)

    if input_errors:
        for msg in input_errors:
            st.markdown(f'<div class="error-card">❌ {msg}</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running quantum inference…"):
            model_input    = make_model_input(input_values, features)
            quantum_score  = quantum_risk_score(model, model_input)
            clinical_score, clinical_reasons, clinical_warnings = clinical_rule_risk(input_values)
            low_cap, low_reasons = low_risk_profile_cap(input_values)

        # Strong low-risk profile: trust the clinical cap as a hard override.
        # The quantum sigmoid can return ~0.5 for neutral inputs (sigmoid(0)=0.5)
        # which would wrongly push clean profiles into "Moderate".
        if low_cap is not None:
            final_risk = low_cap
        else:
            # Weighted blend: 40% quantum + 60% clinical rules.
            # Clinical rules encode domain knowledge more reliably for edge cases.
            final_risk = 0.4 * quantum_score + 0.6 * clinical_score
            # If clinical score is zero (truly clean profile) and quantum is
            # near-neutral (< 0.50), do not inflate above Low Risk threshold.
            if clinical_score == 0.0 and quantum_score < 0.50:
                final_risk = min(final_risk, 0.28)
            elif clinical_score < 0.15 and quantum_score < 0.45:
                final_risk = min(final_risk, 0.28)

        risk_label = "Moderate Risk"
        if final_risk >= 0.60:
            risk_label = "High Risk"
        elif final_risk <= 0.30:
            risk_label = "Low Risk"

        all_warnings = input_warnings + clinical_warnings

        # Result card
        with st.container(border=True):
            render_section_header("Prediction Result", "📊")
            for w in all_warnings:
                st.markdown(f'<div class="warning-card">⚠ {w}</div>', unsafe_allow_html=True)
            render_result_card(risk_label, final_risk * 100)

        # Contributing factors
        display_reasons = clinical_reasons or low_reasons
        if display_reasons:
            with st.container(border=True):
                render_section_header("Contributing Risk Factors", "🔍")
                dot_extra = "teal" if risk_label == "Low Risk" else ""
                items_html = "".join(
                    f'<div class="factor-item">'
                    f'<div class="factor-dot {dot_extra}"></div>'
                    f'<div>{r}</div></div>'
                    for r in display_reasons
                )
                st.markdown(
                    f'<div class="factor-section">'
                    f'<div class="factors-title">Detected Indicators</div>'
                    f'{items_html}</div>',
                    unsafe_allow_html=True,
                )

        # Submitted values table
        with st.container(border=True):
            render_section_header("Submitted Values", "📋")
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

# ── Disclaimer ─────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="disclaimer-card">'
    f'<div class="disc-icon">⚠️</div>'
    f'<p><strong>Clinical Disclaimer:</strong> {DISCLAIMER_TEXT}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="cardioq-footer">'
    '  <div class="footer-brand">CardioQ</div>'
    '  <div>ZZFeatureMap · Bagging QSVC Ensemble</div>'
    '  <div>Academic Research Prototype</div>'
    '</div>',
    unsafe_allow_html=True,
)