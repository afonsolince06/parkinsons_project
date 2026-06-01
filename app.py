"""
app.py
======
Streamlit Web Application — Parkinson's Disease ML Explorer
Theme: health (blue/green), modern layout with custom CSS.
Pages: Home | Dataset | Experiment (GA + PSO) | Results | Education
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
)

# ── Import algorithms ─────────────────────────────────────────────────────────
from genetic_algorithm_c import genetic_algorithm
from pso_c import particle_swarm_optimisation

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's ML Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Feature descriptions (22 acoustic features)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "MDVP:Fo(Hz)":        ("Average vocal fundamental frequency", "pitch", "Hz"),
    "MDVP:Fhi(Hz)":       ("Maximum vocal fundamental frequency", "pitch", "Hz"),
    "MDVP:Flo(Hz)":       ("Minimum vocal fundamental frequency", "pitch", "Hz"),
    "MDVP:Jitter(%)":     ("Cycle-to-cycle variation in fundamental frequency (%)", "jitter", "%"),
    "MDVP:Jitter(Abs)":   ("Absolute cycle-to-cycle variation in fundamental frequency", "jitter", "s"),
    "MDVP:RAP":           ("Relative Average Perturbation — short-term frequency variation", "jitter", ""),
    "MDVP:PPQ":           ("Five-point Period Perturbation Quotient", "jitter", ""),
    "Jitter:DDP":         ("Average absolute difference of differences between cycles", "jitter", ""),
    "MDVP:Shimmer":       ("Cycle-to-cycle variation in amplitude", "shimmer", ""),
    "MDVP:Shimmer(dB)":   ("Cycle-to-cycle amplitude variation in decibels", "shimmer", "dB"),
    "Shimmer:APQ3":       ("Three-point Amplitude Perturbation Quotient", "shimmer", ""),
    "Shimmer:APQ5":       ("Five-point Amplitude Perturbation Quotient", "shimmer", ""),
    "MDVP:APQ":           ("Average Perturbation Quotient (11-point window)", "shimmer", ""),
    "Shimmer:DDA":        ("Average absolute differences between consecutive differences", "shimmer", ""),
    "NHR":                ("Noise-to-Harmonics Ratio — noise in the voice signal", "noise", ""),
    "HNR":                ("Harmonics-to-Noise Ratio — signal quality measure", "noise", "dB"),
    "RPDE":               ("Recurrence Period Density Entropy — nonlinear dynamical complexity", "nonlinear", ""),
    "DFA":                ("Detrended Fluctuation Analysis — signal fractal scaling", "nonlinear", ""),
    "spread1":            ("Nonlinear measure of fundamental frequency variation", "nonlinear", ""),
    "spread2":            ("Nonlinear measure of fundamental frequency variation", "nonlinear", ""),
    "D2":                 ("Correlation dimension of the voice signal", "nonlinear", ""),
    "PPE":                ("Pitch Period Entropy — impaired control of fundamental frequency", "nonlinear", ""),
}

FEATURE_GROUPS = {
    "pitch":     ("🎵 Pitch / Fundamental Frequency", "#1a7fc4"),
    "jitter":    ("📳 Jitter (Frequency Variation)", "#e07b3a"),
    "shimmer":   ("📊 Shimmer (Amplitude Variation)", "#7752be"),
    "noise":     ("🔇 Noise Measures", "#c04848"),
    "nonlinear": ("🌀 Nonlinear Dynamics", "#1e8c6e"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Best hyperparameters found in grid search (from utilss.py)
# ─────────────────────────────────────────────────────────────────────────────
BEST_GA = {
    "hidden":    10,
    "pop_size":  50,
    "n_gen":     50,
    "mut_rate":  0.05,
    "crossover": "arithmetic",
    "metric":    "f1",
}

BEST_PSO = {
    "hidden":      10,
    "n_particles": 50,
    "n_iter":      50,
    "w":           0.9,
    "w_min":       0.4,
    "c1":          2.0,
    "c2":          2.0,
    "metric":      "f1",
    "patience":    15,
}

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d3b66 0%, #1a5f9e 60%, #1e8c6e 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

    /* Hero */
    .hero-banner {
        background: linear-gradient(135deg, #0d3b66 0%, #1a7fc4 45%, #1e8c6e 100%);
        border-radius: 18px; padding: 3rem 2.5rem; margin-bottom: 2rem;
        color: white; box-shadow: 0 8px 32px rgba(13,59,102,0.25);
        position: relative; overflow: hidden;
    }
    .hero-banner::before {
        content: ''; position: absolute; top: -50px; right: -50px;
        width: 250px; height: 250px; background: rgba(255,255,255,0.05); border-radius: 50%;
    }
    .hero-title { font-size: 2.8rem; font-weight: 800; margin: 0 0 0.5rem; line-height: 1.1; }
    .hero-subtitle { font-size: 1.1rem; opacity: 0.88; margin: 0 0 1.5rem; max-width: 600px; line-height: 1.6; }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3); border-radius: 50px;
        padding: 4px 16px; font-size: 0.78rem; font-weight: 600;
        letter-spacing: 0.5px; margin-bottom: 1.2rem; text-transform: uppercase;
    }

    /* Cards */
    .card {
        background: white; border-radius: 14px; padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(13,59,102,0.1); border: 1px solid rgba(26,127,196,0.1);
        height: 100%; transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(13,59,102,0.15); }
    .card-icon { font-size: 2rem; margin-bottom: 0.7rem; display: block; }
    .card-title { font-size: 1rem; font-weight: 700; color: #0d3b66; margin: 0 0 0.4rem; }
    .card-text { font-size: 0.87rem; color: #4a6080; line-height: 1.6; margin: 0; }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1a7fc4, #1e8c6e); border-radius: 14px;
        padding: 1.2rem 1.5rem; color: white; text-align: center;
        box-shadow: 0 4px 16px rgba(26,127,196,0.3);
    }
    .stat-value { font-size: 2.2rem; font-weight: 800; line-height: 1; margin-bottom: 0.3rem; }
    .stat-label { font-size: 0.75rem; font-weight: 500; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Metric cards */
    .metric-card {
        background: white; border-radius: 12px; padding: 1.1rem 1.2rem;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-top: 4px solid #1a7fc4;
    }
    .metric-card.green { border-top-color: #1e8c6e; }
    .metric-card.orange { border-top-color: #e07b3a; }
    .metric-card.purple { border-top-color: #7752be; }
    .metric-value { font-size: 1.9rem; font-weight: 800; color: #0d3b66; }
    .metric-label { font-size: 0.78rem; color: #6b7e96; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.2rem; }

    /* Section headers */
    .section-header { font-size: 1.55rem; font-weight: 700; color: #0d3b66; margin: 0 0 0.3rem; }
    .section-sub { font-size: 0.9rem; color: #5a7090; margin: 0 0 1.2rem; }
    .section-divider {
        height: 3px; background: linear-gradient(90deg, #1a7fc4, #1e8c6e, transparent);
        border: none; border-radius: 2px; margin: 0 0 1.5rem;
    }

    /* Algorithm panel */
    .algo-panel {
        background: white; border-radius: 16px; padding: 1.8rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07); margin-bottom: 2rem;
        border-left: 5px solid #1a7fc4;
    }
    .algo-panel.pso { border-left-color: #e07b3a; }
    .algo-panel-title { font-size: 1.25rem; font-weight: 700; color: #0d3b66; margin: 0 0 0.3rem; }
    .algo-panel-sub { font-size: 0.88rem; color: #5a7090; margin: 0 0 1.2rem; line-height: 1.6; }

    /* Info / warning / success boxes */
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #e8f5f0);
        border-left: 4px solid #1a7fc4; border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem; margin: 0.8rem 0; font-size: 0.88rem;
        color: #1a2433; line-height: 1.6;
    }
    .warn-box {
        background: #fff8e1; border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.87rem; color: #78450a; margin: 0.5rem 0;
    }
    .success-box {
        background: #e6f7ee; border-left: 4px solid #1e8c6e;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.87rem; color: #145c47; margin: 0.5rem 0;
    }

    /* Badges */
    .algo-badge-ga {
        display: inline-block; background: #dbeafe; color: #1e40af;
        border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700;
    }
    .algo-badge-pso {
        display: inline-block; background: #fce8d5; color: #c2510e;
        border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700;
    }

    /* Edu cards */
    .edu-card {
        background: white; border-radius: 16px; padding: 1.8rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07); margin-bottom: 1.5rem;
    }
    .edu-card h3 { color: #0d3b66; font-size: 1.15rem; font-weight: 700; margin-top: 0; }
    .edu-card p, .edu-card li { color: #3a4e66; font-size: 0.9rem; line-height: 1.7; }

    /* Step pipeline */
    .step-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
    .step-item {
        flex: 1; min-width: 100px; background: linear-gradient(135deg, #f0f7ff, #f0faf7);
        border-radius: 10px; padding: 0.9rem 0.5rem; text-align: center;
        border: 1px solid rgba(26,127,196,0.15);
    }
    .step-num {
        width: 32px; height: 32px; background: linear-gradient(135deg, #1a7fc4, #1e8c6e);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.85rem; margin: 0 auto 0.5rem;
    }
    .step-label { font-size: 0.75rem; font-weight: 600; color: #0d3b66; }

    /* Reference cards */
    .ref-card {
        background: #f8fafd; border-radius: 10px; padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem; border-left: 3px solid #1a7fc4;
        font-size: 0.87rem; color: #2a3a50;
    }
    .ref-card a { color: #1a7fc4; text-decoration: none; font-weight: 500; }
    .ref-card a:hover { text-decoration: underline; }

    /* Feature tag */
    .feat-tag {
        display: inline-block; border-radius: 5px; padding: 2px 8px;
        font-size: 0.72rem; font-weight: 600; margin-left: 6px;
    }

    /* Results comparison */
    .result-block {
        background: white; border-radius: 14px; padding: 1.5rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.08); height: 100%;
        border-top: 5px solid #1a7fc4;
    }
    .result-block.pso { border-top-color: #e07b3a; }
    .result-block-title { font-size: 1.1rem; font-weight: 700; color: #0d3b66; margin: 0 0 1rem; }
    .result-metric-row { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid #f0f4f9; }
    .result-metric-name { font-size: 0.85rem; color: #5a7090; font-weight: 500; }
    .result-metric-val { font-size: 1rem; font-weight: 700; color: #0d3b66; }

    /* ── Wix-style Home sections ── */
    .home-section {
        padding: 3.5rem 0 2.5rem;
        border-top: 1px solid #eaf0f8;
    }
    .home-section.alt {
        background: #f7fbff;
        border-radius: 16px;
        padding: 3rem 2rem;
        margin: 1.5rem 0;
    }
    .section-title-center {
        text-align: center;
        font-size: 1.9rem;
        font-weight: 700;
        color: #0d3b66;
        margin: 0 0 0.5rem;
    }
    .section-sub-center {
        text-align: center;
        font-size: 1rem;
        color: #5a7090;
        margin: 0 0 2.5rem;
        line-height: 1.6;
    }
    /* Circular icon (Wix style) */
    .service-icon-wrap {
        width: 80px; height: 80px;
        background: linear-gradient(135deg, #1a7fc4, #1e8c6e);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.1rem;
        font-size: 2rem;
        box-shadow: 0 4px 16px rgba(26,127,196,0.25);
    }
    .service-card {
        text-align: center;
        padding: 1.5rem 1rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(13,59,102,0.07);
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(26,127,196,0.08);
    }
    .service-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(13,59,102,0.13);
    }
    .service-card-title {
        font-size: 1.1rem; font-weight: 700;
        color: #1a7fc4; margin: 0 0 0.6rem;
    }
    .service-card-text {
        font-size: 0.88rem; color: #5a7090;
        line-height: 1.7; margin: 0 0 1rem;
    }
    .cta-link {
        color: #1e8c6e; font-weight: 600;
        font-size: 0.88rem; text-decoration: none;
        display: inline-block;
    }
    /* Stat bar */
    .stat-bar {
        background: linear-gradient(135deg, #0d3b66, #1a7fc4 50%, #1e8c6e);
        border-radius: 16px; padding: 2rem 1.5rem;
        color: white; display: flex;
        justify-content: space-around; align-items: center;
        flex-wrap: wrap; gap: 1rem;
        box-shadow: 0 6px 24px rgba(13,59,102,0.2);
        margin: 1.5rem 0;
    }
    .stat-bar-item { text-align: center; }
    .stat-bar-value { font-size: 2.5rem; font-weight: 800; line-height: 1; }
    .stat-bar-label { font-size: 0.75rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }
    .stat-bar-sep { width: 1px; height: 50px; background: rgba(255,255,255,0.2); }
    /* Info block (2-col layout) */
    .info-block {
        background: white; border-radius: 16px;
        padding: 2.2rem 2rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.07);
        height: 100%;
    }
    .info-block-label {
        display: inline-block;
        background: linear-gradient(135deg, #1a7fc4, #1e8c6e);
        color: white; border-radius: 50px;
        padding: 4px 16px; font-size: 0.72rem;
        font-weight: 700; letter-spacing: 0.5px;
        text-transform: uppercase; margin-bottom: 0.9rem;
    }
    .info-block h3 {
        font-size: 1.3rem; font-weight: 700;
        color: #0d3b66; margin: 0 0 0.8rem;
    }
    .info-block p, .info-block li {
        font-size: 0.9rem; color: #4a6080; line-height: 1.7;
    }
    .info-block ul { padding-left: 1.2rem; margin: 0; }
    /* Algorithm explainer */
    .algo-explainer {
        background: linear-gradient(135deg, #f0f7ff, #edfaf5);
        border-radius: 16px; padding: 2rem;
        border: 1px solid rgba(26,127,196,0.12);
    }
    .algo-explainer-title {
        font-size: 1rem; font-weight: 700;
        color: #0d3b66; margin: 0 0 0.4rem;
    }
    .algo-explainer-text {
        font-size: 0.87rem; color: #4a6080; line-height: 1.65; margin: 0;
    }
    /* Hero full */
    .hero-full {
        background: linear-gradient(135deg, #0d3b66 0%, #1a7fc4 50%, #1e8c6e 100%);
        border-radius: 18px; padding: 4rem 3rem;
        color: white; position: relative; overflow: hidden;
        box-shadow: 0 10px 40px rgba(13,59,102,0.3);
        margin-bottom: 0.5rem;
    }
    .hero-full::before {
        content: '';
        position: absolute; top: -80px; right: -80px;
        width: 350px; height: 350px;
        background: rgba(255,255,255,0.06); border-radius: 50%;
    }
    .hero-full::after {
        content: '';
        position: absolute; bottom: -100px; left: -60px;
        width: 400px; height: 400px;
        background: rgba(255,255,255,0.04); border-radius: 50%;
    }
    .hero-full-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 50px; padding: 5px 18px;
        font-size: 0.75rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase;
        margin-bottom: 1.4rem;
    }
    .hero-full-title {
        font-size: 3.2rem; font-weight: 800;
        line-height: 1.05; margin: 0 0 1rem;
        letter-spacing: -1px;
    }
    .hero-full-sub {
        font-size: 1.15rem; opacity: 0.9;
        line-height: 1.65; margin: 0 0 2rem;
        max-width: 580px;
    }
    .hero-cta-row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .hero-btn-primary {
        display: inline-block;
        background: white; color: #1a7fc4;
        border-radius: 50px; padding: 0.75rem 2rem;
        font-weight: 700; font-size: 0.95rem;
        text-decoration: none; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        transition: transform 0.2s;
        cursor: pointer;
    }
    .hero-btn-outline {
        display: inline-block;
        background: transparent; color: white;
        border: 2px solid rgba(255,255,255,0.7);
        border-radius: 50px; padding: 0.72rem 1.8rem;
        font-weight: 600; font-size: 0.95rem;
        text-decoration: none; cursor: pointer;
        transition: background 0.2s;
    }
    /* Home footer */
    .home-footer {
        background: #0d3b66; color: white;
        border-radius: 16px; padding: 2.5rem 2rem;
        margin-top: 2rem;
    }
    .footer-title {
        font-size: 1.2rem; font-weight: 700;
        margin: 0 0 0.3rem;
    }
    .footer-sub {
        font-size: 0.82rem; opacity: 0.7;
        margin: 0 0 1.5rem; line-height: 1.6;
    }
    .footer-links a {
        color: rgba(255,255,255,0.75);
        text-decoration: none; font-size: 0.82rem;
        margin-right: 1.5rem; line-height: 2;
    }
    .footer-links a:hover { color: white; }
    .footer-bottom {
        border-top: 1px solid rgba(255,255,255,0.12);
        padding-top: 1rem; margin-top: 1.5rem;
        font-size: 0.75rem; opacity: 0.55;
        text-align: center;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE  = 22
OUTPUT_SIZE = 1


@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons_preprocessed.csv")
    return df


def compute_n_params(hidden_size=10):
    return INPUT_SIZE * hidden_size + hidden_size + hidden_size * OUTPUT_SIZE + OUTPUT_SIZE


def unpack_weights(solution, hidden_size):
    idx = 0
    nW1 = INPUT_SIZE * hidden_size
    W1  = solution[idx:idx + nW1].reshape(INPUT_SIZE, hidden_size); idx += nW1
    b1  = solution[idx:idx + hidden_size];                           idx += hidden_size
    nW2 = hidden_size * OUTPUT_SIZE
    W2  = solution[idx:idx + nW2].reshape(hidden_size, OUTPUT_SIZE); idx += nW2
    b2  = solution[idx:idx + OUTPUT_SIZE]
    return [W1, W2], [b1, b2]


def make_fitness(metric="f1", hidden_size=10):
    def fitness_fn(solution, X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_size,), activation="relu",
            solver="sgd", max_iter=1, random_state=42
        )
        clf.fit(np.zeros((2, INPUT_SIZE)), np.array([0, 1]))
        clf.coefs_, clf.intercepts_ = unpack_weights(solution, hidden_size)
        y_pred = clf.predict(X)
        if metric == "recall":
            return recall_score(y, y_pred, pos_label=1, zero_division=0)
        if metric == "f1":
            return f1_score(y, y_pred, pos_label=1, zero_division=0)
        return accuracy_score(y, y_pred)
    return fitness_fn


def full_eval(solution, hidden_size, X, y):
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,), activation="relu",
        solver="sgd", max_iter=1, random_state=42
    )
    clf.fit(np.zeros((2, INPUT_SIZE)), np.array([0, 1]))
    clf.coefs_, clf.intercepts_ = unpack_weights(solution, hidden_size)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "recall":     recall_score(y, y_pred, pos_label=1, zero_division=0),
        "precision":  precision_score(y, y_pred, pos_label=1, zero_division=0),
        "f1":         f1_score(y, y_pred, pos_label=1, zero_division=0),
        "accuracy":   accuracy_score(y, y_pred),
        "n_pred_pos": int((y_pred == 1).sum()),
        "n_pred_neg": int((y_pred == 0).sum()),
        "degenerate": int((y_pred == 0).sum()) == 0,
        "confusion_matrix": cm,
    }


def _make_convergence_fig(history, label, color, title="Convergence Curve"):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history) + 1)),
        y=history,
        mode="lines",
        name=label,
        line=dict(color=color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.08)",
    ))
    fig.update_layout(
        xaxis_title="Generation / Iteration",
        yaxis_title="Best Fitness",
        yaxis=dict(range=[0, 1.05]),
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=12),
        showlegend=False,
        title=dict(text=title, font=dict(size=13, color="#0d3b66"), x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8eef5", gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#e8eef5", gridwidth=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Live execution — GA
# ─────────────────────────────────────────────────────────────────────────────
def run_ga_live(params, X, y, plot_placeholder, progress_bar, status_text):
    fitness_fn = make_fitness(params["metric"], params["hidden"])
    n_params   = compute_n_params(params["hidden"])
    np.random.seed(42)

    pop_size = params["pop_size"]
    n_gen    = params["n_gen"]
    mut_rate = params["mut_rate"]

    population    = np.random.uniform(-1.0, 1.0, size=(pop_size, n_params))
    best_solution = None
    best_fitness  = -np.inf
    history       = []

    for gen in range(n_gen):
        fitnesses = np.array([fitness_fn(ind, X, y) for ind in population])
        gen_best_idx = np.argmax(fitnesses)

        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness  = fitnesses[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        history.append(best_fitness)

        progress_bar.progress((gen + 1) / n_gen)
        status_text.markdown(
            f"<small>Generation **{gen+1}/{n_gen}** — Best fitness: **{best_fitness:.4f}**</small>",
            unsafe_allow_html=True,
        )

        update_every = max(1, n_gen // 20)
        if (gen + 1) % update_every == 0 or gen == n_gen - 1:
            fig = _make_convergence_fig(history, "GA", "#1a7fc4", "GA Convergence")
            plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"ga_{gen}")

        # Next generation
        elite_idx = np.argsort(fitnesses)[::-1][:2]
        next_gen  = [population[i].copy() for i in elite_idx]

        while len(next_gen) < pop_size:
            i1 = np.random.choice(pop_size, size=3, replace=False)
            i2 = np.random.choice(pop_size, size=3, replace=False)
            p1 = population[i1[np.argmax(fitnesses[i1])]].copy()
            p2 = population[i2[np.argmax(fitnesses[i2])]].copy()

            if np.random.rand() < 0.8:
                if params["crossover"] == "arithmetic":
                    a  = np.random.uniform(0, 1)
                    c1 = a * p1 + (1 - a) * p2
                    c2 = (1 - a) * p1 + a * p2
                else:
                    d  = np.abs(p1 - p2)
                    lo = np.minimum(p1, p2) - 0.5 * d
                    hi = np.maximum(p1, p2) + 0.5 * d
                    c1 = np.random.uniform(lo, hi)
                    c2 = np.random.uniform(lo, hi)
            else:
                c1, c2 = p1.copy(), p2.copy()

            mask1 = np.random.rand(n_params) < mut_rate
            mask2 = np.random.rand(n_params) < mut_rate
            c1[mask1] += np.random.normal(0, 0.1, mask1.sum())
            c2[mask2] += np.random.normal(0, 0.1, mask2.sum())

            next_gen.append(c1)
            if len(next_gen) < pop_size:
                next_gen.append(c2)

        population = np.array(next_gen)

    return best_solution, best_fitness, history


# ─────────────────────────────────────────────────────────────────────────────
# Live execution — PSO
# ─────────────────────────────────────────────────────────────────────────────
def run_pso_live(params, X, y, plot_placeholder, progress_bar, status_text):
    fitness_fn  = make_fitness(params["metric"], params["hidden"])
    n_params    = compute_n_params(params["hidden"])
    np.random.seed(42)

    n_part   = params["n_particles"]
    n_iter   = params["n_iter"]
    w_start  = params["w"]
    w_min    = params["w_min"]
    c1_coef  = params["c1"]
    c2_coef  = params["c2"]
    patience = params.get("patience", 15)
    v_max    = 0.4  # (high=1 - low=-1) * 0.2

    positions  = np.random.uniform(-1.0, 1.0, size=(n_part, n_params))
    velocities = np.random.uniform(-v_max, v_max, size=(n_part, n_params))

    # Initial evaluation
    fitnesses = np.array([fitness_fn(positions[i], X, y) for i in range(n_part)])

    personal_best_pos = positions.copy()
    personal_best_fit = fitnesses.copy()
    g_best_idx        = np.argmax(fitnesses)
    global_best_pos   = positions[g_best_idx].copy()
    global_best_fit   = float(fitnesses[g_best_idx])

    history      = []
    no_improve   = 0
    tol          = 1e-6

    for it in range(n_iter):
        # Inertia linear decay
        current_w = w_start - (w_start - w_min) * (it / max(n_iter - 1, 1))

        r1 = np.random.rand(n_part, n_params)
        r2 = np.random.rand(n_part, n_params)

        cognitive  = c1_coef * r1 * (personal_best_pos - positions)
        social     = c2_coef * r2 * (global_best_pos   - positions)

        velocities = current_w * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)
        positions  = positions + velocities

        # Evaluate
        fitnesses = np.array([fitness_fn(positions[i], X, y) for i in range(n_part)])

        # Update personal bests
        improved = fitnesses > personal_best_fit
        personal_best_pos[improved] = positions[improved].copy()
        personal_best_fit[improved] = fitnesses[improved]

        # Update global best
        cur_best_idx = int(np.argmax(personal_best_fit))
        cur_best_fit = float(personal_best_fit[cur_best_idx])

        if cur_best_fit > global_best_fit + tol:
            global_best_fit = cur_best_fit
            global_best_pos = personal_best_pos[cur_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        history.append(global_best_fit)

        progress_bar.progress((it + 1) / n_iter)
        status_text.markdown(
            f"<small>Iteration **{it+1}/{n_iter}** — Best fitness: **{global_best_fit:.4f}** — w: {current_w:.3f}</small>",
            unsafe_allow_html=True,
        )

        update_every = max(1, n_iter // 20)
        if (it + 1) % update_every == 0 or it == n_iter - 1:
            fig = _make_convergence_fig(history, "PSO", "#e07b3a", "PSO Convergence")
            plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"pso_{it}")

        if no_improve >= patience:
            history += [global_best_fit] * (n_iter - len(history))
            break

    return global_best_pos, global_best_fit, history


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home(df):
    n_samples  = len(df)
    n_features = df.shape[1] - 1
    n_pd       = int((df["status"] == 1).sum())
    n_healthy  = int((df["status"] == 0).sum())

    # ── HERO ─────────────────────────────────────────────────────────────────
    col_hero, col_hero_r = st.columns([3, 1])
    with col_hero:
        st.markdown("""
        <div class="hero-full">
            <div class="hero-full-badge">🧠 Optimization Algorithms Project · NOVA IMS 2025/26</div>
            <div class="hero-full-title">Parkinson's Disease<br>ML Explorer</div>
            <div class="hero-full-sub">
                Discover how <strong>Genetic Algorithms</strong> and <strong>Particle Swarm Optimization</strong>
                train a neural network to detect Parkinson's Disease from voice measurements —
                with no backpropagation. Explore the data, run the algorithms, and compare results.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CTA buttons as real Streamlit buttons below the hero
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([2,2,2,3])
    with btn_c1:
        if st.button("ℹ️ Learn More", use_container_width=True, key="hero_learn"):
            st.session_state["page"] = "📚 Education"
            st.rerun()
    with btn_c2:
        if st.button("📊 Explore Data", use_container_width=True, key="hero_data"):
            st.session_state["page"] = "📊 Dataset"
            st.rerun()
    with btn_c3:
        if st.button("▶ Try It Now", use_container_width=True, type="primary", key="hero_try"):
            st.session_state["page"] = "🧬 Experiment"
            st.rerun()

    # ── STAT BAR ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-bar">
        <div class="stat-bar-item">
            <div class="stat-bar-value">{n_samples}</div>
            <div class="stat-bar-label">Voice Recordings</div>
        </div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item">
            <div class="stat-bar-value">{n_features}</div>
            <div class="stat-bar-label">Acoustic Features</div>
        </div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item">
            <div class="stat-bar-value">{n_pd}</div>
            <div class="stat-bar-label">Parkinson's Cases</div>
        </div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item">
            <div class="stat-bar-value">{n_healthy}</div>
            <div class="stat-bar-label">Healthy Controls</div>
        </div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item">
            <div class="stat-bar-value">2</div>
            <div class="stat-bar-label">Algorithms (GA & PSO)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ABOUT PARKINSON'S — Wix service card style ────────────────────────────
    st.markdown("""
    <div class="home-section">
        <div class="section-title-center">Understanding Parkinson's Disease</div>
        <div class="section-sub-center">A neurodegenerative disorder affecting millions worldwide —<br>detectable through the voice before clinical diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    service_cards = [
        ("🌍", "Global Epidemiology",
         "More than <strong>10 million people</strong> worldwide live with Parkinson's — the 2nd most common neurodegenerative disease. Affects ~1% of people over 60 years old.",
         "📚 Education"),
        ("🔬", "Symptoms & Diagnosis",
         "Classic symptoms include <strong>resting tremor</strong>, muscle rigidity, bradykinesia, and postural instability. Voice alterations are among the earliest detectable signs.",
         "📚 Education"),
        ("🎤", "Voice-Based Detection",
         "Acoustic features like <strong>jitter, shimmer and HNR</strong> capture subtle changes in vocal production. This dataset uses 22 features from sustained phonation recordings.",
         "📊 Dataset"),
    ]
    for col, (icon, title, text, page_target) in zip([col1, col2, col3], service_cards):
        with col:
            st.markdown(f"""
            <div class="service-card">
                <div class="service-icon-wrap">{icon}</div>
                <div class="service-card-title">{title}</div>
                <p class="service-card-text">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── HOW IT WORKS — 2-column info blocks ───────────────────────────────────
    st.markdown("""
    <div class="home-section">
        <div class="section-title-center">How the Project Works</div>
        <div class="section-sub-center">We replace traditional backpropagation with evolutionary and swarm intelligence algorithms<br>to optimize the neural network's weights directly.</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("""
        <div class="info-block">
            <div class="info-block-label">🧬 Genetic Algorithm</div>
            <h3>Evolution in Action</h3>
            <p>A population of candidate weight vectors <em>θ</em> evolves over generations
            through <strong>selection</strong>, <strong>crossover</strong>, and <strong>mutation</strong>.
            The fittest individuals survive and reproduce, gradually improving the neural network's performance.</p>
            <ul>
                <li>Tournament selection for parent choice</li>
                <li>Arithmetic or Blend (BLX-α) crossover</li>
                <li>Gaussian mutation for local search</li>
                <li>Elitism to preserve top solutions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="info-block">
            <div class="info-block-label">🌊 Particle Swarm Optimization</div>
            <h3>Collective Intelligence</h3>
            <p>A swarm of particles flies through the high-dimensional weight space,
            each guided by its own best position (<strong>cognitive</strong>) and the
            swarm's global best (<strong>social</strong>). Inertia decays linearly to
            balance exploration and exploitation.</p>
            <ul>
                <li>Linear inertia weight decay (LDIW)</li>
                <li>Cognitive + social acceleration coefficients</li>
                <li>Velocity clamping to prevent divergence</li>
                <li>Early stopping on convergence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EXPLORE SECTION — 3 service cards with Streamlit CTA buttons ──────────
    st.markdown("""
    <div class="home-section alt">
        <div class="section-title-center">Explore the Project</div>
        <div class="section-sub-center">Three interactive sections — from raw data to live algorithm execution.</div>
    </div>
    """, unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    explore_cards = [
        ("e1", "📊", "Explore Dataset",
         "Visualize the 22 acoustic features, class distributions, scatter plots, and correlation heatmaps. Understand what makes a Parkinson's voice different.",
         "📊 Dataset", "btn_ds"),
        ("e2", "🧬", "Run GA & PSO Live",
         "Configure parameters and run Genetic Algorithm or Particle Swarm Optimization in real time. Watch convergence curves update as the algorithm runs!",
         "🧬 Experiment", "btn_algo"),
        ("e3", "📈", "Compare Results",
         "After running both algorithms, compare their convergence curves, metrics (Accuracy, F1, Recall), and confusion matrices side-by-side.",
         "📈 Results", "btn_res"),
    ]
    for col, (_, icon, title, text, page_target, btn_key) in zip([e1, e2, e3], explore_cards):
        with col:
            st.markdown(f"""
            <div class="service-card">
                <div class="service-icon-wrap">{icon}</div>
                <div class="service-card-title">{title}</div>
                <p class="service-card-text">{text}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Go to {title} →", use_container_width=True, key=btn_key):
                st.session_state["page"] = page_target
                st.rerun()

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="home-footer">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:2rem;">
            <div style="flex:1;min-width:200px">
                <div class="footer-title">🧠 Parkinson's ML Explorer</div>
                <div class="footer-sub">Optimization Algorithms Project<br>NOVA IMS · 2025/26</div>
                <div class="footer-links">
                    <a href="https://archive.ics.uci.edu/dataset/174/parkinsons" target="_blank">UCI Dataset</a>
                    <a href="https://www.parkinson.org/" target="_blank">Parkinson's Foundation</a>
                    <a href="https://ieeexplore.ieee.org/document/488968" target="_blank">PSO Paper</a>
                </div>
            </div>
            <div style="flex:1;min-width:200px">
                <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem">Quick Links</div>
                <div class="footer-links" style="display:flex;flex-direction:column">
                    <a href="#">📊 Dataset Explorer</a>
                    <a href="#">🧬 Run Algorithms</a>
                    <a href="#">📈 Results</a>
                    <a href="#">📚 Learn More</a>
                </div>
            </div>
            <div style="flex:1;min-width:200px">
                <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem">Dataset</div>
                <div style="font-size:0.8rem;opacity:0.7;line-height:1.7">
                    Max A. Little et al. (2007)<br>
                    UCI ML Repository<br>
                    195 samples · 22 features<br>
                    Binary classification task
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            © 2025/26 NOVA IMS · Optimization Algorithms Project · Built with Streamlit & Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────
def page_dataset(df):
    st.markdown('<div class="section-header">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Explore the acoustic voice features used for Parkinson\'s detection</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    X_cols    = [c for c in df.columns if c != "status"]
    n_samples  = len(df)
    n_features = len(X_cols)
    n_pd       = int((df["status"] == 1).sum())
    n_healthy  = int((df["status"] == 0).sum())

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_samples}</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card green"><div class="metric-value">{n_features}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card orange"><div class="metric-value">{n_pd} ({n_pd/n_samples:.0%})</div><div class="metric-label">Parkinson (1)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card purple"><div class="metric-value">{n_healthy} ({n_healthy/n_samples:.0%})</div><div class="metric-label">Healthy (0)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class distribution + histogram ────────────────────────────────────────
    st.markdown("#### Class Distribution & Feature Histograms")
    col_pie, col_hist = st.columns([1, 2])

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Parkinson", "Healthy"],
            values=[n_pd, n_healthy],
            hole=0.48,
            marker=dict(colors=["#1a7fc4", "#1e8c6e"],
                        line=dict(color="white", width=2)),
            textinfo="percent+label",
            textposition="outside",
            pull=[0.03, 0],
        ))
        fig_pie.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            font=dict(family="Inter", size=13),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_hist:
        selected_feat = st.selectbox("Select a feature:", X_cols, index=0, key="feat_hist")
        df_plot = df[[selected_feat, "status"]].copy()
        df_plot["Class"] = df_plot["status"].map({1: "Parkinson", 0: "Healthy"})
        fig_hist = px.histogram(
            df_plot, x=selected_feat, color="Class", barmode="overlay",
            color_discrete_map={"Parkinson": "#1a7fc4", "Healthy": "#1e8c6e"},
            nbins=30, opacity=0.72,
        )
        fig_hist.update_layout(
            height=300, margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor="#e8eef5")
        fig_hist.update_yaxes(showgrid=True, gridcolor="#e8eef5")
        # Show feature description
        if selected_feat in FEATURE_DESCRIPTIONS:
            desc, group, unit = FEATURE_DESCRIPTIONS[selected_feat]
            g_label, g_color = FEATURE_GROUPS[group]
            st.markdown(
                f'<span style="background:{g_color}22;color:{g_color};border-radius:5px;'
                f'padding:2px 8px;font-size:0.78rem;font-weight:600">{g_label}</span>'
                f' &nbsp;<span style="font-size:0.87rem;color:#4a6080">{desc}'
                f'{"  (" + unit + ")" if unit else ""}</span>',
                unsafe_allow_html=True
            )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.markdown("#### Scatter Plot — Compare Two Features")
    sc1, sc2 = st.columns(2)
    with sc1:
        feat_x = st.selectbox("Feature X:", X_cols, index=0, key="feat_x")
    with sc2:
        feat_y = st.selectbox("Feature Y:", X_cols, index=1, key="feat_y")

    df_s = df[[feat_x, feat_y, "status"]].copy()
    df_s["Class"] = df_s["status"].map({1: "Parkinson", 0: "Healthy"})
    fig_scatter = px.scatter(
        df_s, x=feat_x, y=feat_y, color="Class",
        color_discrete_map={"Parkinson": "#1a7fc4", "Healthy": "#1e8c6e"},
        opacity=0.72,
    )
    fig_scatter.update_traces(marker=dict(size=7))
    fig_scatter.update_layout(
        height=380, margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=12),
    )
    fig_scatter.update_xaxes(showgrid=True, gridcolor="#e8eef5")
    fig_scatter.update_yaxes(showgrid=True, gridcolor="#e8eef5")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Box plots ─────────────────────────────────────────────────────────────
    st.markdown("#### Box Plots — Feature Distribution by Class")
    n_show = st.slider("Number of features to display:", 4, len(X_cols), 8, 2, key="n_box")
    df_melt = df[X_cols[:n_show] + ["status"]].copy()
    df_melt["Class"] = df_melt["status"].map({1: "Parkinson", 0: "Healthy"})
    df_long = df_melt.melt(id_vars=["Class"], value_vars=X_cols[:n_show],
                           var_name="Feature", value_name="Value")
    fig_box = px.box(
        df_long, x="Feature", y="Value", color="Class",
        color_discrete_map={"Parkinson": "#1a7fc4", "Healthy": "#1e8c6e"},
        points=False,
    )
    fig_box.update_layout(
        height=400, margin=dict(l=40, r=20, t=20, b=80),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature descriptions ──────────────────────────────────────────────────
    with st.expander("📖 Feature Descriptions — All 22 Variables", expanded=False):
        for group_key, (group_label, group_color) in FEATURE_GROUPS.items():
            st.markdown(
                f'<div style="background:{group_color}18;border-left:4px solid {group_color};'
                f'padding:0.5rem 1rem;border-radius:0 8px 8px 0;margin:0.8rem 0 0.4rem;">'
                f'<strong style="color:{group_color}">{group_label}</strong></div>',
                unsafe_allow_html=True
            )
            rows = []
            for feat, (desc, grp, unit) in FEATURE_DESCRIPTIONS.items():
                if grp == group_key:
                    rows.append({
                        "Feature": feat,
                        "Description": desc,
                        "Unit": unit if unit else "—",
                        "PD Mean": f"{df.loc[df.status==1, feat].mean():.4f}" if feat in df.columns else "—",
                        "Healthy Mean": f"{df.loc[df.status==0, feat].mean():.4f}" if feat in df.columns else "—",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    with st.expander("🔥 Correlation Heatmap", expanded=False):
        corr = df[X_cols].corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        )
        fig_corr.update_layout(
            height=550, font=dict(family="Inter", size=10),
            margin=dict(l=60, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Full dataset table ────────────────────────────────────────────────────
    with st.expander("📋 Full Dataset", expanded=False):
        df_display = df.copy()
        df_display["status"] = df_display["status"].map({1: "🔵 Parkinson", 0: "🟢 Healthy"})
        st.dataframe(df_display, use_container_width=True, height=400)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "parkinsons_data.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Experiment (GA + PSO side by side)
# ─────────────────────────────────────────────────────────────────────────────
def page_experiment(df):
    X = df.drop(columns=["status"]).values.astype(float)
    y = df["status"].values.astype(int)

    st.markdown('<div class="section-header">🧬 Run Algorithms</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Configure parameters for each algorithm and run them independently. '
        'Default values are the best configurations found in our grid search.</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        🎯 <strong>How it works:</strong> Both GA and PSO optimize the weights <em>θ</em> of an
        MLP neural network (no backpropagation!). Each algorithm searches for the weight vector
        that maximizes the chosen fitness metric on the Parkinson's dataset.
        Tune the parameters below and click <strong>▶ Run</strong> to see the convergence live.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Two columns: GA | PSO ─────────────────────────────────────────────────
    col_ga, col_pso = st.columns(2, gap="large")

    # ════════════════════════════════════════════════════════
    # GA PANEL
    # ════════════════════════════════════════════════════════
    with col_ga:
        st.markdown("""
        <div class="algo-panel">
            <div class="algo-panel-title">🧬 Genetic Algorithm (GA)</div>
            <div class="algo-panel-sub">
                Inspired by natural selection. A <em>population</em> of candidate weight vectors
                evolves over generations through <strong>selection</strong>, <strong>crossover</strong>
                and <strong>mutation</strong>. The fittest individuals survive and reproduce.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ GA Parameters", expanded=True):
            ga_hidden  = st.slider("Hidden Neurons", 5, 30, BEST_GA["hidden"], 1,
                                   key="ga_hidden", help="Number of neurons in the MLP hidden layer")
            ga_fitness = st.selectbox("Fitness Metric", ["f1", "recall", "accuracy"],
                                      index=["f1", "recall", "accuracy"].index(BEST_GA["metric"]),
                                      key="ga_metric",
                                      help="Metric to maximize. F1 is recommended (balanced).")
            ga_pop     = st.slider("Population Size", 10, 100, BEST_GA["pop_size"], 5,
                                   key="ga_pop", help="Number of individuals per generation")
            ga_gen     = st.slider("Generations", 10, 200, BEST_GA["n_gen"], 5,
                                   key="ga_gen", help="Number of evolution generations")
            ga_mut     = st.slider("Mutation Rate", 0.01, 0.30, BEST_GA["mut_rate"], 0.01,
                                   key="ga_mut", format="%.2f",
                                   help="Probability of mutating each weight gene")
            ga_cx      = st.selectbox("Crossover Method",
                                      ["arithmetic", "blend"],
                                      index=["arithmetic", "blend"].index(BEST_GA["crossover"]),
                                      key="ga_cx",
                                      help="Arithmetic: interpolation | Blend (BLX-α): extrapolation")

        ga_params = {
            "algo": "GA", "hidden": ga_hidden, "metric": ga_fitness,
            "pop_size": ga_pop, "n_gen": ga_gen,
            "mut_rate": ga_mut, "crossover": ga_cx,
        }
        n_params_ga = compute_n_params(ga_hidden)
        st.markdown(
            f'<div class="info-box" style="margin-top:0.5rem">ℹ️ MLP with <strong>{ga_hidden}</strong> hidden neurons '
            f'→ <strong>{n_params_ga}</strong> parameters to optimize</div>',
            unsafe_allow_html=True
        )

        run_ga_btn = st.button("▶ Run GA", use_container_width=True, type="primary", key="run_ga")

        ga_plot_ph  = st.empty()
        ga_prog     = st.progress(0)
        ga_status   = st.empty()

        # Show stored result if exists
        if "last_ga" in st.session_state and not run_ga_btn:
            gd = st.session_state["last_ga"]
            fig = _make_convergence_fig(gd["history"], "GA", "#1a7fc4", "GA Convergence")
            ga_plot_ph.plotly_chart(fig, use_container_width=True, key="ga_stored")
            ga_prog.progress(1.0)
            _show_metrics_mini(gd["metrics"], "GA")

        if run_ga_btn:
            t0 = time.perf_counter()
            best_sol, best_fit, history = run_ga_live(
                ga_params, X, y, ga_plot_ph, ga_prog, ga_status
            )
            elapsed = time.perf_counter() - t0
            metrics = full_eval(best_sol, ga_hidden, X, y)
            ga_status.empty()

            st.session_state["last_ga"] = {
                "params": ga_params, "history": history,
                "metrics": metrics, "elapsed": elapsed,
                "best_solution": best_sol,
            }
            st.markdown(f"""
            <div class="success-box">
                ✅ <strong>GA completed</strong> in {elapsed:.1f}s — Best fitness: <strong>{best_fit:.4f}</strong>
            </div>""", unsafe_allow_html=True)
            if metrics["degenerate"]:
                st.markdown("""
                <div class="warn-box">⚠️ <strong>Degenerate solution</strong> — the model predicted only the positive class.
                Try changing the fitness metric to <em>f1</em>.</div>""", unsafe_allow_html=True)
            _show_metrics_mini(metrics, "GA")

    # ════════════════════════════════════════════════════════
    # PSO PANEL
    # ════════════════════════════════════════════════════════
    with col_pso:
        st.markdown("""
        <div class="algo-panel pso">
            <div class="algo-panel-title">🌊 Particle Swarm Optimization (PSO)</div>
            <div class="algo-panel-sub">
                Inspired by bird flocks and fish schools. A <em>swarm</em> of particles
                flies through the weight space, each guided by its own best memory
                (<strong>cognitive</strong>) and the swarm's best position (<strong>social</strong>).
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ PSO Parameters", expanded=True):
            pso_hidden  = st.slider("Hidden Neurons", 5, 30, BEST_PSO["hidden"], 1,
                                    key="pso_hidden", help="Number of neurons in the MLP hidden layer")
            pso_fitness = st.selectbox("Fitness Metric", ["f1", "recall", "accuracy"],
                                       index=["f1", "recall", "accuracy"].index(BEST_PSO["metric"]),
                                       key="pso_metric",
                                       help="Metric to maximize. F1 is recommended (balanced).")
            pso_part    = st.slider("Number of Particles", 10, 100, BEST_PSO["n_particles"], 5,
                                    key="pso_part", help="Swarm size — more particles = better exploration")
            pso_iter    = st.slider("Iterations", 10, 200, BEST_PSO["n_iter"], 5,
                                    key="pso_iter", help="Number of PSO iterations")
            pso_w       = st.slider("Inertia weight (w)", 0.4, 1.0, BEST_PSO["w"], 0.05,
                                    key="pso_w", format="%.2f",
                                    help="Starting inertia — controls exploration vs exploitation")
            pso_wmin    = st.slider("Min. inertia (w_min)", 0.1, 0.6, BEST_PSO["w_min"], 0.05,
                                    key="pso_wmin", format="%.2f",
                                    help="Final inertia after linear decay")
            pso_c1      = st.slider("Cognitive coeff. (c1)", 0.5, 3.0, BEST_PSO["c1"], 0.1,
                                    key="pso_c1", format="%.1f",
                                    help="Attraction toward each particle's personal best")
            pso_c2      = st.slider("Social coeff. (c2)", 0.5, 3.0, BEST_PSO["c2"], 0.1,
                                    key="pso_c2", format="%.1f",
                                    help="Attraction toward the global swarm best")

        pso_params = {
            "algo": "PSO", "hidden": pso_hidden, "metric": pso_fitness,
            "n_particles": pso_part, "n_iter": pso_iter,
            "w": pso_w, "w_min": pso_wmin, "c1": pso_c1, "c2": pso_c2, "patience": 15,
        }
        n_params_pso = compute_n_params(pso_hidden)
        st.markdown(
            f'<div class="info-box" style="margin-top:0.5rem">ℹ️ MLP with <strong>{pso_hidden}</strong> hidden neurons '
            f'→ <strong>{n_params_pso}</strong> parameters to optimize</div>',
            unsafe_allow_html=True
        )

        run_pso_btn = st.button("▶ Run PSO", use_container_width=True, type="primary", key="run_pso")

        pso_plot_ph = st.empty()
        pso_prog    = st.progress(0)
        pso_status  = st.empty()

        # Show stored result if exists
        if "last_pso" in st.session_state and not run_pso_btn:
            pd_data = st.session_state["last_pso"]
            fig = _make_convergence_fig(pd_data["history"], "PSO", "#e07b3a", "PSO Convergence")
            pso_plot_ph.plotly_chart(fig, use_container_width=True, key="pso_stored")
            pso_prog.progress(1.0)
            _show_metrics_mini(pd_data["metrics"], "PSO")

        if run_pso_btn:
            t0 = time.perf_counter()
            best_sol, best_fit, history = run_pso_live(
                pso_params, X, y, pso_plot_ph, pso_prog, pso_status
            )
            elapsed = time.perf_counter() - t0
            metrics = full_eval(best_sol, pso_hidden, X, y)
            pso_status.empty()

            st.session_state["last_pso"] = {
                "params": pso_params, "history": history,
                "metrics": metrics, "elapsed": elapsed,
                "best_solution": best_sol,
            }
            st.markdown(f"""
            <div class="success-box">
                ✅ <strong>PSO completed</strong> in {elapsed:.1f}s — Best fitness: <strong>{best_fit:.4f}</strong>
            </div>""", unsafe_allow_html=True)
            if metrics["degenerate"]:
                st.markdown("""
                <div class="warn-box">⚠️ <strong>Degenerate solution</strong> — the model predicted only the positive class.
                Try changing the fitness metric to <em>f1</em>.</div>""", unsafe_allow_html=True)
            _show_metrics_mini(metrics, "PSO")

    # Prompt to go to results
    if "last_ga" in st.session_state and "last_pso" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="success-box">
            🏆 Both GA and PSO results are ready! Head to <strong>📈 Results</strong> for a full comparison.
        </div>""", unsafe_allow_html=True)
        if st.button("📈 View Full Comparison", key="go_results"):
            st.session_state["page"] = "📈 Results"
            st.rerun()


def _show_metrics_mini(metrics, algo):
    """Display 4 metric cards in a 2x2 grid."""
    c1, c2 = st.columns(2)
    color   = "#1a7fc4" if algo == "GA" else "#e07b3a"
    items = [
        ("Accuracy",  f'{metrics["accuracy"]:.1%}',  ""),
        ("Recall",    f'{metrics["recall"]:.1%}',    "green"),
        ("Precision", f'{metrics["precision"]:.1%}', "orange"),
        ("F1 Score",  f'{metrics["f1"]:.1%}',        "purple"),
    ]
    for col, (label, val, cls) in zip([c1, c2, c1, c2], items):
        with col:
            st.markdown(
                f'<div class="metric-card {cls}" style="margin-top:0.5rem">'
                f'<div class="metric-value" style="font-size:1.5rem">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Results
# ─────────────────────────────────────────────────────────────────────────────
def page_results(df):
    X = df.drop(columns=["status"]).values.astype(float)
    y = df["status"].values.astype(int)

    st.markdown('<div class="section-header">📈 Results & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Side-by-side comparison of your GA and PSO runs</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    has_ga  = "last_ga"  in st.session_state
    has_pso = "last_pso" in st.session_state

    if not has_ga and not has_pso:
        st.markdown("""
        <div class="info-box">
            ℹ️ No results yet. Go to <strong>🧬 Experiment</strong> and run GA and/or PSO first.
        </div>""", unsafe_allow_html=True)
        if st.button("🧬 Go to Experiment", key="goto_exp"):
            st.session_state["page"] = "🧬 Experiment"
            st.rerun()
        return

    # ── Top summary cards ─────────────────────────────────────────────────────
    if has_ga and has_pso:
        gd, pd_data = st.session_state["last_ga"], st.session_state["last_pso"]
        st.markdown("#### 🏆 Head-to-Head Summary")
        col_ga, col_vs, col_pso = st.columns([5, 1, 5])

        with col_ga:
            st.markdown(f"""
            <div class="result-block">
                <div class="result-block-title">🧬 Genetic Algorithm</div>
                <div class="result-metric-row">
                    <span class="result-metric-name">F1 Score</span>
                    <span class="result-metric-val">{gd["metrics"]["f1"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Accuracy</span>
                    <span class="result-metric-val">{gd["metrics"]["accuracy"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Recall</span>
                    <span class="result-metric-val">{gd["metrics"]["recall"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Precision</span>
                    <span class="result-metric-val">{gd["metrics"]["precision"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Runtime</span>
                    <span class="result-metric-val">{gd["elapsed"]:.1f}s</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Degenerate</span>
                    <span class="result-metric-val">{"⚠️ Yes" if gd["metrics"]["degenerate"] else "✅ No"}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_vs:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;height:100%;'
                'font-size:1.5rem;font-weight:800;color:#0d3b66;padding-top:3rem">VS</div>',
                unsafe_allow_html=True
            )

        with col_pso:
            st.markdown(f"""
            <div class="result-block pso">
                <div class="result-block-title">🌊 Particle Swarm Optimization</div>
                <div class="result-metric-row">
                    <span class="result-metric-name">F1 Score</span>
                    <span class="result-metric-val">{pd_data["metrics"]["f1"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Accuracy</span>
                    <span class="result-metric-val">{pd_data["metrics"]["accuracy"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Recall</span>
                    <span class="result-metric-val">{pd_data["metrics"]["recall"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Precision</span>
                    <span class="result-metric-val">{pd_data["metrics"]["precision"]:.3f}</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Runtime</span>
                    <span class="result-metric-val">{pd_data["elapsed"]:.1f}s</span>
                </div>
                <div class="result-metric-row">
                    <span class="result-metric-name">Degenerate</span>
                    <span class="result-metric-val">{"⚠️ Yes" if pd_data["metrics"]["degenerate"] else "✅ No"}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Winner badge
        ga_f1  = gd["metrics"]["f1"]
        pso_f1 = pd_data["metrics"]["f1"]
        if ga_f1 > pso_f1:
            winner_text = f"🧬 <strong>GA wins</strong> on F1 Score ({ga_f1:.3f} vs {pso_f1:.3f})"
            winner_color = "#1a7fc4"
        elif pso_f1 > ga_f1:
            winner_text = f"🌊 <strong>PSO wins</strong> on F1 Score ({pso_f1:.3f} vs {ga_f1:.3f})"
            winner_color = "#e07b3a"
        else:
            winner_text = f"🤝 <strong>Tie!</strong> Both achieved F1 = {ga_f1:.3f}"
            winner_color = "#1e8c6e"
        st.markdown(
            f'<div style="background:{winner_color}18;border:2px solid {winner_color};border-radius:10px;'
            f'padding:0.9rem 1.2rem;font-size:0.95rem;color:{winner_color};text-align:center">'
            f'{winner_text}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Convergence chart ─────────────────────────────────────────────────────
    st.markdown("#### 📉 Convergence Curves")
    fig_conv = go.Figure()
    if has_ga:
        gd = st.session_state["last_ga"]
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(gd["history"]) + 1)),
            y=gd["history"],
            name="🧬 GA",
            line=dict(color="#1a7fc4", width=2.5),
            mode="lines",
        ))
    if has_pso:
        pd_data = st.session_state["last_pso"]
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(pd_data["history"]) + 1)),
            y=pd_data["history"],
            name="🌊 PSO",
            line=dict(color="#e07b3a", width=2.5),
            mode="lines",
        ))
    fig_conv.update_layout(
        xaxis_title="Generation / Iteration", yaxis_title="Best Fitness",
        yaxis=dict(range=[0, 1.05]), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=13), margin=dict(l=40, r=20, t=30, b=50),
    )
    fig_conv.update_xaxes(showgrid=True, gridcolor="#e8eef5")
    fig_conv.update_yaxes(showgrid=True, gridcolor="#e8eef5")
    st.plotly_chart(fig_conv, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart + radar chart ───────────────────────────────────────────────
    if has_ga and has_pso:
        col_bar, col_radar = st.columns(2)
        gd, pd_data = st.session_state["last_ga"], st.session_state["last_pso"]
        metric_names = ["Accuracy", "Recall", "Precision", "F1"]
        ga_vals  = [gd["metrics"]["accuracy"], gd["metrics"]["recall"],
                    gd["metrics"]["precision"], gd["metrics"]["f1"]]
        pso_vals = [pd_data["metrics"]["accuracy"], pd_data["metrics"]["recall"],
                    pd_data["metrics"]["precision"], pd_data["metrics"]["f1"]]

        with col_bar:
            st.markdown("**Bar Chart — All Metrics**")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="GA",  x=metric_names, y=ga_vals,  marker_color="#1a7fc4"))
            fig_bar.add_trace(go.Bar(name="PSO", x=metric_names, y=pso_vals, marker_color="#e07b3a"))
            fig_bar.update_layout(
                barmode="group", yaxis=dict(range=[0, 1.1]), height=320,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter", size=12), margin=dict(l=30, r=10, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_bar.update_yaxes(showgrid=True, gridcolor="#e8eef5")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_radar:
            st.markdown("**Radar Chart — Performance Profile**")
            cats = metric_names + [metric_names[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=ga_vals + [ga_vals[0]], theta=cats,
                fill="toself", name="GA",
                line=dict(color="#1a7fc4"), fillcolor="rgba(26,127,196,0.15)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=pso_vals + [pso_vals[0]], theta=cats,
                fill="toself", name="PSO",
                line=dict(color="#e07b3a"), fillcolor="rgba(224,123,58,0.15)",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=320, showlegend=True, font=dict(family="Inter", size=12),
                margin=dict(l=30, r=30, t=20, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrices
        st.markdown("#### 🔲 Confusion Matrices")
        cm_cols = st.columns(2)
        for col, (label, data, color) in zip(cm_cols, [
            ("GA",  gd,      "#1a7fc4"),
            ("PSO", pd_data, "#e07b3a"),
        ]):
            with col:
                cm = data["metrics"]["confusion_matrix"]
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Healthy (0)", "Parkinson (1)"],
                    y=["Healthy (0)", "Parkinson (1)"],
                    color_continuous_scale=[[0, "#f0f7ff"], [1, color]],
                    text_auto=True,
                )
                fig_cm.update_layout(
                    title=f"{label} Confusion Matrix",
                    height=280, margin=dict(l=10, r=10, t=40, b=10),
                    font=dict(family="Inter", size=12),
                    coloraxis_showscale=False,
                )
                fig_cm.update_traces(textfont=dict(size=18, color="white"))
                st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Config comparison ─────────────────────────────────────────────────────
    with st.expander("⚙️ Configuration Used", expanded=False):
        rows = []
        if has_ga:
            p = st.session_state["last_ga"]["params"]
            rows.append({"Algorithm": "🧬 GA", "Hidden": p["hidden"], "Fitness": p["metric"],
                         "Pop/Particles": p["pop_size"], "Gen/Iter": p["n_gen"],
                         "Mut Rate": p.get("mut_rate","—"), "Crossover": p.get("crossover","—"),
                         "w": "—", "c1": "—", "c2": "—"})
        if has_pso:
            p = st.session_state["last_pso"]["params"]
            rows.append({"Algorithm": "🌊 PSO", "Hidden": p["hidden"], "Fitness": p["metric"],
                         "Pop/Particles": p["n_particles"], "Gen/Iter": p["n_iter"],
                         "Mut Rate": "—", "Crossover": "—",
                         "w": p["w"], "c1": p["c1"], "c2": p["c2"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Weight vector analysis ────────────────────────────────────────────────
    with st.expander("🔢 Best Weight Vector θ Analysis", expanded=False):
        for key, label, color in [("last_ga","GA","#1a7fc4"), ("last_pso","PSO","#e07b3a")]:
            if key in st.session_state:
                sol = st.session_state[key]["best_solution"]
                st.markdown(f"**{label} — θ heatmap** ({len(sol)} parameters)")
                fig_w = go.Figure(go.Heatmap(
                    z=[sol], colorscale="RdBu_r", zmid=0, showscale=True,
                ))
                fig_w.update_layout(
                    height=100, margin=dict(l=10,r=10,t=10,b=10),
                    yaxis=dict(showticklabels=False), font=dict(family="Inter"),
                )
                st.plotly_chart(fig_w, use_container_width=True)
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("Min", f"{sol.min():.4f}")
                wc2.metric("Max", f"{sol.max():.4f}")
                wc3.metric("Mean", f"{sol.mean():.4f}")
                wc4.metric("Std", f"{sol.std():.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    if has_ga or has_pso:
        st.markdown("<br>", unsafe_allow_html=True)
        export_rows = []
        for key, label in [("last_ga","GA"), ("last_pso","PSO")]:
            if key in st.session_state:
                d = st.session_state[key]
                m = d["metrics"]
                export_rows.append({
                    "Algorithm": label,
                    **{k: v for k, v in d["params"].items()},
                    "accuracy": round(m["accuracy"],4),
                    "recall": round(m["recall"],4),
                    "precision": round(m["precision"],4),
                    "f1": round(m["f1"],4),
                    "runtime_s": round(d["elapsed"],2),
                    "degenerate": m["degenerate"],
                })
        csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Results (CSV)", csv, "results_ga_pso.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Education
# ─────────────────────────────────────────────────────────────────────────────
def page_education():
    st.markdown('<div class="section-header">📚 Learn More</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Understand the algorithms and the disease</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧬 Genetic Algorithm", "🌊 PSO", "🏥 Parkinson's Disease"])

    with tab1:
        st.markdown("""
        <div class="edu-card">
            <h3>🧬 What is a Genetic Algorithm?</h3>
            <p>
                <strong>Genetic Algorithms (GA)</strong> are optimization methods inspired by
                Darwin's <em>natural selection</em>. They simulate biological evolution to find
                progressively better solutions to an optimization problem.
            </p>
            <p>
                In this project, each "individual" in the population represents a weight vector
                <em>θ</em> for the MLP neural network. The goal is to find the θ that maximizes
                recall (or F1) for Parkinson's detection — without using gradient descent.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🔄 Step-by-Step Process**")
        st.markdown("""
        <div class="step-row">
            <div class="step-item"><div class="step-num">1</div><div class="step-label">Initialize<br>Population</div></div>
            <div class="step-item"><div class="step-num">2</div><div class="step-label">Evaluate<br>Fitness</div></div>
            <div class="step-item"><div class="step-num">3</div><div class="step-label">Selection<br>(Tournament)</div></div>
            <div class="step-item"><div class="step-num">4</div><div class="step-label">Crossover<br>(Arithmetic)</div></div>
            <div class="step-item"><div class="step-num">5</div><div class="step-label">Mutation<br>(Gaussian)</div></div>
            <div class="step-item"><div class="step-num">6</div><div class="step-label">Elitism<br>+ New Gen</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="edu-card">
                <h3>📌 Key Parameters</h3>
                <ul>
                    <li><strong>Population size</strong>: number of individuals per generation</li>
                    <li><strong>Mutation rate</strong>: per-gene mutation probability</li>
                    <li><strong>Crossover method</strong>:
                        <em>Arithmetic</em> (interpolation) or
                        <em>Blend BLX-α</em> (extrapolation)</li>
                    <li><strong>Elitism</strong>: top N individuals survive unchanged</li>
                    <li><strong>Tournament size</strong>: selection pressure</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="edu-card">
                <h3>⚖️ Pros & Cons</h3>
                <ul>
                    <li>✅ Robust on multimodal search landscapes</li>
                    <li>✅ Good diversity through crossover operators</li>
                    <li>✅ No gradient needed (black-box optimization)</li>
                    <li>❌ Can be slower to converge than PSO</li>
                    <li>❌ Many hyperparameters to tune</li>
                    <li>❌ Risk of degenerate solutions with recall metric</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        # Illustrative convergence
        st.markdown("**📉 Typical Convergence (illustrative)**")
        np.random.seed(42)
        gens = list(range(1, 51))
        fake = np.clip([0.4 + 0.5*(1 - np.exp(-g/12)) + np.random.normal(0, 0.01) for g in gens], 0, 1)
        st.plotly_chart(_make_convergence_fig(fake.tolist(), "GA", "#1a7fc4", "GA — Typical Convergence"), use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="edu-card">
            <h3>🌊 What is Particle Swarm Optimization?</h3>
            <p>
                <strong>PSO</strong> is inspired by the collective behavior of bird flocks and
                fish schools. Each "particle" is a candidate solution that moves through the
                weight space guided by its own memory and the collective knowledge of the swarm.
            </p>
            <p>
                The velocity update balances three components: <em>inertia</em> (tendency to
                continue in the same direction), <em>cognitive</em> (attraction to personal best)
                and <em>social</em> (attraction to the swarm's global best).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**⚡ Velocity Update Equation**")
        st.latex(r"""
            v_i \leftarrow \underbrace{w \cdot v_i}_{\text{inertia}}
            + \underbrace{c_1 \cdot r_1 \cdot (p_i - x_i)}_{\text{cognitive}}
            + \underbrace{c_2 \cdot r_2 \cdot (g - x_i)}_{\text{social}}
        """)
        st.latex(r"x_i \leftarrow x_i + v_i")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="edu-card">
                <h3>📌 Key Parameters</h3>
                <ul>
                    <li><strong>w</strong>: inertia weight — controls exploration vs exploitation</li>
                    <li><strong>w_min</strong>: minimum inertia (linear decay)</li>
                    <li><strong>c1</strong>: cognitive coefficient — pull toward personal best</li>
                    <li><strong>c2</strong>: social coefficient — pull toward global best</li>
                    <li><strong>v_max</strong>: velocity clamping to prevent divergence</li>
                    <li><strong>Patience</strong>: early stopping without improvement</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="edu-card">
                <h3>⚖️ GA vs PSO</h3>
                <ul>
                    <li>🔵 GA: recombination of parents → diversity via crossover</li>
                    <li>🟠 PSO: velocity-guided movement → smooth convergence</li>
                    <li>🔵 GA: memory: only current population</li>
                    <li>🟠 PSO: memory: each particle tracks its own best</li>
                    <li>🔵 GA: more robust on rugged landscapes</li>
                    <li>🟠 PSO: faster convergence on smooth landscapes</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        st.markdown("**📉 Typical PSO Convergence (illustrative)**")
        np.random.seed(7)
        fake_pso = np.clip([0.35 + 0.55*(1-np.exp(-g/8)) + np.random.normal(0, 0.008) for g in gens], 0, 1)
        st.plotly_chart(_make_convergence_fig(fake_pso.tolist(), "PSO", "#e07b3a", "PSO — Typical Convergence"), use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="edu-card">
            <h3>🏥 Parkinson's Disease — Overview</h3>
            <p>
                Parkinson's disease is a progressive neurodegenerative disorder primarily
                affecting the motor system. It is caused by the loss of dopaminergic neurons
                in the <em>substantia nigra</em>, reducing dopamine production.
            </p>
            <p>
                Classic motor symptoms include <strong>resting tremor</strong>,
                <strong>muscle rigidity</strong>, <strong>bradykinesia</strong> (slowness of
                movement) and <strong>postural instability</strong>. Non-motor symptoms
                such as anxiety, depression, and voice changes are also common.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <span class="card-icon">📊</span>
                <p class="card-title">Epidemiology</p>
                <p class="card-text">
                    • ~10 million affected worldwide<br>
                    • 2nd most common neurodegenerative disease<br>
                    • Incidence: 8–18 per 100,000/year<br>
                    • Peak diagnosis: 60–70 years old<br>
                    • Slightly more common in men
                </p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <span class="card-icon">🎤</span>
                <p class="card-title">Voice & Diagnosis</p>
                <p class="card-text">
                    • ~90% of patients have vocal changes<br>
                    • <strong>Jitter</strong>: frequency cycle variation<br>
                    • <strong>Shimmer</strong>: amplitude variation<br>
                    • <strong>HNR</strong>: harmonic-to-noise ratio<br>
                    • Detectable before clinical diagnosis
                </p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="card">
                <span class="card-icon">💊</span>
                <p class="card-title">Treatment</p>
                <p class="card-text">
                    • No current cure, but manageable<br>
                    • Levodopa: main pharmacological treatment<br>
                    • Deep Brain Stimulation (DBS)<br>
                    • Physiotherapy and speech therapy<br>
                    • Early diagnosis improves prognosis
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔗 References & External Resources**")
        refs = [
            ("UCI ML Repository — Parkinson's Dataset",
             "https://archive.ics.uci.edu/dataset/174/parkinsons",
             "Max A. Little et al. (2007). Dataset used in this project."),
            ("Parkinson's Foundation",
             "https://www.parkinson.org/",
             "Educational resources, research, and support."),
            ("Little MA et al. (2009) — BioMedical Engineering OnLine",
             "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3051371/",
             "Exploiting nonlinear recurrence and fractal scaling for voice disorder detection."),
            ("Kennedy & Eberhart (1995) — Original PSO Paper",
             "https://ieeexplore.ieee.org/document/488968",
             "Particle swarm optimization. IEEE ICNN'95."),
            ("Holland (1975) — Genetic Algorithms",
             "https://www.amazon.com/Adaptation-Natural-Artificial-Systems-Introductory/dp/0262581116",
             "Adaptation in Natural and Artificial Systems. University of Michigan Press."),
        ]
        for title, url, desc in refs:
            st.markdown(f"""
            <div class="ref-card">
                <strong><a href="{url}" target="_blank">🔗 {title}</a></strong><br>
                <span style="color:#5a7090">{desc}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2.8rem">🧠</div>
            <div style="font-size:1.1rem;font-weight:700;letter-spacing:-0.3px;">Parkinson's Explorer</div>
            <div style="font-size:0.72rem;opacity:0.75;margin-top:3px;">GA & PSO · MLP · Voice Data</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        pages = ["🏠 Home", "📊 Dataset", "🧬 Experiment", "📈 Results", "📚 Education"]
        if "page" not in st.session_state:
            st.session_state["page"] = "🏠 Home"

        selected = st.radio("Navigation", pages,
                            index=pages.index(st.session_state["page"]), key="nav_radio")
        if selected != st.session_state["page"]:
            st.session_state["page"] = selected

        st.markdown("---")
        st.markdown("**📋 Stored Runs**")
        if "last_ga" in st.session_state:
            m = st.session_state["last_ga"]["metrics"]
            st.markdown(f'<span class="algo-badge-ga">GA</span>&nbsp; F1: <strong>{m["f1"]:.3f}</strong> &nbsp; Acc: <strong>{m["accuracy"]:.3f}</strong>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="opacity:0.6;font-size:0.8rem">🧬 GA: not run yet</span>', unsafe_allow_html=True)

        if "last_pso" in st.session_state:
            m = st.session_state["last_pso"]["metrics"]
            st.markdown(f'<span class="algo-badge-pso">PSO</span>&nbsp; F1: <strong>{m["f1"]:.3f}</strong> &nbsp; Acc: <strong>{m["accuracy"]:.3f}</strong>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="opacity:0.6;font-size:0.8rem">🌊 PSO: not run yet</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.71rem;opacity:0.65;text-align:center;line-height:1.6;">
            Optimization Algorithms Project<br>
            NOVA IMS · 2025/26<br>
            GA & PSO for MLP Weight Optimization
        </div>
        """, unsafe_allow_html=True)

    return st.session_state["page"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    df   = load_data()
    page = build_sidebar()

    if page == "🏠 Home":
        page_home(df)
    elif page == "📊 Dataset":
        page_dataset(df)
    elif page == "🧬 Experiment":
        page_experiment(df)
    elif page == "📈 Results":
        page_results(df)
    elif page == "📚 Education":
        page_education()


if __name__ == "__main__":
    main()
