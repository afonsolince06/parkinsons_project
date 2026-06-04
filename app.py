"""
app.py
======
Streamlit Web Application — Parkinson's Disease ML Explorer
Theme: health (blue/green), modern layout with custom CSS.
Pages: Home | Dataset | Experiment (GA + PSO + DE + ABC) | Results | Education

Algorithms: GA, PSO, DE, ABC — all share the same fitness interface:
    fitness_fn(solution, X, y) → F1-score ∈ [0, 1]
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
# Best hyperparameters  (from grid search)
# ─────────────────────────────────────────────────────────────────────────────
BEST_GA = {
    "hidden":    10, "pop_size": 50, "n_gen":  50,
    "mut_rate":  0.05, "crossover": "arithmetic", "metric": "f1",
}
BEST_PSO = {
    "hidden": 10, "n_particles": 50, "n_iter": 50,
    "w": 0.9, "w_min": 0.4, "c1": 2.0, "c2": 2.0, "metric": "f1", "patience": 15,
}
BEST_DE = {
    "hidden": 10, "pop_size": 30, "n_iter": 50,
    "F": 0.8, "CR": 0.9,
}
BEST_ABC = {
    "hidden": 10, "n_bees": 40, "n_iter": 50, "limit_mult": 5,
}

# Colour palette for each algorithm
ALGO_COLORS = {
    "GA":  "#1a7fc4",
    "PSO": "#e07b3a",
    "DE":  "#7752be",
    "ABC": "#1e8c6e",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d3b66 0%, #1a5f9e 60%, #1e8c6e 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

    .hero-full {
        background: linear-gradient(135deg, #0d3b66 0%, #1a7fc4 50%, #1e8c6e 100%);
        border-radius: 18px; padding: 4rem 3rem; color: white;
        position: relative; overflow: hidden;
        box-shadow: 0 10px 40px rgba(13,59,102,0.3); margin-bottom: 0.5rem;
    }
    .hero-full::before { content:''; position:absolute; top:-80px; right:-80px;
        width:350px; height:350px; background:rgba(255,255,255,0.06); border-radius:50%; }
    .hero-full::after { content:''; position:absolute; bottom:-100px; left:-60px;
        width:400px; height:400px; background:rgba(255,255,255,0.04); border-radius:50%; }
    .hero-full-badge {
        display:inline-block; background:rgba(255,255,255,0.18);
        border:1px solid rgba(255,255,255,0.35); border-radius:50px;
        padding:5px 18px; font-size:0.75rem; font-weight:700;
        letter-spacing:1px; text-transform:uppercase; margin-bottom:1.4rem;
    }
    .hero-full-title { font-size:3.2rem; font-weight:800; line-height:1.05; margin:0 0 1rem; letter-spacing:-1px; }
    .hero-full-sub { font-size:1.15rem; opacity:0.9; line-height:1.65; margin:0 0 2rem; max-width:580px; }

    .stat-bar {
        background: linear-gradient(135deg, #0d3b66, #1a7fc4 50%, #1e8c6e);
        border-radius:16px; padding:2rem 1.5rem; color:white;
        display:flex; justify-content:space-around; align-items:center;
        flex-wrap:wrap; gap:1rem;
        box-shadow: 0 6px 24px rgba(13,59,102,0.2); margin:1.5rem 0;
    }
    .stat-bar-item { text-align:center; }
    .stat-bar-value { font-size:2.5rem; font-weight:800; line-height:1; }
    .stat-bar-label { font-size:0.75rem; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px; margin-top:0.3rem; }
    .stat-bar-sep { width:1px; height:50px; background:rgba(255,255,255,0.2); }

    .home-section { padding:3.5rem 0 2.5rem; border-top:1px solid #eaf0f8; }
    .home-section.alt { background:#f7fbff; border-radius:16px; padding:3rem 2rem; margin:1.5rem 0; }
    .section-title-center { text-align:center; font-size:1.9rem; font-weight:700; color:#0d3b66; margin:0 0 0.5rem; }
    .section-sub-center { text-align:center; font-size:1rem; color:#5a7090; margin:0 0 2.5rem; line-height:1.6; }

    .service-icon-wrap {
        width:80px; height:80px;
        background:linear-gradient(135deg, #1a7fc4, #1e8c6e); border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        margin:0 auto 1.1rem; font-size:2rem;
        box-shadow:0 4px 16px rgba(26,127,196,0.25);
    }
    .service-card {
        text-align:center; padding:1.5rem 1rem; background:white;
        border-radius:16px; box-shadow:0 2px 12px rgba(13,59,102,0.07);
        height:100%; transition:transform 0.2s, box-shadow 0.2s;
        border:1px solid rgba(26,127,196,0.08);
    }
    .service-card:hover { transform:translateY(-4px); box-shadow:0 8px 28px rgba(13,59,102,0.13); }
    .service-card-title { font-size:1.1rem; font-weight:700; color:#1a7fc4; margin:0 0 0.6rem; }
    .service-card-text { font-size:0.88rem; color:#5a7090; line-height:1.7; margin:0 0 1rem; }

    .info-block { background:white; border-radius:16px; padding:2.2rem 2rem;
        box-shadow:0 2px 14px rgba(0,0,0,0.07); height:100%; }
    .info-block-label {
        display:inline-block;
        background:linear-gradient(135deg, #1a7fc4, #1e8c6e);
        color:white; border-radius:50px; padding:4px 16px;
        font-size:0.72rem; font-weight:700; letter-spacing:0.5px;
        text-transform:uppercase; margin-bottom:0.9rem;
    }
    .info-block h3 { font-size:1.3rem; font-weight:700; color:#0d3b66; margin:0 0 0.8rem; }
    .info-block p, .info-block li { font-size:0.9rem; color:#4a6080; line-height:1.7; }
    .info-block ul { padding-left:1.2rem; margin:0; }

    .home-footer { background:#0d3b66; color:white; border-radius:16px; padding:2.5rem 2rem; margin-top:2rem; }
    .footer-title { font-size:1.2rem; font-weight:700; margin:0 0 0.3rem; }
    .footer-sub { font-size:0.82rem; opacity:0.7; margin:0 0 1.5rem; line-height:1.6; }
    .footer-links a { color:rgba(255,255,255,0.75); text-decoration:none; font-size:0.82rem; margin-right:1.5rem; line-height:2; }
    .footer-links a:hover { color:white; }
    .footer-bottom { border-top:1px solid rgba(255,255,255,0.12); padding-top:1rem; margin-top:1.5rem; font-size:0.75rem; opacity:0.55; text-align:center; }

    .section-header { font-size:1.55rem; font-weight:700; color:#0d3b66; margin:0 0 0.3rem; }
    .section-sub { font-size:0.9rem; color:#5a7090; margin:0 0 1.2rem; }
    .section-divider { height:3px; background:linear-gradient(90deg, #1a7fc4, #1e8c6e, transparent); border:none; border-radius:2px; margin:0 0 1.5rem; }

    .card { background:white; border-radius:14px; padding:1.5rem;
        box-shadow:0 2px 12px rgba(13,59,102,0.1); border:1px solid rgba(26,127,196,0.1);
        height:100%; transition:transform 0.2s, box-shadow 0.2s; }
    .card:hover { transform:translateY(-3px); box-shadow:0 6px 24px rgba(13,59,102,0.15); }
    .card-icon { font-size:2rem; margin-bottom:0.7rem; display:block; }
    .card-title { font-size:1rem; font-weight:700; color:#0d3b66; margin:0 0 0.4rem; }
    .card-text { font-size:0.87rem; color:#4a6080; line-height:1.6; margin:0; }

    .algo-panel { background:white; border-radius:16px; padding:1.8rem;
        box-shadow:0 2px 16px rgba(0,0,0,0.07); margin-bottom:1.5rem;
        border-left:5px solid #1a7fc4; }
    .algo-panel.pso { border-left-color:#e07b3a; }
    .algo-panel.de  { border-left-color:#7752be; }
    .algo-panel.abc { border-left-color:#1e8c6e; }
    .algo-panel-title { font-size:1.2rem; font-weight:700; color:#0d3b66; margin:0 0 0.3rem; }
    .algo-panel-sub { font-size:0.87rem; color:#5a7090; margin:0 0 1.1rem; line-height:1.6; }

    .metric-card { background:white; border-radius:12px; padding:1.1rem 1.2rem;
        text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.08); border-top:4px solid #1a7fc4; }
    .metric-card.green { border-top-color:#1e8c6e; }
    .metric-card.orange { border-top-color:#e07b3a; }
    .metric-card.purple { border-top-color:#7752be; }
    .metric-value { font-size:1.9rem; font-weight:800; color:#0d3b66; }
    .metric-label { font-size:0.78rem; color:#6b7e96; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-top:0.2rem; }

    .info-box { background:linear-gradient(135deg, #e8f4fd, #e8f5f0);
        border-left:4px solid #1a7fc4; border-radius:0 10px 10px 0;
        padding:0.9rem 1.1rem; margin:0.8rem 0; font-size:0.88rem; color:#1a2433; line-height:1.6; }
    .warn-box { background:#fff8e1; border-left:4px solid #f59e0b;
        border-radius:0 8px 8px 0; padding:0.8rem 1rem; font-size:0.87rem; color:#78450a; margin:0.5rem 0; }
    .success-box { background:#e6f7ee; border-left:4px solid #1e8c6e;
        border-radius:0 8px 8px 0; padding:0.8rem 1rem; font-size:0.87rem; color:#145c47; margin:0.5rem 0; }

    .algo-badge-ga  { display:inline-block; background:#dbeafe; color:#1e40af; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700; }
    .algo-badge-pso { display:inline-block; background:#fce8d5; color:#c2510e; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700; }
    .algo-badge-de  { display:inline-block; background:#ede9fe; color:#5b21b6; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700; }
    .algo-badge-abc { display:inline-block; background:#d1fae5; color:#065f46; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700; }

    .result-block { background:white; border-radius:14px; padding:1.5rem;
        box-shadow:0 2px 14px rgba(0,0,0,0.08); height:100%; border-top:5px solid #1a7fc4; }
    .result-block.pso { border-top-color:#e07b3a; }
    .result-block.de  { border-top-color:#7752be; }
    .result-block.abc { border-top-color:#1e8c6e; }
    .result-block-title { font-size:1.05rem; font-weight:700; color:#0d3b66; margin:0 0 1rem; }
    .result-metric-row { display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid #f0f4f9; }
    .result-metric-name { font-size:0.85rem; color:#5a7090; font-weight:500; }
    .result-metric-val { font-size:1rem; font-weight:700; color:#0d3b66; }

    .edu-card { background:white; border-radius:16px; padding:1.8rem;
        box-shadow:0 2px 16px rgba(0,0,0,0.07); margin-bottom:1.5rem; }
    .edu-card h3 { color:#0d3b66; font-size:1.15rem; font-weight:700; margin-top:0; }
    .edu-card p, .edu-card li { color:#3a4e66; font-size:0.9rem; line-height:1.7; }

    .step-row { display:flex; gap:0.8rem; margin:1rem 0; flex-wrap:wrap; }
    .step-item { flex:1; min-width:100px; background:linear-gradient(135deg,#f0f7ff,#f0faf7);
        border-radius:10px; padding:0.9rem 0.5rem; text-align:center;
        border:1px solid rgba(26,127,196,0.15); }
    .step-num { width:32px; height:32px; background:linear-gradient(135deg,#1a7fc4,#1e8c6e);
        border-radius:50%; display:flex; align-items:center; justify-content:center;
        color:white; font-weight:700; font-size:0.85rem; margin:0 auto 0.5rem; }
    .step-label { font-size:0.75rem; font-weight:600; color:#0d3b66; }

    .ref-card { background:#f8fafd; border-radius:10px; padding:0.9rem 1.1rem;
        margin-bottom:0.7rem; border-left:3px solid #1a7fc4; font-size:0.87rem; color:#2a3a50; }
    .ref-card a { color:#1a7fc4; text-decoration:none; font-weight:500; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers (single-hidden-layer, to keep app.py self-contained)
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


def fitness_f1(solution, X, y, hidden_size=10):
    """
    Shared fitness function for GA, PSO, DE, and ABC.
    Returns F1-score (pos_label=1).

    CLINICAL NOTE: Recall would be the clinical gold standard, but pure recall
    maximisation produces degenerate solutions (all-positive prediction).
    F1-score prevents degeneracy while still strongly rewarding recall.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,), activation="relu",
        solver="sgd", max_iter=1, random_state=42
    )
    clf.fit(np.zeros((2, INPUT_SIZE)), np.array([0, 1]))
    clf.coefs_, clf.intercepts_ = unpack_weights(solution, hidden_size)
    y_pred = clf.predict(X)
    return f1_score(y, y_pred, pos_label=1, zero_division=0)


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


def _conv_fig(history, label, color, title=""):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history)+1)), y=history,
        mode="lines", name=label, line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)",
    ))
    fig.update_layout(
        xaxis_title="Generation / Iteration", yaxis_title="Best F1-Score",
        yaxis=dict(range=[0, 1.05]), height=280,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=12), showlegend=False,
        title=dict(text=title or label, font=dict(size=13, color="#0d3b66"), x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8eef5")
    fig.update_yaxes(showgrid=True, gridcolor="#e8eef5")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Live runners (one per algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ga(p, X, y, plot_ph, prog, status):
    n_params = compute_n_params(p["hidden"])
    np.random.seed(42)
    pop  = np.random.uniform(-1, 1, (p["pop_size"], n_params))
    best_sol, best_fit, history = None, -np.inf, []

    for gen in range(p["n_gen"]):
        fits = np.array([fitness_f1(ind, X, y, p["hidden"]) for ind in pop])
        bi = np.argmax(fits)
        if fits[bi] > best_fit:
            best_fit, best_sol = fits[bi], pop[bi].copy()
        history.append(best_fit)
        prog.progress((gen+1)/p["n_gen"])
        status.markdown(f"<small>Gen **{gen+1}/{p['n_gen']}** · Best F1: **{best_fit:.4f}**</small>", unsafe_allow_html=True)
        if (gen+1) % max(1, p["n_gen"]//15) == 0 or gen == p["n_gen"]-1:
            plot_ph.plotly_chart(_conv_fig(history, "GA", ALGO_COLORS["GA"], "GA — F1 Convergence"), use_container_width=True, key=f"ga_{gen}")

        # Next gen
        elite = [pop[i].copy() for i in np.argsort(fits)[::-1][:2]]
        next_gen = elite[:]
        while len(next_gen) < p["pop_size"]:
            def tourn():
                idx = np.random.choice(p["pop_size"], 3, replace=False)
                return pop[idx[np.argmax(fits[idx])]].copy()
            p1, p2 = tourn(), tourn()
            if np.random.rand() < 0.8:
                a = np.random.rand()
                c1 = a*p1 + (1-a)*p2
                c2 = (1-a)*p1 + a*p2
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in [c1, c2]:
                m = np.random.rand(n_params) < p["mut_rate"]
                c[m] += np.random.normal(0, 0.1, m.sum())
                next_gen.append(c)
                if len(next_gen) >= p["pop_size"]:
                    break
        pop = np.array(next_gen[:p["pop_size"]])

    return best_sol, best_fit, history


def _run_pso(p, X, y, plot_ph, prog, status):
    n_params = compute_n_params(p["hidden"])
    np.random.seed(42)
    pos  = np.random.uniform(-1, 1, (p["n_particles"], n_params))
    vel  = np.random.uniform(-0.4, 0.4, (p["n_particles"], n_params))
    fits = np.array([fitness_f1(pos[i], X, y, p["hidden"]) for i in range(p["n_particles"])])
    pb_pos, pb_fit = pos.copy(), fits.copy()
    gi = np.argmax(fits)
    g_pos, g_fit = pos[gi].copy(), float(fits[gi])
    history, no_imp = [], 0

    for it in range(p["n_iter"]):
        w = p["w"] - (p["w"] - p["w_min"]) * (it / max(p["n_iter"]-1, 1))
        r1, r2 = np.random.rand(p["n_particles"], n_params), np.random.rand(p["n_particles"], n_params)
        vel = np.clip(w*vel + p["c1"]*r1*(pb_pos-pos) + p["c2"]*r2*(g_pos-pos), -0.4, 0.4)
        pos = pos + vel
        fits = np.array([fitness_f1(pos[i], X, y, p["hidden"]) for i in range(p["n_particles"])])
        impr = fits > pb_fit
        pb_pos[impr], pb_fit[impr] = pos[impr].copy(), fits[impr]
        cur_best = float(pb_fit.max())
        if cur_best > g_fit + 1e-6:
            g_fit, g_pos, no_imp = cur_best, pb_pos[np.argmax(pb_fit)].copy(), 0
        else:
            no_imp += 1
        history.append(g_fit)
        prog.progress((it+1)/p["n_iter"])
        status.markdown(f"<small>Iter **{it+1}/{p['n_iter']}** · Best F1: **{g_fit:.4f}** · w={w:.3f}</small>", unsafe_allow_html=True)
        if (it+1) % max(1, p["n_iter"]//15) == 0 or it == p["n_iter"]-1:
            plot_ph.plotly_chart(_conv_fig(history, "PSO", ALGO_COLORS["PSO"], "PSO — F1 Convergence"), use_container_width=True, key=f"pso_{it}")
        if no_imp >= p.get("patience", 15):
            history += [g_fit] * (p["n_iter"] - len(history)); break

    return g_pos, g_fit, history


def _run_de(p, X, y, plot_ph, prog, status):
    n_params = compute_n_params(p["hidden"])
    np.random.seed(42)
    pop   = np.random.uniform(-1, 1, (p["pop_size"], n_params))
    fits  = np.array([fitness_f1(pop[i], X, y, p["hidden"]) for i in range(p["pop_size"])])
    bi    = np.argmax(fits)
    best_sol, best_fit, history = pop[bi].copy(), float(fits[bi]), []

    for gen in range(p["n_iter"]):
        for i in range(p["pop_size"]):
            cands = [j for j in range(p["pop_size"]) if j != i]
            r1, r2, r3 = np.random.choice(cands, 3, replace=False)
            mutant = np.clip(pop[r1] + p["F"] * (pop[r2] - pop[r3]), -1, 1)
            j_rand = np.random.randint(0, n_params)
            mask   = np.random.rand(n_params) < p["CR"]; mask[j_rand] = True
            trial  = np.where(mask, mutant, pop[i])
            tf = fitness_f1(trial, X, y, p["hidden"])
            if tf >= fits[i]:
                pop[i], fits[i] = trial, tf
                if tf > best_fit:
                    best_fit, best_sol = tf, trial.copy()
        history.append(best_fit)
        prog.progress((gen+1)/p["n_iter"])
        status.markdown(f"<small>Gen **{gen+1}/{p['n_iter']}** · Best F1: **{best_fit:.4f}**</small>", unsafe_allow_html=True)
        if (gen+1) % max(1, p["n_iter"]//15) == 0 or gen == p["n_iter"]-1:
            plot_ph.plotly_chart(_conv_fig(history, "DE", ALGO_COLORS["DE"], "DE — F1 Convergence"), use_container_width=True, key=f"de_{gen}")

    return best_sol, best_fit, history


def _run_abc(p, X, y, plot_ph, prog, status):
    n_params   = compute_n_params(p["hidden"])
    np.random.seed(42)
    n_emp      = max(2, p["n_bees"] // 2)
    n_onl      = p["n_bees"] - n_emp
    limit      = n_emp * p["limit_mult"]
    sources    = np.random.uniform(-1, 1, (n_emp, n_params))
    fits       = np.array([fitness_f1(sources[i], X, y, p["hidden"]) for i in range(n_emp)])
    trials     = np.zeros(n_emp, dtype=int)
    bi         = np.argmax(fits)
    best_sol, best_fit, history = sources[bi].copy(), float(fits[bi]), []

    def _search(i):
        k = i
        while k == i: k = np.random.randint(0, n_emp)
        j = np.random.randint(0, n_params)
        v = sources[i].copy()
        v[j] = np.clip(sources[i, j] + np.random.uniform(-1,1)*(sources[i,j]-sources[k,j]), -1, 1)
        return v

    for it in range(p["n_iter"]):
        # Employed
        for i in range(n_emp):
            v = _search(i); fv = fitness_f1(v, X, y, p["hidden"])
            if fv >= fits[i]: sources[i], fits[i], trials[i] = v, fv, 0
            else: trials[i] += 1
            if fits[i] > best_fit: best_fit, best_sol = fits[i], sources[i].copy()
        # Onlooker
        shifted = fits - fits.min() + 1e-8; probs = shifted / shifted.sum()
        for _ in range(n_onl):
            i = int(np.random.choice(n_emp, p=probs))
            v = _search(i); fv = fitness_f1(v, X, y, p["hidden"])
            if fv >= fits[i]: sources[i], fits[i], trials[i] = v, fv, 0
            else: trials[i] += 1
            if fits[i] > best_fit: best_fit, best_sol = fits[i], sources[i].copy()
        # Scout
        for i in range(n_emp):
            if trials[i] >= limit:
                sources[i] = np.random.uniform(-1, 1, n_params)
                fits[i]    = fitness_f1(sources[i], X, y, p["hidden"]); trials[i] = 0
                if fits[i] > best_fit: best_fit, best_sol = fits[i], sources[i].copy()

        history.append(best_fit)
        prog.progress((it+1)/p["n_iter"])
        status.markdown(f"<small>Iter **{it+1}/{p['n_iter']}** · Best F1: **{best_fit:.4f}**</small>", unsafe_allow_html=True)
        if (it+1) % max(1, p["n_iter"]//15) == 0 or it == p["n_iter"]-1:
            plot_ph.plotly_chart(_conv_fig(history, "ABC", ALGO_COLORS["ABC"], "ABC — F1 Convergence"), use_container_width=True, key=f"abc_{it}")

    return best_sol, best_fit, history


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2.8rem">🧠</div>
            <div style="font-size:1.1rem;font-weight:700;letter-spacing:-0.3px;">Parkinson's Explorer</div>
            <div style="font-size:0.72rem;opacity:0.75;margin-top:3px;">GA · PSO · DE · ABC · F1 Fitness</div>
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
        badges = [("last_ga","GA","algo-badge-ga"), ("last_pso","PSO","algo-badge-pso"),
                  ("last_de","DE","algo-badge-de"), ("last_abc","ABC","algo-badge-abc")]
        for key, label, badge_cls in badges:
            if key in st.session_state:
                m = st.session_state[key]["metrics"]
                st.markdown(f'<span class="{badge_cls}">{label}</span>&nbsp; F1: <strong>{m["f1"]:.3f}</strong>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="opacity:0.6;font-size:0.8rem">{label}: not run yet</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.71rem;opacity:0.65;text-align:center;line-height:1.6;">
            Optimization Algorithms Project<br>NOVA IMS · 2025/26<br>
            Fitness: F1-Score (pos_label=1)
        </div>
        """, unsafe_allow_html=True)

    return st.session_state["page"]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home(df):
    n_samples = len(df); n_features = df.shape[1]-1
    n_pd = int((df["status"]==1).sum()); n_healthy = int((df["status"]==0).sum())

    col_hero, _ = st.columns([3, 1])
    with col_hero:
        st.markdown("""
        <div class="hero-full">
            <div class="hero-full-badge">🧠 Optimization Algorithms Project · NOVA IMS 2025/26</div>
            <div class="hero-full-title">Parkinson's Disease<br>ML Explorer</div>
            <div class="hero-full-sub">
                Discover how <strong>Genetic Algorithms</strong>, <strong>PSO</strong>,
                <strong>Differential Evolution</strong> and <strong>Artificial Bee Colony</strong>
                train a neural network to detect Parkinson's Disease from voice measurements —
                all optimising the same <strong>F1-score fitness function</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns([2,2,2,3])
    with b1:
        if st.button("ℹ️ Learn More", use_container_width=True, key="hero_learn"):
            st.session_state["page"] = "📚 Education"; st.rerun()
    with b2:
        if st.button("📊 Explore Data", use_container_width=True, key="hero_data"):
            st.session_state["page"] = "📊 Dataset"; st.rerun()
    with b3:
        if st.button("▶ Try It Now", use_container_width=True, type="primary", key="hero_try"):
            st.session_state["page"] = "🧬 Experiment"; st.rerun()

    st.markdown(f"""
    <div class="stat-bar">
        <div class="stat-bar-item"><div class="stat-bar-value">{n_samples}</div><div class="stat-bar-label">Voice Recordings</div></div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item"><div class="stat-bar-value">{n_features}</div><div class="stat-bar-label">Acoustic Features</div></div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item"><div class="stat-bar-value">{n_pd}</div><div class="stat-bar-label">Parkinson's Cases</div></div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item"><div class="stat-bar-value">{n_healthy}</div><div class="stat-bar-label">Healthy Controls</div></div>
        <div class="stat-bar-sep"></div>
        <div class="stat-bar-item"><div class="stat-bar-value">4</div><div class="stat-bar-label">Algorithms</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-section">
        <div class="section-title-center">Understanding Parkinson's Disease</div>
        <div class="section-sub-center">A neurodegenerative disorder affecting millions worldwide —<br>detectable through the voice before clinical diagnosis.</div>
    </div>""", unsafe_allow_html=True)

    for col, (icon, title, text) in zip(st.columns(3), [
        ("🌍","Global Epidemiology","More than <strong>10 million people</strong> worldwide live with Parkinson's — the 2nd most common neurodegenerative disease. Affects ~1% of people over 60 years old."),
        ("🔬","Symptoms & Diagnosis","Classic symptoms include <strong>resting tremor</strong>, muscle rigidity, bradykinesia, and postural instability. Voice alterations are among the earliest detectable signs."),
        ("🎤","Voice-Based Detection","Acoustic features like <strong>jitter, shimmer and HNR</strong> capture subtle changes in vocal production. This dataset uses 22 features from sustained phonation recordings."),
    ]):
        with col:
            st.markdown(f'<div class="service-card"><div class="service-icon-wrap">{icon}</div><div class="service-card-title">{title}</div><p class="service-card-text">{text}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="home-section">
        <div class="section-title-center">Four Algorithms, One Fitness: F1-Score</div>
        <div class="section-sub-center">All algorithms share the same objective function: maximise the F1-score of an MLP neural network on the Parkinson's dataset.</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("""
        <div class="info-block">
            <div class="info-block-label">🧬 GA &nbsp;|&nbsp; 🌊 PSO</div>
            <h3>Evolutionary & Swarm Intelligence</h3>
            <p><strong>GA</strong>: A population evolves via selection, crossover, and mutation. Fittest individuals survive.</p>
            <p><strong>PSO</strong>: Particles fly through weight space guided by personal and global best positions. Inertia decays linearly.</p>
            <ul>
                <li>Both use population-based search</li>
                <li>Both require no gradient information</li>
                <li>Both optimise the same F1 fitness function</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="info-block">
            <div class="info-block-label">🔮 DE &nbsp;|&nbsp; 🐝 ABC</div>
            <h3>Differential Evolution & Bee Colony</h3>
            <p><strong>DE</strong>: Creates mutant vectors from random population triplets (rand/1/bin). Greedy selection keeps the best.</p>
            <p><strong>ABC</strong>: Employed, onlooker, and scout bees cooperate to exploit and explore food sources (solutions).</p>
            <ul>
                <li>DE: very effective on continuous high-dimensional problems</li>
                <li>ABC: division of labour for balanced exploration</li>
                <li>Both compatible with the shared F1 fitness interface</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="home-section alt">
        <div class="section-title-center">Explore the Project</div>
        <div class="section-sub-center">From raw data to live algorithm execution — all in one app.</div>
    </div>""", unsafe_allow_html=True)

    for col, (icon, title, text, pg, key) in zip(st.columns(3), [
        ("📊","Explore Dataset","Visualize 22 acoustic features, class distributions, scatter plots, and correlation heatmaps.","📊 Dataset","btn_ds"),
        ("🧬","Run All 4 Algorithms","Configure GA, PSO, DE, and ABC independently and watch F1 convergence curves in real time!","🧬 Experiment","btn_algo"),
        ("📈","Compare Results","After running algorithms, compare F1 convergence, confusion matrices, and metric tables side-by-side.","📈 Results","btn_res"),
    ]):
        with col:
            st.markdown(f'<div class="service-card"><div class="service-icon-wrap">{icon}</div><div class="service-card-title">{title}</div><p class="service-card-text">{text}</p></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Go to {title} →", use_container_width=True, key=key):
                st.session_state["page"] = pg; st.rerun()

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
                <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem">Algorithms</div>
                <div style="font-size:0.8rem;opacity:0.7;line-height:1.9">
                    🧬 Genetic Algorithm (GA)<br>
                    🌊 Particle Swarm Optimization (PSO)<br>
                    🔮 Differential Evolution (DE)<br>
                    🐝 Artificial Bee Colony (ABC)
                </div>
            </div>
            <div style="flex:1;min-width:200px">
                <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem">Dataset</div>
                <div style="font-size:0.8rem;opacity:0.7;line-height:1.7">
                    Max A. Little et al. (2007)<br>UCI ML Repository<br>195 samples · 22 features<br>Fitness: F1-score (pos_label=1)
                </div>
            </div>
        </div>
        <div class="footer-bottom">© 2025/26 NOVA IMS · Optimization Algorithms Project · Built with Streamlit & Plotly</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────
def page_dataset(df):
    st.markdown('<div class="section-header">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Explore the acoustic voice features used for Parkinson\'s detection</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    X_cols = [c for c in df.columns if c != "status"]
    n_samples = len(df); n_features = len(X_cols)
    n_pd = int((df["status"]==1).sum()); n_healthy = int((df["status"]==0).sum())

    for col, (val, lbl, cls) in zip(st.columns(4), [
        (n_samples,"Total Samples",""), (n_features,"Features","green"),
        (f"{n_pd} ({n_pd/n_samples:.0%})","Parkinson (1)","orange"),
        (f"{n_healthy} ({n_healthy/n_samples:.0%})","Healthy (0)","purple"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Class Distribution & Feature Histograms")
    col_pie, col_hist = st.columns([1, 2])

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Parkinson","Healthy"], values=[n_pd, n_healthy], hole=0.48,
            marker=dict(colors=["#1a7fc4","#1e8c6e"], line=dict(color="white", width=2)),
            textinfo="percent+label", textposition="outside", pull=[0.03, 0],
        ))
        fig_pie.update_layout(height=320, margin=dict(l=20,r=20,t=30,b=20),
            showlegend=False, font=dict(family="Inter",size=13),
            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_hist:
        sel = st.selectbox("Select a feature:", X_cols, index=0, key="feat_hist")
        df_p = df[[sel,"status"]].copy(); df_p["Class"] = df_p["status"].map({1:"Parkinson",0:"Healthy"})
        fig_h = px.histogram(df_p, x=sel, color="Class", barmode="overlay",
            color_discrete_map={"Parkinson":"#1a7fc4","Healthy":"#1e8c6e"}, nbins=30, opacity=0.72)
        fig_h.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
            plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Inter",size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_h.update_xaxes(showgrid=True, gridcolor="#e8eef5")
        fig_h.update_yaxes(showgrid=True, gridcolor="#e8eef5")
        if sel in FEATURE_DESCRIPTIONS:
            desc, group, unit = FEATURE_DESCRIPTIONS[sel]
            g_label, g_color = FEATURE_GROUPS[group]
            st.markdown(
                f'<span style="background:{g_color}22;color:{g_color};border-radius:5px;padding:2px 8px;font-size:0.78rem;font-weight:600">{g_label}</span>'
                f' &nbsp;<span style="font-size:0.87rem;color:#4a6080">{desc}{"  ("+unit+")" if unit else ""}</span>',
                unsafe_allow_html=True)
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Scatter Plot — Compare Two Features")
    sc1, sc2 = st.columns(2)
    with sc1: fx = st.selectbox("Feature X:", X_cols, index=0, key="feat_x")
    with sc2: fy = st.selectbox("Feature Y:", X_cols, index=1, key="feat_y")
    df_s = df[[fx,fy,"status"]].copy(); df_s["Class"] = df_s["status"].map({1:"Parkinson",0:"Healthy"})
    fig_sc = px.scatter(df_s, x=fx, y=fy, color="Class",
        color_discrete_map={"Parkinson":"#1a7fc4","Healthy":"#1e8c6e"}, opacity=0.72)
    fig_sc.update_traces(marker=dict(size=7))
    fig_sc.update_layout(height=380, margin=dict(l=40,r=20,t=20,b=40),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Inter",size=12))
    fig_sc.update_xaxes(showgrid=True, gridcolor="#e8eef5")
    fig_sc.update_yaxes(showgrid=True, gridcolor="#e8eef5")
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Box Plots — Feature Distribution by Class")
    n_show = st.slider("Number of features:", 4, len(X_cols), 8, 2, key="n_box")
    df_melt = df[X_cols[:n_show]+["status"]].copy()
    df_melt["Class"] = df_melt["status"].map({1:"Parkinson",0:"Healthy"})
    df_long = df_melt.melt(id_vars=["Class"], value_vars=X_cols[:n_show], var_name="Feature", value_name="Value")
    fig_box = px.box(df_long, x="Feature", y="Value", color="Class",
        color_discrete_map={"Parkinson":"#1a7fc4","Healthy":"#1e8c6e"}, points=False)
    fig_box.update_layout(height=400, margin=dict(l=40,r=20,t=20,b=80),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Inter",size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02), xaxis_tickangle=-30)
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("📖 Feature Descriptions — All 22 Variables"):
        for group_key, (group_label, group_color) in FEATURE_GROUPS.items():
            st.markdown(f'<div style="background:{group_color}18;border-left:4px solid {group_color};padding:0.5rem 1rem;border-radius:0 8px 8px 0;margin:0.8rem 0 0.4rem;"><strong style="color:{group_color}">{group_label}</strong></div>', unsafe_allow_html=True)
            rows = [{"Feature":f,"Description":d,"Unit":u or "—",
                     "PD Mean":f"{df.loc[df.status==1,f].mean():.4f}",
                     "Healthy Mean":f"{df.loc[df.status==0,f].mean():.4f}"}
                    for f,(d,g,u) in FEATURE_DESCRIPTIONS.items() if g==group_key and f in df.columns]
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("🔥 Correlation Heatmap"):
        corr = df[X_cols].corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=550, font=dict(family="Inter",size=10), margin=dict(l=60,r=20,t=20,b=60))
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📋 Full Dataset"):
        df_disp = df.copy(); df_disp["status"] = df_disp["status"].map({1:"🔵 Parkinson",0:"🟢 Healthy"})
        st.dataframe(df_disp, use_container_width=True, height=400)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "parkinsons_data.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: mini metric cards
# ─────────────────────────────────────────────────────────────────────────────
def _show_metrics_mini(metrics, algo):
    c1, c2 = st.columns(2)
    for col, (label, val, cls) in zip([c1,c2,c1,c2], [
        ("F1 Score", f'{metrics["f1"]:.1%}', ""),
        ("Recall",   f'{metrics["recall"]:.1%}', "green"),
        ("Precision",f'{metrics["precision"]:.1%}', "orange"),
        ("Accuracy", f'{metrics["accuracy"]:.1%}', "purple"),
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card {cls}" style="margin-top:0.5rem">'
                f'<div class="metric-value" style="font-size:1.4rem">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True)


def _algo_panel(algo, color, css_class, desc, params_fn, run_fn, store_key, X, y):
    """Render one algorithm panel: description → params → run button → live plot."""
    st.markdown(f'<div class="algo-panel {css_class}"><div class="algo-panel-title">{algo}</div><div class="algo-panel-sub">{desc}</div></div>', unsafe_allow_html=True)
    params = params_fn()
    n_p = compute_n_params(params["hidden"])
    st.markdown(f'<div class="info-box" style="margin-top:0">ℹ️ MLP with <strong>{params["hidden"]}</strong> hidden neurons → <strong>{n_p}</strong> parameters · Fitness: <strong>F1-score</strong></div>', unsafe_allow_html=True)
    run_btn = st.button(f"▶ Run {algo.split()[0]}", use_container_width=True, type="primary", key=f"run_{store_key}")
    plot_ph = st.empty(); prog = st.progress(0); status = st.empty()

    if store_key in st.session_state and not run_btn:
        d = st.session_state[store_key]
        plot_ph.plotly_chart(_conv_fig(d["history"], algo.split()[0], color, f"{algo.split()[0]} — F1 Convergence"), use_container_width=True, key=f"{store_key}_stored")
        prog.progress(1.0); _show_metrics_mini(d["metrics"], store_key)

    if run_btn:
        t0 = time.perf_counter()
        best_sol, best_fit, history = run_fn(params, X, y, plot_ph, prog, status)
        elapsed = time.perf_counter() - t0
        metrics = full_eval(best_sol, params["hidden"], X, y)
        status.empty()
        st.session_state[store_key] = {"params": params, "history": history,
                                       "metrics": metrics, "elapsed": elapsed,
                                       "best_solution": best_sol}
        st.markdown(f'<div class="success-box">✅ <strong>{algo.split()[0]} completed</strong> in {elapsed:.1f}s — Best F1: <strong>{best_fit:.4f}</strong></div>', unsafe_allow_html=True)
        if metrics["degenerate"]:
            st.markdown('<div class="warn-box">⚠️ <strong>Degenerate solution</strong> — model predicts only positives. Try F1 fitness (default).</div>', unsafe_allow_html=True)
        _show_metrics_mini(metrics, store_key)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Experiment
# ─────────────────────────────────────────────────────────────────────────────
def page_experiment(df):
    X = df.drop(columns=["status"]).values.astype(float)
    y = df["status"].values.astype(int)

    st.markdown('<div class="section-header">🧬 Run Algorithms</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Configure and run GA, PSO, DE, and ABC independently. All optimise the same F1-score fitness function.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        🎯 <strong>Shared Fitness Interface:</strong>
        All four algorithms call <code>fitness_f1(θ, X, y)</code> which evaluates the MLP forward pass
        with weight vector <em>θ</em> and returns the F1-score.
        No backpropagation is used — weights are optimised directly by each algorithm.
        <br><em>Clinical note: Recall is clinically preferred, but F1 prevents degenerate all-positive predictions.</em>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: GA | PSO ───────────────────────────────────────────────────────
    col_ga, col_pso = st.columns(2, gap="large")

    with col_ga:
        def ga_params():
            with st.expander("⚙️ GA Parameters", expanded=True):
                h  = st.slider("Hidden Neurons", 5, 30, BEST_GA["hidden"], 1, key="ga_h")
                ps = st.slider("Population Size", 10, 100, BEST_GA["pop_size"], 5, key="ga_ps",
                               help="Number of individuals per generation")
                ng = st.slider("Generations", 10, 200, BEST_GA["n_gen"], 5, key="ga_ng")
                mr = st.slider("Mutation Rate", 0.01, 0.30, BEST_GA["mut_rate"], 0.01,
                               format="%.2f", key="ga_mr", help="Per-gene Gaussian mutation probability")
                cx = st.selectbox("Crossover", ["arithmetic","blend"],
                                  index=["arithmetic","blend"].index(BEST_GA["crossover"]), key="ga_cx",
                                  help="Arithmetic: linear interpolation | Blend: extrapolation (BLX-α)")
            return {"hidden":h,"pop_size":ps,"n_gen":ng,"mut_rate":mr,"crossover":cx}

        _algo_panel(
            "🧬 Genetic Algorithm (GA)", ALGO_COLORS["GA"], "",
            "A population evolves via <strong>tournament selection</strong>, "
            "<strong>arithmetic crossover</strong>, and <strong>Gaussian mutation</strong>. "
            "Elitism preserves top solutions each generation.",
            ga_params, _run_ga, "last_ga", X, y)

    with col_pso:
        def pso_params():
            with st.expander("⚙️ PSO Parameters", expanded=True):
                h   = st.slider("Hidden Neurons", 5, 30, BEST_PSO["hidden"], 1, key="pso_h")
                np_ = st.slider("Particles", 10, 100, BEST_PSO["n_particles"], 5, key="pso_np",
                                help="Swarm size — more particles = better exploration")
                ni  = st.slider("Iterations", 10, 200, BEST_PSO["n_iter"], 5, key="pso_ni")
                w   = st.slider("Inertia w", 0.4, 1.0, BEST_PSO["w"], 0.05, format="%.2f", key="pso_w",
                                help="Starting inertia weight (decays linearly to w_min)")
                wm  = st.slider("Min inertia w_min", 0.1, 0.6, BEST_PSO["w_min"], 0.05, format="%.2f", key="pso_wm")
                c1  = st.slider("Cognitive c1", 0.5, 3.0, BEST_PSO["c1"], 0.1, format="%.1f", key="pso_c1",
                                help="Attraction toward each particle's personal best position")
                c2  = st.slider("Social c2", 0.5, 3.0, BEST_PSO["c2"], 0.1, format="%.1f", key="pso_c2",
                                help="Attraction toward the global swarm best position")
            return {"hidden":h,"n_particles":np_,"n_iter":ni,"w":w,"w_min":wm,"c1":c1,"c2":c2,"patience":15}

        _algo_panel(
            "🌊 Particle Swarm Optimization (PSO)", ALGO_COLORS["PSO"], "pso",
            "Particles fly through weight space guided by <strong>cognitive</strong> "
            "(personal best) and <strong>social</strong> (global best) forces. "
            "Inertia weight decays linearly from w to w_min.",
            pso_params, _run_pso, "last_pso", X, y)

    st.markdown("<br><hr style='border-color:#eaf0f8'><br>", unsafe_allow_html=True)

    # ── Row 2: DE | ABC ───────────────────────────────────────────────────────
    col_de, col_abc = st.columns(2, gap="large")

    with col_de:
        def de_params():
            with st.expander("⚙️ DE Parameters", expanded=True):
                h  = st.slider("Hidden Neurons", 5, 30, BEST_DE["hidden"], 1, key="de_h")
                ps = st.slider("Population Size", 10, 100, BEST_DE["pop_size"], 5, key="de_ps",
                               help="Number of candidate solutions (minimum 4 for rand/1/bin)")
                ni = st.slider("Iterations", 10, 200, BEST_DE["n_iter"], 5, key="de_ni")
                F  = st.slider("Mutation Factor F", 0.1, 2.0, BEST_DE["F"], 0.05,
                               format="%.2f", key="de_F",
                               help="Magnitude of the difference mutation: v = x_r1 + F*(x_r2 − x_r3). Higher F → more exploration.")
                CR = st.slider("Crossover Rate CR", 0.1, 1.0, BEST_DE["CR"], 0.05,
                               format="%.2f", key="de_CR",
                               help="Probability of inheriting from mutant vector. Higher CR → more exploitation of mutant.")
            return {"hidden":h,"pop_size":ps,"n_iter":ni,"F":F,"CR":CR}

        _algo_panel(
            "🔮 Differential Evolution (DE)", ALGO_COLORS["DE"], "de",
            "<strong>rand/1/bin strategy</strong>: mutant = x_r1 + F·(x_r2 − x_r3). "
            "Binomial crossover creates a trial vector. "
            "Greedy selection keeps the better of trial vs. original.",
            de_params, _run_de, "last_de", X, y)

    with col_abc:
        def abc_params():
            with st.expander("⚙️ ABC Parameters", expanded=True):
                h    = st.slider("Hidden Neurons", 5, 30, BEST_ABC["hidden"], 1, key="abc_h")
                nb   = st.slider("Colony Size (n_bees)", 10, 100, BEST_ABC["n_bees"], 2, key="abc_nb",
                                 help="Total bees: n_employed = n_onlooker = n_bees ÷ 2")
                ni   = st.slider("Iterations", 10, 200, BEST_ABC["n_iter"], 5, key="abc_ni")
                lm   = st.slider("Limit multiplier", 1, 20, BEST_ABC["limit_mult"], 1, key="abc_lm",
                                 help="Abandonment limit = n_employed × multiplier. Higher → more exploitation before scouting.")
            return {"hidden":h,"n_bees":nb,"n_iter":ni,"limit_mult":lm}

        _algo_panel(
            "🐝 Artificial Bee Colony (ABC)", ALGO_COLORS["ABC"], "abc",
            "<strong>Employed bees</strong> exploit known sources; "
            "<strong>onlooker bees</strong> select sources proportional to fitness; "
            "<strong>scout bees</strong> replace exhausted sources with random exploration.",
            abc_params, _run_abc, "last_abc", X, y)

    # Compare prompt
    done = [k for k in ["last_ga","last_pso","last_de","last_abc"] if k in st.session_state]
    if len(done) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="success-box">🏆 {len(done)} algorithm(s) completed. Head to <strong>📈 Results</strong> for full comparison.</div>', unsafe_allow_html=True)
        if st.button("📈 View Full Comparison →", key="go_results"):
            st.session_state["page"] = "📈 Results"; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Results
# ─────────────────────────────────────────────────────────────────────────────
def page_results(df):
    X = df.drop(columns=["status"]).values.astype(float)
    y = df["status"].values.astype(int)

    st.markdown('<div class="section-header">📈 Results & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Side-by-side F1 convergence and metric comparison for all algorithms run</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    ALGO_KEYS = [
        ("last_ga",  "🧬 GA",  "GA",  ""),
        ("last_pso", "🌊 PSO", "PSO", "pso"),
        ("last_de",  "🔮 DE",  "DE",  "de"),
        ("last_abc", "🐝 ABC", "ABC", "abc"),
    ]
    available = [(k,ll,s,c) for k,ll,s,c in ALGO_KEYS if k in st.session_state]

    if not available:
        st.markdown('<div class="info-box">ℹ️ No results yet. Go to <strong>🧬 Experiment</strong> and run at least one algorithm.</div>', unsafe_allow_html=True)
        if st.button("🧬 Go to Experiment"):
            st.session_state["page"] = "🧬 Experiment"; st.rerun()
        return

    # ── Summary result-block cards ────────────────────────────────────────────
    if len(available) > 1:
        st.markdown("#### 🏆 Summary")
        cols = st.columns(len(available))
        for col, (key, long_label, short, css_cls) in zip(cols, available):
            d = st.session_state[key]; m = d["metrics"]
            with col:
                st.markdown(f"""
                <div class="result-block {css_cls}">
                    <div class="result-block-title">{long_label}</div>
                    <div class="result-metric-row"><span class="result-metric-name">F1 Score</span><span class="result-metric-val">{m["f1"]:.3f}</span></div>
                    <div class="result-metric-row"><span class="result-metric-name">Accuracy</span><span class="result-metric-val">{m["accuracy"]:.3f}</span></div>
                    <div class="result-metric-row"><span class="result-metric-name">Recall</span><span class="result-metric-val">{m["recall"]:.3f}</span></div>
                    <div class="result-metric-row"><span class="result-metric-name">Precision</span><span class="result-metric-val">{m["precision"]:.3f}</span></div>
                    <div class="result-metric-row"><span class="result-metric-name">Runtime</span><span class="result-metric-val">{d["elapsed"]:.1f}s</span></div>
                    <div class="result-metric-row"><span class="result-metric-name">Degenerate</span><span class="result-metric-val">{"⚠️ Yes" if m["degenerate"] else "✅ No"}</span></div>
                </div>""", unsafe_allow_html=True)

        # Winner badge
        best_key, best_ll, _, _ = max(available, key=lambda t: st.session_state[t[0]]["metrics"]["f1"])
        best_f1 = st.session_state[best_key]["metrics"]["f1"]
        color = ALGO_COLORS[best_ll.split()[-1]]
        st.markdown(f'<div style="background:{color}18;border:2px solid {color};border-radius:10px;padding:0.9rem 1.2rem;font-size:0.95rem;color:{color};text-align:center;margin-top:1rem">{best_ll} <strong>wins</strong> on F1-Score with <strong>{best_f1:.3f}</strong></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Convergence overlay ───────────────────────────────────────────────────
    st.markdown("#### 📉 F1 Convergence Curves")
    fig_conv = go.Figure()
    for key, long_label, short, _ in available:
        hist = st.session_state[key]["history"]
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(hist)+1)), y=hist,
            name=long_label, mode="lines",
            line=dict(color=ALGO_COLORS[short], width=2.5),
        ))
    fig_conv.update_layout(
        xaxis_title="Generation / Iteration", yaxis_title="Best F1-Score",
        yaxis=dict(range=[0, 1.05]), height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=13), margin=dict(l=40,r=20,t=30,b=50),
    )
    fig_conv.update_xaxes(showgrid=True, gridcolor="#e8eef5")
    fig_conv.update_yaxes(showgrid=True, gridcolor="#e8eef5")
    st.plotly_chart(fig_conv, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar + Radar ───────────────────────────────────────────────────────────
    if len(available) >= 2:
        col_bar, col_radar = st.columns(2)
        metric_names = ["F1","Accuracy","Recall","Precision"]

        with col_bar:
            st.markdown("**Bar Chart — All Metrics**")
            fig_bar = go.Figure()
            for key, long_label, short, _ in available:
                m = st.session_state[key]["metrics"]
                fig_bar.add_trace(go.Bar(
                    name=long_label,
                    x=metric_names,
                    y=[m["f1"],m["accuracy"],m["recall"],m["precision"]],
                    marker_color=ALGO_COLORS[short],
                ))
            fig_bar.update_layout(barmode="group", yaxis=dict(range=[0,1.1]), height=320,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter",size=12), margin=dict(l=30,r=10,t=20,b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig_bar.update_yaxes(showgrid=True, gridcolor="#e8eef5")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_radar:
            st.markdown("**Radar Chart — Performance Profile**")
            cats = metric_names + [metric_names[0]]
            fig_radar = go.Figure()
            for key, long_label, short, _ in available:
                m = st.session_state[key]["metrics"]
                vals = [m["f1"],m["accuracy"],m["recall"],m["precision"]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats, fill="toself", name=long_label,
                    line=dict(color=ALGO_COLORS[short]),
                    fillcolor=f"{ALGO_COLORS[short]}26",
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=320, showlegend=True,
                font=dict(family="Inter",size=12), margin=dict(l=30,r=30,t=20,b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("<br>")
    st.markdown("#### 🔲 Confusion Matrices")
    cm_cols = st.columns(len(available))
    for col, (key, long_label, short, _) in zip(cm_cols, available):
        with col:
            cm = st.session_state[key]["metrics"]["confusion_matrix"]
            fig_cm = px.imshow(cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Healthy","Parkinson"], y=["Healthy","Parkinson"],
                color_continuous_scale=[[0,"#f0f7ff"],[1,ALGO_COLORS[short]]],
                text_auto=True)
            fig_cm.update_layout(title=long_label, height=260,
                margin=dict(l=10,r=10,t=40,b=10),
                font=dict(family="Inter",size=12), coloraxis_showscale=False)
            fig_cm.update_traces(textfont=dict(size=16, color="white"))
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("<br>")
    with st.expander("📋 Full Metrics Table", expanded=True):
        rows = []
        for key, long_label, short, _ in available:
            d = st.session_state[key]; m = d["metrics"]
            rows.append({
                "Algorithm": long_label,
                "F1 Score":  f'{m["f1"]:.4f}',
                "Accuracy":  f'{m["accuracy"]:.4f}',
                "Recall":    f'{m["recall"]:.4f}',
                "Precision": f'{m["precision"]:.4f}',
                "Degenerate": "⚠️ Yes" if m["degenerate"] else "✅ No",
                "Runtime (s)": f'{d["elapsed"]:.1f}',
                "Hidden":    d["params"].get("hidden","—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("⚙️ Configurations Used"):
        cfg_rows = []
        for key, long_label, short, _ in available:
            p = st.session_state[key]["params"]
            cfg_rows.append({"Algorithm": long_label, **{k:v for k,v in p.items()}})
        st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True, hide_index=True)

    with st.expander("🔢 Best Weight Vector θ Analysis"):
        for key, long_label, short, _ in available:
            sol = st.session_state[key]["best_solution"]
            st.markdown(f"**{long_label}** — θ heatmap ({len(sol)} parameters)")
            fig_w = go.Figure(go.Heatmap(z=[sol], colorscale="RdBu_r", zmid=0, showscale=True))
            fig_w.update_layout(height=90, margin=dict(l=10,r=10,t=5,b=10),
                yaxis=dict(showticklabels=False), font=dict(family="Inter"))
            st.plotly_chart(fig_w, use_container_width=True)
            wc1,wc2,wc3,wc4 = st.columns(4)
            wc1.metric("Min", f"{sol.min():.4f}"); wc2.metric("Max", f"{sol.max():.4f}")
            wc3.metric("Mean", f"{sol.mean():.4f}"); wc4.metric("Std", f"{sol.std():.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<br>")
    export_rows = []
    for key, long_label, short, _ in available:
        d = st.session_state[key]; m = d["metrics"]
        export_rows.append({
            "Algorithm": short, **{k:v for k,v in d["params"].items()},
            "f1":round(m["f1"],4), "accuracy":round(m["accuracy"],4),
            "recall":round(m["recall"],4), "precision":round(m["precision"],4),
            "runtime_s":round(d["elapsed"],2), "degenerate":m["degenerate"],
        })
    csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Results (CSV)", csv, "results_all_algos.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Education
# ─────────────────────────────────────────────────────────────────────────────
def page_education():
    st.markdown('<div class="section-header">📚 Learn More</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Understand the algorithms, the fitness function, and the disease</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🧬 GA", "🌊 PSO", "🔮 DE & 🐝 ABC", "🏥 Parkinson's"])

    with tab1:
        st.markdown("""
        <div class="edu-card">
            <h3>🧬 Genetic Algorithm (GA)</h3>
            <p>Inspired by Darwin's <em>natural selection</em>, GA evolves a <strong>population</strong> of
            candidate weight vectors θ over generations. The fittest individuals survive, reproduce, and pass
            their "genetic material" to the next generation through crossover and mutation.</p>
            <p><strong>Fitness function:</strong> F1-score of the MLP on the Parkinson's dataset.
            <em>Clinical note: Recall is preferred in practice, but F1 prevents degenerate all-positive solutions.</em></p>
        </div>""", unsafe_allow_html=True)

        st.markdown("**🔄 Step-by-Step Process**")
        st.markdown("""<div class="step-row">
            <div class="step-item"><div class="step-num">1</div><div class="step-label">Init Population</div></div>
            <div class="step-item"><div class="step-num">2</div><div class="step-label">Evaluate F1</div></div>
            <div class="step-item"><div class="step-num">3</div><div class="step-label">Tournament Select</div></div>
            <div class="step-item"><div class="step-num">4</div><div class="step-label">Crossover</div></div>
            <div class="step-item"><div class="step-num">5</div><div class="step-label">Mutation</div></div>
            <div class="step-item"><div class="step-num">6</div><div class="step-label">Elitism + New Gen</div></div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""<div class="edu-card"><h3>📌 Key Parameters</h3><ul>
                <li><strong>Population size</strong>: individuals per generation</li>
                <li><strong>Mutation rate</strong>: per-gene Gaussian perturbation prob.</li>
                <li><strong>Crossover</strong>: Arithmetic or Blend (BLX-α)</li>
                <li><strong>Elitism</strong>: top 2 individuals always survive</li>
            </ul></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="edu-card"><h3>⚖️ Pros & Cons</h3><ul>
                <li>✅ Robust on multimodal landscapes</li>
                <li>✅ Good diversity via crossover</li>
                <li>✅ No gradient needed</li>
                <li>❌ Slower convergence than DE/PSO</li>
                <li>❌ Many hyperparameters</li>
            </ul></div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="edu-card">
            <h3>🌊 Particle Swarm Optimization (PSO)</h3>
            <p>Inspired by bird flocks and fish schools. Each <strong>particle</strong> is a weight vector θ
            that flies through the search space guided by its own best memory
            (<strong>cognitive</strong>) and the swarm's best position (<strong>social</strong>).</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("**⚡ Velocity Update Equation**")
        st.latex(r"""
            v_i \leftarrow \underbrace{w \cdot v_i}_{\text{inertia}}
            + \underbrace{c_1 r_1 (p_i - x_i)}_{\text{cognitive}}
            + \underbrace{c_2 r_2 (g - x_i)}_{\text{social}}
        """)
        st.latex(r"x_i \leftarrow x_i + v_i")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""<div class="edu-card"><h3>📌 Key Parameters</h3><ul>
                <li><strong>w</strong>: inertia — exploration/exploitation balance</li>
                <li><strong>w_min</strong>: final inertia (linear decay)</li>
                <li><strong>c1</strong>: cognitive coefficient</li>
                <li><strong>c2</strong>: social coefficient</li>
                <li><strong>Patience</strong>: early stopping threshold</li>
            </ul></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="edu-card"><h3>⚖️ GA vs PSO</h3><ul>
                <li>GA: recombination of parents → diversity</li>
                <li>PSO: velocity-guided → smooth convergence</li>
                <li>GA: population memory only</li>
                <li>PSO: each particle tracks personal best</li>
                <li>GA: slower but more robust</li>
                <li>PSO: faster on smooth landscapes</li>
            </ul></div>""", unsafe_allow_html=True)

    with tab3:
        col_de, col_abc = st.columns(2, gap="large")
        with col_de:
            st.markdown("""
            <div class="edu-card">
                <h3>🔮 Differential Evolution (DE)</h3>
                <p>DE creates a <strong>mutant</strong> vector by adding a scaled difference of two random
                population members to a third base vector: <em>v = x_r1 + F·(x_r2 − x_r3)</em>.</p>
                <p>A <strong>binomial crossover</strong> produces a trial vector. The trial replaces the
                original only if it achieves equal or higher F1 (greedy selection).</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("**DE Mutation Formula**")
            st.latex(r"v = x_{r_1} + F \cdot (x_{r_2} - x_{r_3})")
            st.latex(r"u_j = \begin{cases} v_j & \text{if } \mathcal{U}(0,1) < CR \text{ or } j = j_\text{rand} \\ x_{i,j} & \text{otherwise} \end{cases}")
            st.markdown("""<div class="edu-card"><h3>📌 DE Parameters</h3><ul>
                <li><strong>F</strong>: mutation factor (0.4–0.9 typical) — controls exploration</li>
                <li><strong>CR</strong>: crossover rate (0.7–0.9 typical) — controls exploitation</li>
                <li><strong>Strategy</strong>: rand/1/bin (default and most common)</li>
                <li><strong>Pop size</strong>: ≥ 4 required for rand/1/bin</li>
            </ul></div>""", unsafe_allow_html=True)

        with col_abc:
            st.markdown("""
            <div class="edu-card">
                <h3>🐝 Artificial Bee Colony (ABC)</h3>
                <p>Three bee types cooperate to find the best weight vector:</p>
                <ul>
                    <li><strong>Employed bees</strong>: exploit known food sources (solutions) by
                    searching in their neighbourhood (1-dimension perturbation).</li>
                    <li><strong>Onlooker bees</strong>: select sources proportional to their fitness
                    (waggle dance analogy) and apply neighbourhood search.</li>
                    <li><strong>Scout bees</strong>: when a source is exhausted (trial > limit),
                    abandon it and explore a new random solution.</li>
                </ul>
            </div>""", unsafe_allow_html=True)
            st.markdown("**ABC Neighbourhood Search**")
            st.latex(r"v_{i,j} = x_{i,j} + \phi_{i,j} \cdot (x_{i,j} - x_{k,j})")
            st.markdown(r"where $\phi \sim \mathcal{U}(-1, 1)$ and $k \neq i$ is a random partner.")
            st.markdown("""<div class="edu-card"><h3>📌 ABC Parameters</h3><ul>
                <li><strong>n_bees</strong>: colony size (n_employed = n_onlooker = n_bees/2)</li>
                <li><strong>Limit</strong>: failure threshold before scout phase; heuristic: n_emp × n_params</li>
                <li><strong>Selection</strong>: fitness-proportionate for onlookers (roulette-wheel)</li>
            </ul></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="edu-card" style="margin-top:1rem">
            <h3>📊 Algorithm Comparison Summary</h3>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
                <tr style="border-bottom:2px solid #e8eef5;color:#0d3b66;font-weight:700">
                    <td style="padding:0.4rem">Property</td>
                    <td>GA 🧬</td><td>PSO 🌊</td><td>DE 🔮</td><td>ABC 🐝</td>
                </tr>
                <tr style="border-bottom:1px solid #f0f4f9">
                    <td style="font-weight:600;color:#5a7090">Search mechanism</td>
                    <td>Crossover + mutation</td><td>Velocity update</td><td>Difference mutation</td><td>Neighbourhood search</td>
                </tr>
                <tr style="border-bottom:1px solid #f0f4f9">
                    <td style="font-weight:600;color:#5a7090">Memory</td>
                    <td>Population only</td><td>Personal + global best</td><td>Population only</td><td>Sources + trial counts</td>
                </tr>
                <tr style="border-bottom:1px solid #f0f4f9">
                    <td style="font-weight:600;color:#5a7090">Key params</td>
                    <td>pop, mut_rate, cx</td><td>w, c1, c2</td><td>F, CR</td><td>n_bees, limit</td>
                </tr>
                <tr>
                    <td style="font-weight:600;color:#5a7090">Best for</td>
                    <td>Rugged landscapes</td><td>Smooth landscapes</td><td>High-dim continuous</td><td>Multimodal problems</td>
                </tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("""<div class="edu-card">
            <h3>🏥 Parkinson's Disease — Overview</h3>
            <p>Parkinson's disease is a progressive neurodegenerative disorder caused by the loss of
            dopaminergic neurons in the <em>substantia nigra</em>. Classic motor symptoms include
            <strong>resting tremor</strong>, <strong>muscle rigidity</strong>, <strong>bradykinesia</strong>
            and <strong>postural instability</strong>.</p>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="card"><span class="card-icon">📊</span>
                <p class="card-title">Epidemiology</p>
                <p class="card-text">~10 million worldwide · 2nd most common neurodegenerative disease · Incidence: 8–18/100,000/year · Peak onset: 60–70 years old</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="card"><span class="card-icon">🎤</span>
                <p class="card-title">Voice & Diagnosis</p>
                <p class="card-text">~90% of patients have vocal changes · <strong>Jitter</strong>: frequency cycle variation · <strong>Shimmer</strong>: amplitude variation · <strong>HNR</strong>: harmonic-to-noise ratio</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="card"><span class="card-icon">💊</span>
                <p class="card-title">Treatment</p>
                <p class="card-text">No current cure · Levodopa (main treatment) · Deep Brain Stimulation (DBS) · Physiotherapy & speech therapy · Early diagnosis improves prognosis</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>")
        st.markdown("**🔗 References**")
        for title, url, desc in [
            ("UCI ML Repository — Parkinson's Dataset","https://archive.ics.uci.edu/dataset/174/parkinsons","Max A. Little et al. (2007). Dataset used in this project."),
            ("Parkinson's Foundation","https://www.parkinson.org/","Educational resources and support."),
            ("Little MA et al. (2009) — Nonlinear voice analysis","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3051371/","Exploiting nonlinear recurrence for voice disorder detection."),
            ("Kennedy & Eberhart (1995) — Original PSO","https://ieeexplore.ieee.org/document/488968","Particle swarm optimization. IEEE ICNN'95."),
            ("Storn & Price (1997) — Differential Evolution","https://link.springer.com/article/10.1023/A:1008202821328","DE for real-valued optimisation. J. Global Optimization."),
            ("Karaboga (2005) — Artificial Bee Colony","https://link.springer.com/article/10.1007/s10898-007-9149-x","ABC algorithm for numeric optimisation."),
        ]:
            st.markdown(f'<div class="ref-card"><strong><a href="{url}" target="_blank">🔗 {title}</a></strong><br><span style="color:#5a7090">{desc}</span></div>', unsafe_allow_html=True)


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
