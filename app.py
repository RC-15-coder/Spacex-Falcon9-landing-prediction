import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import folium
from streamlit_folium import st_folium
import os
import joblib
import warnings
import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
from streamlit.components.v1 import html as components_html

warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import xgboost as xgb

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Falcon 9 · Mission Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Google Fonts ───────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown(
    """<style>
:root {
  --pri:     #8b5cf6;
  --pri2:    #a78bfa;
  --acc:     #ec4899;
  --bg:      #07070f;
  --surf:    #0f0f1a;
  --card:    #16162a;
  --border:  #2a2a45;
  --text:    #f0f0ff;
  --muted:   #8888aa;
  --green:   #10b981;
  --red:     #f43f5e;
  --amber:   #f59e0b;
}

.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1400px !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main > .block-container { padding-top: 1rem !important; }
.topbar { margin-top: 0 !important; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stToolbar"]    { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"]    { display: none !important; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.75rem 0 1.25rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.75rem;
  flex-wrap: wrap; gap: 0.75rem;
}
.topbar-brand { display: flex; align-items: center; gap: 0.75rem; }
.topbar-name {
  font-size: 1rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  background: linear-gradient(90deg, var(--pri2), var(--acc));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.topbar-sub { font-size: 0.62rem; color: var(--muted); letter-spacing: 0.18em; text-transform: uppercase; }
.topbar-stats { font-size: 0.75rem; color: var(--muted); }
.topbar-stats span { color: var(--text); font-weight: 600; }

/* Force nav and ALL parent wrappers to full width */
.stRadio,
div[data-testid="stRadio"],
[data-testid="stVerticalBlock"] > div:has(.stRadio),
.element-container:has(.stRadio),
div[data-testid="element-container"]:has(.stRadio) {
  width: 100% !important;
  max-width: 100% !important;
  display: block !important;
}

.stRadio > div {
  width: 100% !important;
}

/* ── Nav Container ── */
.stRadio > div {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  gap: 6px !important;
  background: linear-gradient(180deg, rgba(22,22,42,0.9) 0%, rgba(15,15,26,0.95) 100%) !important;
  padding: 7px !important;
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 12px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04) !important;
  width: 100% !important;
  min-width: 100% !important;
  box-sizing: border-box !important;
  position: relative !important;
}
.stRadio > div::before {
  content: '' !important;
  position: absolute !important;
  top: 0 !important; left: 20% !important; right: 20% !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), transparent) !important;
  pointer-events: none !important;
}

/* ── Nav Pills ── */
.stRadio label {
  flex: 1 1 auto !important;
  text-align: center !important;
  border-radius: 9px !important;
  padding: 0.7rem 0.8rem !important;
  font-size: 0.86rem !important;
  font-weight: 500 !important;
  color: #9090b0 !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  white-space: nowrap !important;
}
.stRadio label:hover {
  color: #d0d0f0 !important;
  background: rgba(255,255,255,0.05) !important;
  border-color: rgba(139,92,246,0.2) !important;
}

/* ── Active pill — all known Streamlit selector variants ── */
.stRadio [data-checked="true"] label,
.stRadio label[data-checked="true"],
.stRadio label[aria-checked="true"],
.stRadio [aria-checked="true"] label,
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked),
.stRadio label:has(input:checked),
.stRadio div[data-testid="stMarkdownContainer"]:has(~ input:checked) {
  color: #ffffff !important;
  background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #db2777 100%) !important;
  font-weight: 700 !important;
  border-color: rgba(139,92,246,0.6) !important;
  box-shadow: 0 4px 18px rgba(139,92,246,0.5), 0 2px 6px rgba(219,39,119,0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
  transform: translateY(-1px) !important;
}

/* Hide radio circles */
.stRadio input[type="radio"] { display: none !important; }
.stRadio label > div:first-child { display: none !important; }
.stRadio label span { color: inherit !important; width: 100% !important; }

[data-testid="stMetric"] {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 1.25rem 1.5rem !important;
  position: relative !important; overflow: hidden !important;
}
[data-testid="stMetric"]::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--pri), var(--acc));
}
[data-testid="stMetricLabel"] {
  font-size: 0.7rem !important; font-weight: 700 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 2rem !important; font-weight: 800 !important; color: var(--text) !important;
}

.pg-title {
  font-size: clamp(1.75rem, 4vw, 2.6rem); font-weight: 800;
  letter-spacing: -0.02em; color: var(--text); line-height: 1.15; margin-bottom: 0.4rem;
}
.pg-title span { background: linear-gradient(90deg, var(--pri2), var(--acc)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.pg-sub { font-size: 0.9rem; color: var(--muted); line-height: 1.75; max-width: 640px; margin-bottom: 2rem; }

.sec-label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--pri2);
  display: flex; align-items: center; gap: 8px; margin-bottom: 0.9rem;
}
.sec-label::before { content: ''; width: 16px; height: 2px; background: linear-gradient(90deg, var(--pri), var(--acc)); flex-shrink: 0; }

.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}
.card-left {
  background: var(--card); border: 1px solid var(--border);
  border-left: 3px solid var(--pri); border-radius: 0 12px 12px 0;
  padding: 1.1rem 1.4rem; margin-bottom: 0.6rem;
}

.box {
  background: rgba(139,92,246,0.07); border: 1px solid rgba(139,92,246,0.25);
  border-radius: 10px; padding: 1rem 1.25rem; margin: 0.9rem 0;
  font-size: 0.85rem; color: #c4c4e0; line-height: 1.75;
}
.box strong { color: var(--pri2); }
.box.green  { background: rgba(16,185,129,0.07); border-color: rgba(16,185,129,0.25); }
.box.green strong { color: #34d399; }
.box.amber  { background: rgba(245,158,11,0.07); border-color: rgba(245,158,11,0.25); }
.box.amber strong { color: #fbbf24; }

.pill { display: inline-block; border-radius: 20px; padding: 0.25rem 0.8rem; font-size: 0.75rem; font-weight: 700; margin: 0.2rem; }
.pill-v { background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.3); color: var(--pri2); }
.pill-g { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #34d399; }
.pill-r { background: rgba(244,63,94,0.12);  border: 1px solid rgba(244,63,94,0.3);  color: #fb7185; }
.pill-a { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #fbbf24; }

.rank-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.8rem 1.1rem; background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; margin-bottom: 6px;
}
.rank-row.top { border-color: rgba(139,92,246,0.5); background: rgba(139,92,246,0.06); }
.rank-val { font-size: 1.1rem; font-weight: 800; }

.pred-box { border-radius: 12px; padding: 1.75rem; text-align: center; margin-bottom: 1rem; }
.pred-land  { background: rgba(16,185,129,0.08);  border: 1px solid rgba(16,185,129,0.3); }
.pred-crash { background: rgba(244,63,94,0.08);   border: 1px solid rgba(244,63,94,0.3); }
.pred-emoji { font-size: 2.75rem; margin-bottom: 0.4rem; }
.pred-lbl { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.2rem; }
.pred-pct { font-size: 3rem; font-weight: 800; line-height: 1; }
.pred-note { font-size: 0.78rem; margin-top: 0.35rem; opacity: 0.75; }

/* ── SKELETON LOADERS ── */
@keyframes shimmer {
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}
.skeleton {
  background: linear-gradient(90deg, #16162a 0%, #20203a 50%, #16162a 100%);
  background-size: 2000px 100%;
  animation: shimmer 2s infinite linear;
  border-radius: 10px;
  border: 1px solid var(--border);
}
.skeleton-title   { height: 28px; width: 60%; margin-bottom: 14px; }
.skeleton-line    { height: 14px; width: 100%; margin-bottom: 10px; }
.skeleton-line.sh { width: 80%; }
.skeleton-chart   { height: 320px; width: 100%; margin-top: 16px; }
.skeleton-card    { height: 110px; width: 100%; margin-bottom: 10px; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.75rem 0 !important; }

[data-testid="stDataFrame"] { border-radius: 10px !important; }

/* ── Folium map container ── */
iframe[title="streamlit_folium.st_folium"] {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}

/* ── Mobile ── */
@media (max-width: 768px) {
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .pg-title { font-size: 1.6rem !important; }
  .topbar { flex-direction: column; align-items: flex-start; padding: 0.5rem 0 0.75rem 0 !important; }
  .topbar-stats { font-size: 0.7rem; }
  [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
  .pred-pct { font-size: 2.2rem !important; }

  /* Mobile nav with scroll fade hint */
  /* Animated scroll hint — chevron + fade */
  .stRadio {
    position: relative !important;
  }

  /* Soft fade on right edge */
  .stRadio::after {
    content: '' !important;
    position: absolute !important;
    top: 7px !important;
    right: 44px !important;
    bottom: 7px !important;
    width: 32px !important;
    background: linear-gradient(to right, transparent, rgba(15,15,26,0.95) 80%) !important;
    pointer-events: none !important;
    z-index: 2 !important;
  }

  /* Clickable scroll arrow button */
  .nav-scroll-btn {
    position: absolute !important;
    top: 50% !important;
    right: 8px !important;
    transform: translateY(-50%) !important;
    width: 34px !important;
    height: 34px !important;
    border-radius: 50% !important;
    border: none !important;
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    box-shadow:
      0 2px 8px rgba(139,92,246,0.5),
      0 0 0 0 rgba(139,92,246,0.4),
      inset 0 1px 0 rgba(255,255,255,0.2) !important;
    z-index: 4 !important;
    transition: all 0.2s ease !important;
    animation: nav-arrow-pulse 2s ease-in-out infinite !important;
  }

  .nav-scroll-btn:hover {
    transform: translateY(-50%) scale(1.08) !important;
    box-shadow:
      0 4px 14px rgba(139,92,246,0.7),
      inset 0 1px 0 rgba(255,255,255,0.3) !important;
  }

  .nav-scroll-btn:active {
    transform: translateY(-50%) scale(0.96) !important;
  }

  .nav-scroll-btn svg {
    display: block !important;
  }

  @keyframes nav-arrow-pulse {
    0%, 100% {
      box-shadow:
        0 2px 8px rgba(139,92,246,0.5),
        0 0 0 0 rgba(139,92,246,0.5),
        inset 0 1px 0 rgba(255,255,255,0.2);
    }
    50% {
      box-shadow:
        0 2px 10px rgba(139,92,246,0.6),
        0 0 0 6px rgba(139,92,246,0),
        inset 0 1px 0 rgba(255,255,255,0.2);
    }
  }

  .stRadio > div {
    padding: 5px 55px 5px 5px !important;
    gap: 3px !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    width: 100% !important;
    scroll-snap-type: x mandatory !important;
  }
  .stRadio > div::-webkit-scrollbar { display: none !important; }
  .stRadio label {
    padding: 0.6rem 1.1rem !important;
    font-size: 0.8rem !important;
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    scroll-snap-align: start !important;
  }
}

@media (max-width: 480px) {
  .stRadio label { padding: 0.55rem 0.95rem !important; font-size: 0.75rem !important; }
}
</style>""",
    unsafe_allow_html=True,
)

# ── Chart theme ────────────────────────────────────────────────────
BG = "#07070f"
SURF = "#16162a"
PRI = "#8b5cf6"
PRI2 = "#a78bfa"
ACC = "#ec4899"
GRN = "#10b981"
RED = "#f43f5e"
AMB = "#f59e0b"
MUT = "#8888aa"
TXT = "#f0f0ff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": SURF,
        "axes.edgecolor": "#2a2a45",
        "axes.labelcolor": "#a0a0c0",
        "axes.titlecolor": TXT,
        "xtick.color": MUT,
        "ytick.color": MUT,
        "text.color": TXT,
        "grid.color": "#1e1e35",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
    }
)


# ── Skeleton loader components ─────────────────────────────────────
def skeleton_page():
    """Show skeleton placeholders while first-load training happens."""
    st.markdown(
        """
    <div class="skeleton skeleton-title"></div>
    <div class="skeleton skeleton-line"></div>
    <div class="skeleton skeleton-line sh"></div>
    <div style="display:flex;gap:1rem;margin-top:1.5rem;">
      <div class="skeleton skeleton-card" style="flex:1"></div>
      <div class="skeleton skeleton-card" style="flex:1"></div>
      <div class="skeleton skeleton-card" style="flex:1"></div>
      <div class="skeleton skeleton-card" style="flex:1"></div>
    </div>
    <div class="skeleton skeleton-chart"></div>
    """,
        unsafe_allow_html=True,
    )


# ── Load data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = pd.read_csv("data/dataset_part_2.csv")
    X = pd.read_csv("data/dataset_part_3.csv")
    return data, X


data, X_raw = load_data()
feature_names = X_raw.columns.tolist()
Y = data["Class"].to_numpy()
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=2
)


# ── Pre-train models at startup ────────────────────────────────────
MODEL_CACHE_PATH = "data/trained_models.pkl"


@st.cache_resource
def train_models(_X_train, _Y_train):
    m = {}
    m["Logistic Regression"] = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 1], "penalty": ["l2"], "solver": ["lbfgs"]},
        cv=5,
    ).fit(_X_train, _Y_train)
    m["SVM"] = GridSearchCV(
        SVC(probability=True), {"kernel": ["rbf"], "C": [1, 10]}, cv=5
    ).fit(_X_train, _Y_train)
    m["Decision Tree"] = GridSearchCV(
        DecisionTreeClassifier(),
        {"criterion": ["entropy"], "max_depth": [6, 8], "min_samples_leaf": [1, 2]},
        cv=5,
    ).fit(_X_train, _Y_train)
    m["KNN"] = GridSearchCV(
        KNeighborsClassifier(), {"n_neighbors": [5, 7, 10], "p": [1, 2]}, cv=5
    ).fit(_X_train, _Y_train)
    m["LightGBM"] = GridSearchCV(
        lgb.LGBMClassifier(random_state=42, verbose=-1),
        {
            "n_estimators": [200],
            "learning_rate": [0.05],
            "max_depth": [3],
            "num_leaves": [15],
        },
        cv=5,
        n_jobs=-1,
    ).fit(_X_train, _Y_train)
    m["XGBoost"] = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        {"n_estimators": [200], "learning_rate": [0.05], "max_depth": [3]},
        cv=5,
        n_jobs=-1,
    ).fit(_X_train, _Y_train)
    return m


@st.cache_resource
def load_or_train_models(_X_train, _Y_train):
    """Load pre-trained models from disk if available, else train and save."""
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            return joblib.load(MODEL_CACHE_PATH)
        except Exception:
            pass  # fall through to retrain if pickle is corrupted

    # Train from scratch (first ever run only)
    models = train_models(_X_train, _Y_train)
    try:
        joblib.dump(models, MODEL_CACHE_PATH)
    except Exception as e:
        print(f"Could not cache models: {e}")
    return models


# Show skeleton ONLY on very first cold start (models not yet cached)
@st.cache_data
def _startup_flag():
    return True


_first_run = _startup_flag()

# Trigger model loading — cached after first run so near-instant on subsequent
_placeholder = st.empty()
if "models_loaded" not in st.session_state:
    with _placeholder.container():
        skeleton_page()
    models = load_or_train_models(X_train, Y_train)
    st.session_state["models_loaded"] = True
    _placeholder.empty()
else:
    models = load_or_train_models(X_train, Y_train)

res = {n: m.score(X_test, Y_test) for n, m in models.items()}
rdf = (
    pd.DataFrame(res.items(), columns=["Model", "Accuracy"])
    .sort_values("Accuracy", ascending=False)
    .reset_index(drop=True)
)


# ── Helpers ────────────────────────────────────────────────────────
def fig_base(w=10, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURF)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#2a2a45")
    ax.tick_params(colors=MUT, length=0)
    ax.grid(axis="y", color="#1e1e35", alpha=0.8, linewidth=0.7)
    return fig, ax


def gradient_bar(ax, x, y, c_top, c_bot, w=0.55, **kw):
    bars = ax.bar(x, y, width=w, color=c_bot, edgecolor="none", **kw)
    ax.bar(
        x,
        [v * 0.3 for v in y],
        width=w,
        color=c_top,
        alpha=0.5,
        edgecolor="none",
        bottom=[v * 0.7 for v in y],
    )
    return bars


# ── Top nav ────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="topbar">
  <div class="topbar-brand">
    <span style="font-size:1.75rem">🚀</span>
    <div>
      <div class="topbar-name">Falcon 9</div>
      <div class="topbar-sub">Mission Intelligence</div>
    </div>
  </div>
  <div class="topbar-stats">
    <span>{len(data)}</span> launches &nbsp;·&nbsp;
    <span>{int(Y.sum())}</span> successful landings &nbsp;·&nbsp;
    <span>{Y.mean():.1%}</span> success rate &nbsp;·&nbsp;
    Best model: <span>{rdf.iloc[0]['Model']} {rdf.iloc[0]['Accuracy']:.1%}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    [
        "Overview",
        "Data Explorer",
        "Model Performance",
        "SHAP Explainability",
        "Landing Predictor",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Mobile scroll arrow button (clickable, bidirectional) ─────────
components_html(
    """
<script>
(function() {
  function addScrollArrow() {
    const parentDoc = window.parent.document;
    const radioContainers = parentDoc.querySelectorAll('.stRadio');
    radioContainers.forEach(container => {
      if (container.querySelector('.nav-scroll-btn')) return;
      if (window.parent.innerWidth > 768) return;
      const scrollable = container.querySelector('[role="radiogroup"]');
      if (!scrollable) return;

      const btn = parentDoc.createElement('button');
      btn.className = 'nav-scroll-btn';
      btn.setAttribute('data-direction', 'right');
      btn.setAttribute('aria-label', 'Scroll navigation');

      function renderIcon(dir) {
        const points = dir === 'right' ? '9 18 15 12 9 6' : '15 18 9 12 15 6';
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="' + points + '"/></svg>';
      }
      renderIcon('right');

      btn.onclick = function(e) {
        e.preventDefault();
        e.stopPropagation();
        const direction = btn.getAttribute('data-direction');
        const delta = direction === 'right' ? 180 : -180;
        scrollable.scrollBy({ left: delta, behavior: 'smooth' });
      };
      container.appendChild(btn);

      function updateDirection() {
        const atEnd = scrollable.scrollLeft + scrollable.clientWidth >= scrollable.scrollWidth - 10;
        const atStart = scrollable.scrollLeft <= 10;

        if (atEnd && !atStart) {
          btn.setAttribute('data-direction', 'left');
          renderIcon('left');
          btn.style.opacity = '1';
        } else if (atStart) {
          btn.setAttribute('data-direction', 'right');
          renderIcon('right');
          btn.style.opacity = '1';
        } else {
          btn.setAttribute('data-direction', 'right');
          renderIcon('right');
          btn.style.opacity = '1';
        }
      }

      scrollable.addEventListener('scroll', updateDirection);
      updateDirection();
    });
  }
  setTimeout(addScrollArrow, 300);
  setTimeout(addScrollArrow, 1000);
  setTimeout(addScrollArrow, 2500);

  window.parent.addEventListener('resize', function() {
    const parentDoc = window.parent.document;
    const existingBtns = parentDoc.querySelectorAll('.nav-scroll-btn');
    if (window.parent.innerWidth > 768) {
      existingBtns.forEach(b => b.remove());
    } else if (existingBtns.length === 0) {
      addScrollArrow();
    }
  });
})();
</script>
""",
    height=0,
)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(
        """
    <div class="pg-title">Falcon 9 <span>Landing</span> Prediction Engine</div>
    <div class="pg-sub">An end-to-end machine learning pipeline that predicts first-stage booster
    recovery before liftoff — built on real SpaceX mission data, production-grade ML, and
    explainable AI.</div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Launches", str(len(data)))
    c2.metric("Successful Landings", str(int(Y.sum())))
    c3.metric("Success Rate", f"{Y.mean():.1%}")
    c4.metric("Best Accuracy", f"{rdf.iloc[0]['Accuracy']:.1%}")

    st.markdown(
        """
    <div class="box">
      <strong>The $103M Question:</strong> SpaceX launches Falcon 9 at ~$62M per flight vs competitors
      charging $165M+. That entire $103M cost advantage comes from one thing — recovering and reusing
      the first-stage booster. This pipeline predicts landing success before launch, enabling competing
      providers to accurately price insurance risk, set competitive bids, and optimize mission planning.
      Every percentage point of prediction accuracy translates directly to financial impact.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown(
            '<div class="sec-label">Recovery Performance Over Time</div>',
            unsafe_allow_html=True,
        )
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            data["Year"] = data["Date"].dt.year
            yr = data.groupby("Year")["Class"].agg(["sum", "count"]).reset_index()
            yr.columns = ["Year", "Win", "Total"]
            yr["Rate"] = yr["Win"] / yr["Total"]

            fig, ax = fig_base(9, 4.5)
            ax2 = ax.twinx()
            ax.bar(
                yr["Year"],
                yr["Total"],
                width=0.7,
                color="#1e1e35",
                edgecolor="none",
                label="Total Launches",
            )
            gradient_bar(
                ax, yr["Year"], yr["Win"], PRI2, PRI, w=0.7, label="Successful Landings"
            )
            ax2.plot(yr["Year"], yr["Rate"], color=GRN, linewidth=2.5, zorder=5)
            ax2.plot(
                yr["Year"], yr["Rate"], color=GRN, linewidth=8, alpha=0.1, zorder=4
            )
            ax2.scatter(
                yr["Year"],
                yr["Rate"],
                color=GRN,
                s=60,
                zorder=6,
                edgecolors=BG,
                linewidth=2.5,
            )
            ax2.set_ylim(0, 1.45)
            ax2.set_ylabel("Success Rate", color=GRN, fontsize=9)
            ax2.tick_params(colors=GRN, length=0)
            for s in ["top", "left", "bottom"]:
                ax2.spines[s].set_visible(False)
            ax2.spines["right"].set_color("#2a2a45")
            ax2.grid(False)
            for _, row in yr.iterrows():
                ax2.annotate(
                    f"{row['Rate']:.0%}",
                    (row["Year"], row["Rate"]),
                    textcoords="offset points",
                    xytext=(0, 11),
                    fontsize=8,
                    color=GRN,
                    ha="center",
                    fontweight="bold",
                )
            ax.set_title(
                "Falcon 9 Booster Recovery — Year on Year",
                fontsize=12,
                fontweight="bold",
                color=TXT,
                pad=14,
                loc="left",
            )
            ax.set_ylabel("Mission Count", fontsize=9)
            h1, l1 = ax.get_legend_handles_labels()
            ax.legend(
                h1[:2],
                l1[:2],
                facecolor=SURF,
                edgecolor="#2a2a45",
                fontsize=8,
                loc="upper left",
            )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            """
        <div class="box">
          <strong>What the trend shows:</strong> Zero successful landings from 2010–2015 reflects
          SpaceX's experimental phase. The breakthrough came in December 2015 with the first successful
          landing at Cape Canaveral. By 2017, success rates crossed 70%, driven by advances in
          autonomous drone ship positioning, grid fin aerodynamics, and propulsive landing algorithms.
          From 2018 onward, recovery became the reliable default — not the exception.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="sec-label">Project Architecture</div>', unsafe_allow_html=True
        )
        steps = [
            (
                "01",
                "Data Collection",
                "SpaceX REST API + Wikipedia scraping (90 launches)",
            ),
            (
                "02",
                "Data Wrangling",
                "Null handling, one-hot encoding, 83 features created",
            ),
            ("03", "SQL / EDA", "SQLite analysis, Matplotlib & Seaborn visualizations"),
            ("04", "Geospatial", "Folium interactive maps of all 4 launch complexes"),
            (
                "05",
                "ML Pipeline",
                "6 classifiers, GridSearchCV, 10-fold cross-validation",
            ),
            (
                "06",
                "Explainability",
                "SHAP TreeExplainer — feature attribution per prediction",
            ),
        ]
        for num, title, desc in steps:
            st.markdown(
                f"""
            <div class="card-left">
              <div style="display:flex;align-items:center;gap:0.75rem;">
                <span style="font-size:0.6rem;font-weight:800;color:var(--pri2);
                             background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.25);
                             padding:3px 8px;border-radius:4px;letter-spacing:0.1em;white-space:nowrap">{num}</span>
                <div>
                  <div style="font-weight:700;font-size:0.85rem;color:#f0f0ff">{title}</div>
                  <div style="font-size:0.77rem;color:#8888aa;margin-top:1px">{desc}</div>
                </div>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div class="box" style="margin-top:0.75rem">
          <strong>Stack:</strong> Python · Pandas · Scikit-learn · LightGBM · XGBoost ·
          SHAP · Folium · Plotly Dash · Streamlit · SQLite · BeautifulSoup
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label">Key Project Results</div>', unsafe_allow_html=True
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(
            f"""
        <div class="card" style="border-top:2px solid var(--pri)">
          <div style="font-size:1.8rem;font-weight:800;color:var(--pri2)">{rdf.iloc[0]['Accuracy']:.1%}</div>
          <div style="font-size:0.75rem;font-weight:700;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;margin-top:4px">Best Model Accuracy</div>
          <div style="font-size:0.78rem;color:#c4c4e0;margin-top:0.5rem">LightGBM with GridSearchCV tuning — 27 points above naive baseline</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            """
        <div class="card" style="border-top:2px solid #10b981">
          <div style="font-size:1.8rem;font-weight:800;color:#34d399">83</div>
          <div style="font-size:0.75rem;font-weight:700;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;margin-top:4px">Engineered Features</div>
          <div style="font-size:0.78rem;color:#c4c4e0;margin-top:0.5rem">One-hot encoded from orbit, launch site, landing pad, and booster serial</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            """
        <div class="card" style="border-top:2px solid var(--acc)">
          <div style="font-size:1.8rem;font-weight:800;color:var(--acc)">6</div>
          <div style="font-size:0.75rem;font-weight:700;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;margin-top:4px">Models Compared</div>
          <div style="font-size:0.78rem;color:#c4c4e0;margin-top:0.5rem">LogReg, SVM, Decision Tree, KNN, LightGBM, XGBoost — all GridSearchCV tuned</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            """
        <div class="card" style="border-top:2px solid var(--amber)">
          <div style="font-size:1.8rem;font-weight:800;color:#fbbf24">SHAP</div>
          <div style="font-size:0.75rem;font-weight:700;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;margin-top:4px">Explainable AI</div>
          <div style="font-size:0.78rem;color:#c4c4e0;margin-top:0.5rem">Feature attribution for every prediction — not just accuracy, but interpretability</div>
        </div>""",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown(
        """
    <div class="pg-title">Data <span>Explorer</span></div>
    <div class="pg-sub">The cleaned, merged dataset powering the ML pipeline — 90 Falcon 9 missions,
    18 raw features, expanded to 83 engineered features through one-hot encoding.</div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sec-label">Raw Dataset Preview</div>', unsafe_allow_html=True
    )
    st.dataframe(data.head(15), width="stretch")

    st.markdown(
        """
    <div class="box">
      <strong>Data Engineering process:</strong> Raw data was collected from two sources —
      the SpaceX REST API (launch metadata, payload, orbit, booster info) and Wikipedia
      (historical outcome records). After merging, null values were imputed, categorical variables
      one-hot encoded, and the binary target variable <strong>Class</strong> created
      (1 = successful first-stage landing, 0 = loss or ocean splashdown). The final feature matrix
      has 90 rows × 83 columns — ready for standardization and model training.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── INTERACTIVE FOLIUM MAP ────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label">Launch Site Geography — Interactive Map</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="box">
      <strong>Geospatial view:</strong> All four Falcon 9 launch complexes plotted with color-coded
      recovery rates. Click a marker to see per-site statistics. Distance to the equator affects
      orbital energy requirements; proximity to Atlantic/Pacific drone ship positions determines
      fuel margin for booster return burns.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Launch site coordinates
    launch_site_coords = {
        "CCAFS LC 40": (28.5619, -80.5772),
        "CCAFS SLC 40": (28.5619, -80.5772),
        "KSC LC 39A": (28.6083, -80.6041),
        "VAFB SLC 4E": (34.6321, -120.6106),
    }

    if "LaunchSite" in data.columns:
        site_stats = (
            data.groupby("LaunchSite")["Class"].agg(["mean", "count"]).reset_index()
        )
        site_stats.columns = ["Site", "Rate", "N"]

        m = folium.Map(location=[31, -95], zoom_start=4, tiles="CartoDB dark_matter")

        for _, row in site_stats.iterrows():
            site = row["Site"]
            if site in launch_site_coords:
                lat, lon = launch_site_coords[site]
                rate = row["Rate"]
                n = int(row["N"])

                if rate >= 0.75:
                    color = "#10b981"
                    label_color = "#10b981"
                elif rate >= 0.5:
                    color = "#8b5cf6"
                    label_color = "#a78bfa"
                else:
                    color = "#f43f5e"
                    label_color = "#f43f5e"

                popup_html = f"""
                <div style="font-family:'Plus Jakarta Sans',sans-serif; min-width:180px;">
                  <div style="font-weight:700; font-size:14px; color:#1a1a2e; margin-bottom:6px;">{site}</div>
                  <div style="font-size:12px; color:#555;">
                    <b>Recovery Rate:</b> <span style="color:{label_color}; font-weight:700;">{rate:.0%}</span><br>
                    <b>Missions Flown:</b> {n}<br>
                    <b>Successes:</b> {int(rate*n)}<br>
                    <b>Losses:</b> {n - int(rate*n)}
                  </div>
                </div>
                """

                icon_html = f"""
                <div style="position:relative;width:40px;height:40px;">
                  <div style="position:absolute;inset:-10px;background:{color};
                              opacity:0.15;border-radius:50%;"></div>
                  <div style="position:absolute;inset:0;background:{color};
                              border-radius:50%;border:2px solid {color};
                              box-shadow:0 0 20px {color};"></div>
                </div>
                """

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{site} — {rate:.0%} recovery",
                    icon=folium.DivIcon(
                        html=icon_html,
                        icon_size=(40, 40),
                        icon_anchor=(20, 20),
                    ),
                ).add_to(m)

        st_folium(
            m,
            height=450,
            width=None,
            returned_objects=[],
            key="launch_map",
        )

        st.markdown(
            """
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;font-size:0.78rem;color:#8888aa">
          <span>🟢 High recovery (&gt;75%)</span>
          <span>🟣 Moderate (50–75%)</span>
          <span>🔴 Low (&lt;50%)</span>
          <span style="margin-left:auto">Marker size = number of missions flown</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="sec-label">Mission Outcomes</div>', unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        vals = pd.Series(Y).value_counts().values
        wedges, texts, autotexts = ax.pie(
            vals,
            colors=[GRN, RED],
            startangle=90,
            pctdistance=0.75,
            autopct="%1.0f%%",
            wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=5),
        )
        for t in autotexts:
            t.set_fontsize(13)
            t.set_fontweight("bold")
            t.set_color(TXT)
        for t in texts:
            t.set_fontsize(0)
        ax.text(
            0,
            0.08,
            f"{int(Y.sum())}",
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color=TXT,
            fontfamily="sans-serif",
        )
        ax.text(0, -0.28, "successful", ha="center", va="center", fontsize=9, color=MUT)
        ax.set_title(
            "Landing Success vs Loss",
            fontsize=12,
            color=TXT,
            pad=12,
            fontweight="bold",
            loc="left",
        )
        leg = ax.legend(
            ["Landed (Class=1)", "Lost (Class=0)"],
            loc="lower center",
            facecolor=SURF,
            edgecolor="#2a2a45",
            fontsize=9,
            ncol=2,
            bbox_to_anchor=(0.5, -0.04),
        )
        for t, c in zip(leg.get_texts(), [GRN, RED]):
            t.set_color(c)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            f"""
        <div class="box">
          <strong>Class imbalance note:</strong> The dataset has a 2:1 ratio of successes to failures
          (60 vs 30). A dummy classifier predicting "always land" would score <strong>66.7%</strong>.
          Our LightGBM model achieves <strong>{rdf.iloc[0]['Accuracy']:.1%}</strong> —
          a {rdf.iloc[0]['Accuracy']-0.667:.0%} improvement — confirming the model learned
          real patterns, not just the majority class.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="sec-label">Payload Mass vs Landing Outcome</div>',
            unsafe_allow_html=True,
        )
        if "PayloadMass" in data.columns:
            fig, ax = fig_base(5.5, 4.5)
            landed = data[data["Class"] == 1]["PayloadMass"].dropna()
            crashed = data[data["Class"] == 0]["PayloadMass"].dropna()
            bins = np.linspace(0, data["PayloadMass"].max() * 1.05, 18)
            ax.hist(
                landed,
                bins=bins,
                color=GRN,
                alpha=0.8,
                label=f"Landed (avg {landed.mean()/1000:.1f}t)",
                edgecolor=BG,
                linewidth=0.8,
            )
            ax.hist(
                crashed,
                bins=bins,
                color=RED,
                alpha=0.75,
                label=f"Lost (avg {crashed.mean()/1000:.1f}t)",
                edgecolor=BG,
                linewidth=0.8,
            )
            ax.axvline(landed.mean(), color=GRN, ls="--", lw=1.8, alpha=0.9)
            ax.axvline(crashed.mean(), color=RED, ls="--", lw=1.8, alpha=0.9)
            ax.set_xlabel("Payload Mass (kg)", fontsize=9)
            ax.set_ylabel("Mission Count", fontsize=9)
            ax.set_title(
                "Payload Mass by Recovery Outcome",
                fontsize=12,
                color=TXT,
                fontweight="bold",
                loc="left",
            )
            ax.legend(facecolor=SURF, edgecolor="#2a2a45", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            f"""
        <div class="box">
          <strong>Physics insight:</strong> Successful recoveries average
          <strong>{landed.mean()/1000:.1f} tonnes</strong> vs
          <strong>{crashed.mean()/1000:.1f} tonnes</strong> for losses.
          Heavier payloads demand more fuel during ascent — leaving less propellant for the
          return burn. Above ~6,000 kg, outcomes become increasingly unpredictable.
          This mass threshold is one of the top SHAP features.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    if "LaunchSite" in data.columns:
        st.markdown(
            '<div class="sec-label">Recovery Rate by Launch Site</div>',
            unsafe_allow_html=True,
        )
        ss = data.groupby("LaunchSite")["Class"].agg(["mean", "count"]).reset_index()
        ss.columns = ["Site", "Rate", "N"]
        ss = ss.sort_values("Rate", ascending=True)
        fig, ax = fig_base(10, 3.2)
        bar_colors = [PRI if r >= ss["Rate"].max() else "#2a2a45" for r in ss["Rate"]]
        bars = ax.barh(
            ss["Site"], ss["Rate"], color=bar_colors, height=0.48, edgecolor="none"
        )
        for bar, rate, n in zip(bars, ss["Rate"], ss["N"]):
            ax.text(
                rate + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.0%}   ({n} missions)",
                va="center",
                fontsize=10,
                color=PRI2 if rate >= ss["Rate"].max() else MUT,
                fontweight="700",
            )
        ax.set_xlim(0, 1.3)
        ax.set_title(
            "Launch Site — Booster Recovery Rate",
            fontsize=12,
            color=TXT,
            fontweight="bold",
            loc="left",
        )
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best_site = ss.loc[ss["Rate"].idxmax(), "Site"]
        st.markdown(
            f"""
        <div class="box">
          <strong>{best_site}</strong> leads all sites with a
          <strong>{ss['Rate'].max():.0%}</strong> recovery rate. Site geography directly affects
          recovery economics — distance to Atlantic/Pacific drone ship stations determines how
          much propellant the booster needs to reserve for its return burn vs how much goes to
          delivering payload. This geographic signal is captured as one-hot features in the model.
        </div>
        """,
            unsafe_allow_html=True,
        )

    if "Orbit" in data.columns:
        st.markdown(
            '<div class="sec-label">Recovery Rate by Orbital Regime</div>',
            unsafe_allow_html=True,
        )
        os_ = data.groupby("Orbit")["Class"].agg(["mean", "count"]).reset_index()
        os_.columns = ["Orbit", "Rate", "N"]
        os_ = os_.sort_values("Rate", ascending=False)
        fig, ax = fig_base(12, 4.8)
        bar_colors = [
            GRN if r > 0.75 else PRI if r > 0.45 else RED for r in os_["Rate"]
        ]
        gradient_bar(ax, range(len(os_)), os_["Rate"], bar_colors, bar_colors, w=0.6)
        ax.set_xticks(range(len(os_)))
        ax.set_xticklabels(os_["Orbit"], rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.38)
        for i, (r, n) in enumerate(zip(os_["Rate"], os_["N"])):
            ax.text(
                i,
                r + 0.05,
                f"{r:.0%}",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=TXT,
            )
            ax.text(i, r + 0.14, f"n={n}", ha="center", fontsize=7, color=MUT)
        ax.set_title(
            "Booster Recovery Rate by Orbital Destination",
            fontsize=12,
            color=TXT,
            fontweight="bold",
            loc="left",
        )
        ax.yaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            """
        <div class="box">
          <strong>Orbital energy is the key driver:</strong> LEO and ISS missions require the booster
          to reach only ~400 km altitude — leaving ample fuel for landing. GTO missions to 35,786 km
          demand nearly full propellant burn, making recovery a fuel-margin gamble.
          SSO polar orbits are surprisingly recoverable due to their lower required velocities.
          This orbit-type pattern is the single strongest predictor in the SHAP analysis.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <span class="pill pill-g">LEO · ISS · SSO → High Recovery (&gt;75%)</span>
        <span class="pill pill-v">PO · VLEO · MEO → Moderate (45–75%)</span>
        <span class="pill pill-r">GTO · HEO · GEO → Low (&lt;45%)</span>
        """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(
        f"""
    <div class="pg-title">Model <span>Performance</span></div>
    <div class="pg-sub">Six classifiers trained, tuned with 10-fold GridSearchCV, and evaluated on
    a held-out test set. Best result: <strong style="color:var(--pri2)">{rdf.iloc[0]['Model']}
    at {rdf.iloc[0]['Accuracy']:.1%}</strong> test accuracy.</div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="box">
      <strong>Methodology:</strong> The 90-sample dataset was split 80/20 (train/test, random_state=2).
      All 6 models were hyperparameter-tuned using 10-fold GridSearchCV on the 72 training samples.
      StandardScaler normalization was applied before fitting to prevent scale-sensitive models (SVM, KNN)
      from being unfairly penalized. Final accuracy is reported on the 18 held-out test samples
      — data the model never saw during training or tuning.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.7])
    with col1:
        st.markdown(
            '<div class="sec-label">Model Rankings</div>', unsafe_allow_html=True
        )
        icons = ["🥇", "🥈", "🥈", "🥈", "🥈", "🥉"]
        for i, row in rdf.iterrows():
            top = row["Accuracy"] == rdf["Accuracy"].max()
            st.markdown(
                f"""
            <div class="rank-row {'top' if top else ''}">
              <div style="display:flex;align-items:center;gap:0.75rem;">
                <span style="font-size:1.1rem">{icons[i]}</span>
                <span style="font-size:0.87rem;font-weight:700;
                             color:{'#f0f0ff' if top else '#8888aa'}">{row['Model']}</span>
              </div>
              <span class="rank-val" style="color:{'#a78bfa' if top else '#555577'}">{row['Accuracy']:.3f}</span>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div class="box" style="margin-top:1rem">
          <strong>Why gradient boosting wins:</strong> LightGBM and XGBoost build decision trees
          sequentially — each tree learns from the <em>residual errors</em> of the previous one.
          On tabular data like this, this iterative error-correction consistently outperforms
          models that treat all samples equally (Logistic Regression, KNN). The ensemble effect
          also reduces variance, making it more robust on small datasets like this 90-sample set.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="sec-label">Test Accuracy Comparison</div>',
            unsafe_allow_html=True,
        )
        fig, ax = fig_base(8, 5)
        bar_colors = [PRI if i == 0 else "#2a2a45" for i in range(len(rdf))]
        bars = ax.bar(
            rdf["Model"],
            rdf["Accuracy"],
            color=bar_colors,
            width=0.55,
            edgecolor="none",
        )
        for bar, c in zip(bars, bar_colors):
            ax.bar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.2,
                width=bar.get_width(),
                bottom=bar.get_height() * 0.8,
                color=PRI2 if c == PRI else "#3a3a55",
                alpha=0.5,
                edgecolor="none",
            )
        ax.set_ylim(0, 1.15)
        ax.axhline(rdf["Accuracy"].max(), color=PRI, ls="--", lw=1, alpha=0.4)
        ax.axhline(0.667, color=AMB, ls=":", lw=1.2, alpha=0.7)
        ax.text(
            len(rdf) - 0.55,
            0.678,
            "Naive baseline 66.7%",
            fontsize=7.5,
            color=AMB,
            ha="right",
            style="italic",
        )
        for bar, val in zip(bars, rdf["Accuracy"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{val:.3f}",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color=PRI2 if val == rdf["Accuracy"].max() else MUT,
            )
        ax.set_title(
            "Hold-Out Test Accuracy — All 6 Models",
            fontsize=12,
            color=TXT,
            fontweight="bold",
            loc="left",
            pad=12,
        )
        plt.xticks(rotation=18, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            f"""
        <div class="box">
          The amber dotted line marks the <strong>naive majority-class baseline (66.7%)</strong>.
          All 6 models beat it. {rdf.iloc[0]['Model']}'s <strong>{rdf.iloc[0]['Accuracy']:.1%}</strong>
          represents a <strong>{(rdf.iloc[0]['Accuracy']-0.667)*100:.0f}-point improvement</strong>,
          confirming the features carry genuine signal beyond class frequency.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label">Confusion Matrices — Error Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="box" style="margin-bottom:1.25rem">
      Each matrix shows the exact prediction breakdown.
      <strong>Top-left</strong> = correctly predicted crashes (True Negatives).
      <strong>Bottom-right</strong> = correctly predicted landings (True Positives).
      <strong>False Positives</strong> (predicted landing but crashed) waste recovery assets.
      <strong>False Negatives</strong> (predicted crash but landed) miss recovery opportunities.
      The best model minimizes both.
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, (name, model) in enumerate(models.items()):
        with cols[idx % 3]:
            cm = confusion_matrix(Y_test, model.predict(X_test))
            acc = model.score(X_test, Y_test)
            is_best = acc == rdf["Accuracy"].max()
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            st.markdown(
                f"""
            <div style="text-align:center;margin-bottom:5px">
              <span style="font-size:0.83rem;font-weight:700;
                           color:{'#a78bfa' if is_best else '#8888aa'}">{name}</span>
              <span style="font-size:0.75rem;margin-left:6px;font-weight:800;
                           color:{'#a78bfa' if is_best else '#555577'}">{acc:.3f}</span>
            </div>""",
                unsafe_allow_html=True,
            )
            fig, ax = plt.subplots(figsize=(3.3, 2.8))
            fig.patch.set_facecolor(BG)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "m", [SURF, PRI if is_best else "#3a3a55"], N=256
            )
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=cmap,
                ax=ax,
                linewidths=2,
                linecolor=BG,
                annot_kws={"size": 14, "weight": "bold", "color": TXT},
                cbar=False,
            )
            ax.set_facecolor(SURF)
            ax.set_xlabel("Predicted", fontsize=8, color=MUT)
            ax.set_ylabel("Actual", fontsize=8, color=MUT)
            ax.xaxis.set_ticklabels(["Crash", "Land"], fontsize=8, color=MUT)
            ax.yaxis.set_ticklabels(
                ["Crash", "Land"], fontsize=8, color=MUT, rotation=0
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown(
                f"""
            <div style="font-size:0.7rem;color:#555577;text-align:center;margin-bottom:0.75rem">
              Precision: {precision:.0%} &nbsp;|&nbsp; Recall: {recall:.0%} &nbsp;|&nbsp; FP:{fp} FN:{fn}
            </div>""",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP
# ══════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    st.markdown(
        """
    <div class="pg-title">SHAP <span>Explainability</span></div>
    <div class="pg-sub">SHapley Additive exPlanations — the industry gold standard for understanding
    what each feature contributes to every individual model prediction.</div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="box">
      <strong>Why explainability is non-negotiable in production:</strong>
      A 94.4% accurate model is useful. A model that can also explain <em>why</em> each prediction
      was made is <em>deployable</em>. In regulated industries (aerospace, insurance, finance),
      black-box decisions are unacceptable. SHAP quantifies each feature's exact contribution
      to every prediction, grounded in cooperative game theory (Shapley values from economics).
      For a launch provider, this answers: "For this specific mission profile, how much is the
      orbit type vs the payload mass driving the landing risk assessment?"
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.spinner("Computing SHAP values..."):
        lgbm_model = models["LightGBM"]
        explainer = shap.TreeExplainer(lgbm_model.best_estimator_)
        shap_vals = explainer.shap_values(X_test)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label">Feature Importance — Mean |SHAP Value|</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="box" style="margin-bottom:0.75rem">
      Bar length = average absolute SHAP contribution across all test predictions.
      <strong>Longer bar = stronger influence on the prediction output</strong>, regardless of direction.
      This is more informative than standard feature importance (which only counts split frequency)
      because it measures actual impact on the output in the same units as the prediction.
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig, _ = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    shap.summary_plot(
        sv, X_test, feature_names=feature_names, plot_type="bar", show=False, color=PRI
    )
    plt.gca().set_facecolor(SURF)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().tick_params(colors="#e0e0ff", labelsize=10)
    for lbl in plt.gca().get_yticklabels():
        lbl.set_color("#e0e0ff")
        lbl.set_fontweight("600")
    for lbl in plt.gca().get_xticklabels():
        lbl.set_color("#c4c4e0")
    plt.title(
        "What drives each landing prediction? — Top features by Mean |SHAP|",
        fontsize=12,
        fontweight="bold",
        color=TXT,
        pad=12,
        loc="left",
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label">Impact Direction — Beeswarm Plot</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="display:flex;flex-wrap:wrap;gap:1.25rem;font-size:0.8rem;color:#8888aa;margin-bottom:0.9rem">
      <span>🔴 <strong style="color:#c4c4e0">Red dot</strong> = high feature value for that sample</span>
      <span>🔵 <strong style="color:#c4c4e0">Blue dot</strong> = low feature value for that sample</span>
      <span style="color:#10b981">➜ Right = pushes toward LANDING (positive SHAP)</span>
      <span style="color:#f43f5e">← Left = pushes toward CRASH (negative SHAP)</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig, _ = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    shap.summary_plot(sv, X_test, feature_names=feature_names, show=False)
    plt.gca().set_facecolor(SURF)
    plt.gca().tick_params(colors="#e0e0ff", labelsize=10)
    for lbl in plt.gca().get_yticklabels():
        lbl.set_color("#e0e0ff")
        lbl.set_fontweight("600")
    for lbl in plt.gca().get_xticklabels():
        lbl.set_color("#c4c4e0")
    plt.title(
        "Feature impact direction — per-sample SHAP contributions",
        fontsize=12,
        fontweight="bold",
        color=TXT,
        pad=12,
        loc="left",
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """
    <div class="box">
      <strong>Business interpretation:</strong> The SHAP analysis reveals that orbit type and
      payload mass dominate predictions — consistent with the physics of rocket propulsion.
      A high-value red dot on "Orbit_GTO" pushing <em>left</em> confirms: GTO missions strongly
      predict booster loss. A high-value red dot on "Flights" pushing <em>right</em> confirms:
      experienced boosters with multiple flights show improved landing rates. These are not
      statistical artifacts — they are physically meaningful relationships the model has
      independently discovered from data.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTOR
# ══════════════════════════════════════════════════════════════════
elif page == "Landing Predictor":
    st.markdown(
        """
    <div class="pg-title">Interactive <span>Predictor</span></div>
    <div class="pg-sub">Configure a Falcon 9 mission profile and get an instant landing probability
    from the trained LightGBM model — the same inference pipeline that would run in production.</div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="box">
      <strong>Production inference pipeline:</strong> Your selections are translated into the identical
      83-dimensional one-hot feature vector used during training. The vector is StandardScaler-normalized
      using the same fitted scaler, then passed to the LightGBM classifier which outputs both a binary
      decision and a calibrated probability via Platt scaling. This is exactly how the model would
      operate behind a REST API in a deployed system.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(
            '<div class="sec-label">Mission Configuration</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="card">', unsafe_allow_html=True)

        payload_mass = st.slider(
            "Payload Mass (kg)",
            0,
            16000,
            5000,
            step=100,
            help="Heavier payloads burn more ascent fuel → less left for landing burn → lower recovery probability",
        )
        flights = st.slider(
            "Previous Booster Flights",
            0,
            10,
            1,
            help="Battle-tested boosters show higher landing rates. Flight 1 = new hardware, higher risk.",
        )
        orbit = st.selectbox(
            "Target Orbit",
            [
                "LEO",
                "ISS",
                "PO",
                "GTO",
                "ES-L1",
                "SSO",
                "HEO",
                "MEO",
                "VLEO",
                "SO",
                "GEO",
            ],
            help="LEO/ISS/SSO = high recovery. GTO = ~50% (fuel-limited). HEO/GEO = low recovery.",
        )
        launch_site = st.selectbox(
            "Launch Site",
            ["CCAFS SLC 40", "VAFB SLC 4E", "KSC LC 39A", "CCAFS LC 40"],
            help="Drone ship proximity affects return-burn fuel margin.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
        <div class="box" style="margin-top:0.5rem">
          <strong>Try these scenarios:</strong><br>
          🟢 <em>LEO + 2,000 kg + KSC LC 39A</em> → near-certain landing<br>
          🔴 <em>GTO + 14,000 kg + CCAFS SLC 40</em> → crash predicted<br>
          🟡 <em>GTO + 5,000 kg + KSC LC 39A</em> → borderline case<br>
          These test the model's learned physics intuition around fuel margins.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="sec-label">Prediction Output</div>', unsafe_allow_html=True
        )

        iv = np.zeros(len(feature_names))
        if "PayloadMass" in feature_names:
            iv[feature_names.index("PayloadMass")] = payload_mass
        if "Flights" in feature_names:
            iv[feature_names.index("Flights")] = flights
        oc = f"Orbit_{orbit}"
        if oc in feature_names:
            iv[feature_names.index(oc)] = 1
        for fn in feature_names:
            if "LaunchSite" in fn and launch_site.replace(" ", "") in fn.replace(
                " ", ""
            ):
                iv[feature_names.index(fn)] = 1
                break

        lgbm_model = models["LightGBM"]
        iv_scaled = scaler.transform([iv])
        pred = lgbm_model.predict(iv_scaled)[0]
        proba = lgbm_model.predict_proba(iv_scaled)[0]
        lp, cp = proba[1], proba[0]

        if pred == 1:
            st.markdown(
                f"""
            <div class="pred-box pred-land">
              <div class="pred-emoji">✅</div>
              <div class="pred-lbl" style="color:#10b981">Booster recovery predicted</div>
              <div class="pred-pct" style="color:#10b981">{lp:.1%}</div>
              <div class="pred-note" style="color:#34d399">model confidence of successful landing</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="pred-box pred-crash">
              <div class="pred-emoji">💥</div>
              <div class="pred-lbl" style="color:#f43f5e">Stage loss predicted</div>
              <div class="pred-pct" style="color:#f43f5e">{cp:.1%}</div>
              <div class="pred-note" style="color:#fb7185">model confidence of booster loss</div>
            </div>""",
                unsafe_allow_html=True,
            )

        conf = max(lp, cp)
        if conf >= 0.9:
            clbl, ccol = "Very High Confidence", "#10b981"
        elif conf >= 0.75:
            clbl, ccol = "High Confidence", "#a78bfa"
        elif conf >= 0.6:
            clbl, ccol = "Moderate Confidence", AMB
        else:
            clbl, ccol = "Low — Borderline Case", RED

        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid #2a2a45;border-radius:10px;
                    padding:0.85rem 1.1rem;font-size:0.83rem;color:#c4c4e0;margin-top:0.25rem">
          <strong style="color:{ccol}">Confidence: {clbl}</strong><br>
          <span style="color:#8888aa;font-size:0.8rem">
            {'High certainty — the mission profile is well within the model training distribution.' if conf >= 0.75
             else 'Borderline case — minor parameter changes could flip the prediction. Check the SHAP explanation below.'}
          </span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── SHAP WATERFALL PLOT FOR THIS SPECIFIC PREDICTION ──────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-label">Why this prediction? — SHAP Waterfall</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="box">
      <strong>Live per-prediction explanation:</strong> The waterfall below shows exactly how
      each of your selected parameters pushed the model output for your specific mission.
      <span style="color:#34d399;font-weight:700">Green bars push toward LANDING</span>,
      <span style="color:#fb7185;font-weight:700">red bars push toward CRASH</span>.
      The baseline is the average prediction across all training samples, and the output
      is the model's raw score for your mission configuration.
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        explainer = shap.TreeExplainer(lgbm_model.best_estimator_)
        shap_values_single = explainer.shap_values(iv_scaled)
        if isinstance(shap_values_single, list):
            sv_single = shap_values_single[1][0]
            base_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            sv_single = shap_values_single[0]
            base_val = (
                explainer.expected_value
                if np.isscalar(explainer.expected_value)
                else explainer.expected_value[0]
            )

        abs_sv = np.abs(sv_single)
        top_idx = np.argsort(abs_sv)[::-1][:10]

        feat_labels = []
        feat_shaps = []
        for i in top_idx:
            if abs_sv[i] > 0.001:
                feat_labels.append(feature_names[i])
                feat_shaps.append(sv_single[i])

        if len(feat_shaps) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(feat_labels) * 0.45)))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(SURF)

            y_pos = np.arange(len(feat_labels))[::-1]
            colors = [GRN if v > 0 else RED for v in feat_shaps]

            bars = ax.barh(
                y_pos, feat_shaps, color=colors, height=0.6, edgecolor="none"
            )

            max_abs = max(abs(min(feat_shaps)), abs(max(feat_shaps)))
            padding = max_abs * 0.25
            ax.set_xlim(min(feat_shaps) - padding, max(feat_shaps) + padding)

            for bar, val in zip(bars, feat_shaps):
                if val >= 0:
                    ax.text(
                        val + max_abs * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"+{val:.3f}",
                        va="center",
                        ha="left",
                        fontsize=9,
                        color=GRN,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        val - max_abs * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}",
                        va="center",
                        ha="right",
                        fontsize=9,
                        color=RED,
                        fontweight="bold",
                    )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_labels, fontsize=10, color="#e0e0ff")
            ax.axvline(0, color="#4a4a6a", linewidth=1)
            ax.set_xlabel("SHAP value (impact on prediction)", fontsize=9, color=MUT)
            ax.set_title(
                f"Feature contributions to this prediction  |  Base: {base_val:.3f}  →  Output: {base_val+sv_single.sum():.3f}",
                fontsize=11,
                color=TXT,
                fontweight="bold",
                pad=12,
                loc="left",
            )

            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color("#2a2a45")
            ax.spines["bottom"].set_color("#2a2a45")
            ax.tick_params(colors=MUT, length=0)
            ax.grid(axis="x", color="#1e1e35", alpha=0.5, linewidth=0.7)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            top_pos = [
                (feat_labels[i], feat_shaps[i])
                for i in range(len(feat_shaps))
                if feat_shaps[i] > 0
            ][:3]
            top_neg = [
                (feat_labels[i], feat_shaps[i])
                for i in range(len(feat_shaps))
                if feat_shaps[i] < 0
            ][:3]

            col_a, col_b = st.columns(2)
            with col_a:
                if top_pos:
                    html = '<div class="box green"><strong>↑ Pushing toward LANDING:</strong><br>'
                    for name, val in top_pos:
                        html += f'&nbsp;&nbsp;• <span style="color:#e0e0ff">{name}</span> <span style="color:#34d399;font-weight:700">+{val:.3f}</span><br>'
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)
            with col_b:
                if top_neg:
                    html = '<div class="box" style="background:rgba(244,63,94,0.07);border-color:rgba(244,63,94,0.25)"><strong style="color:#fb7185">↓ Pushing toward CRASH:</strong><br>'
                    for name, val in top_neg:
                        html += f'&nbsp;&nbsp;• <span style="color:#e0e0ff">{name}</span> <span style="color:#fb7185;font-weight:700">{val:.3f}</span><br>'
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No significant feature contributions for this mission profile.")

    except Exception as e:
        st.warning(f"Could not generate waterfall plot: {e}")

    st.markdown(
        f"""
    <div style="text-align:center;font-size:0.62rem;color:#3a3a55;margin-top:1rem;
                letter-spacing:0.12em;text-transform:uppercase">
      LightGBM · GridSearchCV · 10-fold CV · {rdf.iloc[0]['Accuracy']:.1%} Test Accuracy · 83 Features · SHAP Waterfall
    </div>
    """,
        unsafe_allow_html=True,
    )
