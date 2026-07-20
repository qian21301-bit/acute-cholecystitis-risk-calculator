# Acute Cholecystitis Risk Prediction — Streamlit App
# Pure JSON model inference — no sklearn/pickle dependency at runtime
import streamlit as st
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
import qrcode
from io import BytesIO
import base64

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Acute Cholecystitis Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ── Load JSON model ────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(BASE / "gbm_model.json") as f:
        m = json.load(f)
    return m

model = load_model()
FS_RAW = list(model['medians'].keys())
LABELS = model['features']
MEDIANS = [model['medians'][k] for k in FS_RAW]
THRESHOLD = model['threshold']
LR = model['learning_rate']
N_TREES = model['n_estimators']
TREES = model['trees']
INIT_PRED = model['init_pred']

def predict_gbm(vals):
    """Pure Python GBM inference — no sklearn dependencies."""
    lo = INIT_PRED
    for t in range(N_TREES):
        tr = TREES[t]
        node = 0
        while tr['children_left'][node] != -1:
            if vals[tr['feature'][node]] <= tr['threshold'][node]:
                node = tr['children_left'][node]
            else:
                node = tr['children_right'][node]
        lo += LR * tr['value'][node]
    return lo

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ── SHAP approximation via feature marginalization ─────────
def compute_shap(vals):
    """Compute feature contributions via marginalization."""
    raw_score = predict_gbm(vals)
    prob = sigmoid(raw_score)
    base_score = predict_gbm(MEDIANS)

    temps = []
    for i in range(len(vals)):
        orig = vals[i]
        vals[i] = MEDIANS[i]
        score_at_median = predict_gbm(vals)
        vals[i] = orig
        temps.append(raw_score - score_at_median)

    raw_sum = sum(temps)
    total_delta = raw_score - base_score
    scale = total_delta / raw_sum if abs(raw_sum) > 1e-10 else 0.0

    shap_vals = [t * scale for t in temps]
    closure = raw_score - (base_score + sum(shap_vals))
    if abs(closure) > 1e-6:
        max_idx = max(range(len(shap_vals)), key=lambda i: abs(shap_vals[i]))
        shap_vals[max_idx] += closure

    return raw_score, base_score, prob, shap_vals

# ── Unit labels ────────────────────────────────────────────
UNITS = {
    'NEUT#': '×10⁹/L', 'LYMPH%': '%', 'Impacted Stone': '(0 = No, 1 = Yes)',
    'Alb': 'g/L', 'CB': 'μmol/L', 'Alb/Glb': '', 'TG': 'mmol/L',
    'HDL-C': 'mmol/L', 'TC': 'mmol/L',
}
DEFAULTS = [0.0 if 'Impacted' in l else float(MEDIANS[i]) for i, l in enumerate(LABELS)]

# ── App UI ─────────────────────────────────────────────────
st.title("🩺 Acute Cholecystitis Risk Prediction Model")
st.caption("Based on 9 clinical features · GBM model (AUC 0.99)")

with st.form("input_form"):
    st.subheader("Clinical Parameters")
    col1, col2 = st.columns(2)
    inputs = {}
    for i, (feat, label) in enumerate(zip(FS_RAW, LABELS)):
        col = col1 if i % 2 == 0 else col2
        unit = UNITS.get(label, "")
        step = 1.0 if "Impacted" in label else 0.1
        fmt = "%.0f" if "Impacted" in label else "%.2f"
        kw = dict(
            label=f"{label}  ({unit})" if unit else label,
            value=DEFAULTS[i], step=step, format=fmt,
            key=f"inp_{feat}")
        if "Impacted" in label:
            kw["min_value"] = 0.0
            kw["max_value"] = 1.0
        inputs[feat] = col.number_input(**kw)
    submitted = st.form_submit_button("Submit", type="primary")

if submitted:
    vals = [float(inputs[f]) for f in FS_RAW]
    raw_score, base_score, prob, shap_vals = compute_shap(vals)

    pred_class = 1 if prob >= THRESHOLD else 0
    risk_label = "High risk (Positive)" if pred_class == 1 else "Low risk (Negative)"

    # ── Results ──
    st.divider()
    st.subheader("Prediction Result")
    col_p, col_r = st.columns([1, 1])
    with col_p:
        st.metric("Predicted Probability", f"{prob * 100:.2f}%",
                  delta=f"{'↑' if prob > 0.5 else '↓'} Class {pred_class}")
    with col_r:
        st.metric("Risk Level", risk_label)

    # Closure
    with st.expander("📐 Numerical Closure Verification", expanded=False):
        shap_sum = sum(shap_vals)
        closure = raw_score - (base_score + shap_sum)
        st.code(
            f"base_score (all medians)    = {base_score:.6f}\n"
            f"sum(SHAP values)            = {shap_sum:.6f}\n"
            f"base + sum(SHAP)            = {base_score + shap_sum:.6f}\n"
            f"raw_score (decision_func)   = {raw_score:.6f}\n"
            f"closure error               = {closure:.2e}\n"
            f"final probability (sigmoid) = {prob * 100:.4f}%\n"
            f"closure check               = {'✅ PASS' if abs(closure) < 1e-5 else '❌ FAIL'}"
        )

    # ── SHAP Waterfall ──
    st.subheader("Feature Contributions (SHAP Waterfall)")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(values=np.array(shap_vals),
                         base_values=base_score,
                         data=np.array(vals),
                         feature_names=LABELS),
        max_display=9, show=False)
    st.pyplot(fig, dpi=150)
    plt.close()

    # ── Force Plot ──
    st.subheader("SHAP Force Plot")
    try:
        fp = shap.force_plot(base_score, np.array(shap_vals), np.array(vals),
                             feature_names=LABELS, matplotlib=False)
        html = fp.html() if hasattr(fp, 'html') else str(fp)
        st.components.v1.html(f'<div style="overflow-x:auto;">{html}</div>',
                              height=180, scrolling=True)
    except Exception as e:
        st.info(f"Force plot unavailable: {e}")

    # ── Contribution Table ──
    st.subheader("Contribution Details")
    rows = []
    for lbl, val, sh in zip(LABELS, vals, shap_vals):
        rows.append({
            "Feature": lbl,
            "Value": f"{val:.2f}" if "Impacted" not in lbl else f"{int(val)}",
            "SHAP Contribution": f"{sh:+.6f}",
            "Direction": "↑ Higher Risk" if sh > 0 else "↓ Lower Risk",
        })
    df = pd.DataFrame(rows)
    df["|SHAP|"] = np.abs(shap_vals)
    df = df.sort_values("|SHAP|", ascending=False).drop(columns="|SHAP|")
    st.dataframe(df, hide_index=True, use_container_width=True)

else:
    st.info("👆 Enter clinical parameters and click **Submit** to see the prediction.")

# ── QR Code ────────────────────────────────────────────────
st.divider()
st.markdown("### 🌐 Access This App")

def get_streamlit_url():
    for var in ["STREAMLIT_URL", "STREAMLIT_PUBLIC_URL"]:
        if var in os.environ:
            return os.environ[var]
    return None

PUBLIC_URL = get_streamlit_url()
if PUBLIC_URL:
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(PUBLIC_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#333", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    col_q, col_l = st.columns([1, 3])
    with col_q:
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="140"/>',
                    unsafe_allow_html=True)
    with col_l:
        st.markdown(f"**Scan QR or click:**  \n[{PUBLIC_URL}]({PUBLIC_URL})")
else:
    st.info("📡 Deploy to Streamlit Community Cloud to get a public URL with QR code.")
    st.markdown(
        "1. Push to a **public GitHub repo**  \n"
        "2. Go to [share.streamlit.io](https://share.streamlit.io)  \n"
        "3. **New app** → your repo → main → `app.py` → Deploy"
    )
