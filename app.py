# Acute Cholecystitis Risk Prediction — Streamlit App
# ====================================================
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
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

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    gbm = joblib.load(BASE / "gbm_model.joblib")
    with open(BASE / "feature_meta.json") as f:
        meta = json.load(f)
    return gbm, meta

gbm, meta = load_model()
FS = meta["features"]
MEDIANS = meta["medians"]
THRESHOLD = meta["threshold"]

# Build SHAP explainer once
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(gbm)

explainer = load_explainer()
base_value = float(explainer.expected_value)

# ── Display labels ──────────────────────────────────────────
SHORT = {
    'NEUT# (中性粒细胞绝对值)10^9/L': 'NEUT#',
    'LYMPH% (淋巴细胞百分比)%': 'LYMPH%',
    '结石嵌顿': 'Impacted Stone',
    'Alb(白蛋白)g/L': 'Alb',
    'CB(结合胆红素)mol/L': 'CB',
    'Alb/Glb(白蛋白/球蛋白)': 'Alb/Glb',
    'TG (甘油三酯)mmol/L': 'TG',
    'HDL-C (高密度脂蛋白胆固醇)mmol/L': 'HDL-C',
    'TC (总胆固醇)mmol/L': 'TC',
}
UNITS = {
    'NEUT#': '×10⁹/L',
    'LYMPH%': '%',
    'Impacted Stone': '(0 = No, 1 = Yes)',
    'Alb': 'g/L',
    'CB': 'μmol/L',
    'Alb/Glb': '',
    'TG': 'mmol/L',
    'HDL-C': 'mmol/L',
    'TC': 'mmol/L',
}
LABELS = [SHORT[f] for f in FS]

# ── App UI ──────────────────────────────────────────────────
st.title("🩺 Acute Cholecystitis Risk Prediction Model")
st.caption("Based on 9 clinical features · GBM model (AUC 0.99)")

with st.form("input_form"):
    st.subheader("Clinical Parameters")
    col1, col2 = st.columns(2)
    inputs = {}
    for i, (feat, label) in enumerate(zip(FS, LABELS)):
        col = col1 if i % 2 == 0 else col2
        med = MEDIANS[feat]
        unit = UNITS.get(label, "")
        step = 1.0 if "Impacted" in label else 0.1
        fmt = "%.0f" if "Impacted" in label else "%.2f"
        default = 0.0 if "Impacted" in label else med
        kw = dict(label=f"{label}  ({unit})" if unit else label,
                  value=default, step=step, format=fmt,
                  key=f"inp_{feat}")
        if "Impacted" in label:
            kw["min_value"] = 0
            kw["max_value"] = 1
        inputs[feat] = col.number_input(**kw)

    submitted = st.form_submit_button("Submit", type="primary")

if submitted:
    # ── Build feature vector ──
    vals = np.array([inputs[f] for f in FS]).reshape(1, -1)

    # ── SHAP values ──
    sv = explainer.shap_values(vals)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    shap_vals = sv.flatten()

    # ── Prediction ──
    prob = float(gbm.predict_proba(vals)[0, 1])
    raw_score = float(np.dot(gbm.coef_.T, vals.T)) if hasattr(gbm, 'coef_') else 0
    # For GBM use decision_function
    raw_score = gbm.decision_function(vals)[0] if hasattr(gbm, 'decision_function') else float(np.log(prob / (1 - prob + 1e-10)))

    pred_class = 1 if prob >= THRESHOLD else 0
    risk_label = "High risk (Positive)" if pred_class == 1 else "Low risk (Negative)"

    # ── Closure check ──
    shap_sum = float(shap_vals.sum())
    closure = raw_score - (base_value + shap_sum)

    # ── Results ──
    st.divider()
    st.subheader("Prediction Result")

    col_p, col_r = st.columns([1, 1])
    with col_p:
        st.metric("Predicted Probability", f"{prob * 100:.2f}%",
                  delta=f"{'↑' if prob > 0.5 else '↓'} Class {pred_class}")
    with col_r:
        st.metric("Risk Level", risk_label,
                  help=f"Threshold ≥ {THRESHOLD}")

    # Closure validation
    with st.expander("📐 Numerical Closure Verification", expanded=False):
        st.code(
            f"base_value (SHAP expected)  = {base_value:.6f}\n"
            f"sum(SHAP values)            = {shap_sum:.6f}\n"
            f"base + sum(SHAP)            = {base_value + shap_sum:.6f}\n"
            f"raw_score (decision_func)   = {raw_score:.6f}\n"
            f"closure error               = {closure:.2e}\n"
            f"final probability (sigmoid) = {prob * 100:.4f}%\n"
            f"closure check               = {'✅ PASS' if abs(closure) < 1e-5 else '❌ FAIL'}"
        )

    # ── SHAP Waterfall Plot ──
    st.subheader("Feature Contributions (SHAP Waterfall)")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(values=shap_vals,
                         base_values=base_value,
                         data=vals[0],
                         feature_names=LABELS),
        max_display=9, show=False
    )
    st.pyplot(fig, dpi=150)
    plt.close()

    # ── SHAP Force Plot (HTML) ──
    st.subheader("SHAP Force Plot")
    try:
        fp = shap.force_plot(
            base_value, shap_vals, vals[0],
            feature_names=LABELS, matplotlib=False
        )
        if hasattr(fp, 'html'):
            shap_html = fp.html()
        else:
            shap_html = str(fp)
        st.components.v1.html(
            f'<div style="overflow-x:auto;">{shap_html}</div>',
            height=180, scrolling=True
        )
    except Exception as e:
        st.warning(f"Force plot not available: {e}")
        # Fallback: show static waterfall
        st.info("Using waterfall plot instead.")

    # ── Feature Contribution Table ──
    st.subheader("Contribution Details")
    rows = []
    for lbl, val, sh in zip(LABELS, vals[0], shap_vals):
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
    # Placeholder
    st.info("👆 Enter clinical parameters and click **Submit** to see the prediction.")

# ── QR Code & Deployed Link (bottom of page) ──
st.divider()
st.markdown("### 🌐 Access This App")

# Try to detect public URL from Streamlit Cloud
# Falls back to localhost notice if running locally
import socket

def get_streamlit_url():
    """Detect if running on Streamlit Cloud and return the public URL."""
    # Streamlit Cloud sets these env vars
    if "STREAMLIT_URL" in os.environ:
        return os.environ["STREAMLIT_URL"]
    if "STREAMLIT_PUBLIC_URL" in os.environ:
        return os.environ["STREAMLIT_PUBLIC_URL"]
    # On Streamlit Cloud, the app URL follows this pattern
    if "STREAMLIT_SHARE" in os.environ:
        return os.environ.get("STREAMLIT_URLS", "").split(",")[0]
    return None

PUBLIC_URL = get_streamlit_url()

if PUBLIC_URL:
    # Generate QR code
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(PUBLIC_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#333", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    col_qr, col_link = st.columns([1, 3])
    with col_qr:
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" width="140" alt="QR Code"/>',
            unsafe_allow_html=True,
        )
    with col_link:
        st.markdown(
            f"**Scan QR code or click the link to open:**  \n"
            f"[{PUBLIC_URL}]({PUBLIC_URL})",
        )
else:
    st.info(
        "📡 **Deploy to Streamlit Community Cloud** to get a public URL.  \n"
        "After deployment, a QR code will appear here automatically."
    )
    st.markdown(
        "#### Deploy now:\n"
        "1. Push this folder to a **public GitHub repo**  \n"
        "2. Go to [share.streamlit.io](https://share.streamlit.io)  \n"
        "3. Click **'New app'** → select your repo → branch → `app.py`  \n"
        "4. Deploy!  \n\n"
        "Once deployed, the public URL will appear above with a QR code."
    )
