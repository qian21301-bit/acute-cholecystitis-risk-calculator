# Acute Cholecystitis Risk Prediction Calculator

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |
| `gbm_model.joblib` | Trained GBM model (200 trees, depth=4) |
| `feature_meta.json` | Feature names, medians, and thresholds |
| `train_model.py` | Script to re-train the model (run once) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit theme configuration |
| `开发组.xlsx` / `外部验证.xlsx` | Source data (not required for deployment) |

## Deploy to Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   # Create a new public repo, then:
   git init
   git add app.py gbm_model.joblib feature_meta.json requirements.txt .streamlit/config.toml
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy**
   - Go to [https://share.streamlit.io](https://share.streamlit.io)
   - Click **"New app"**
   - Select your repo, branch `main`, file `app.py`
   - Click **"Deploy"**

3. **Get Public URL**
   - After deployment, Streamlit Cloud gives you a URL like:
     `https://YOUR_USER-YOUR_REPO-APP_NAME.streamlit.app`
   - The QR code will appear automatically at the bottom of the page

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- **Algorithm**: Gradient Boosting Machine (GBM)
- **Parameters**: 200 trees, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=50
- **Features**: 9 clinical indicators
- **Training**: 5-fold cross-validation on development set (n=2,365)
- **OOF AUC**: 0.990
- **Interpretation**: SHAP (TreeExplainer) for feature contribution explanation
