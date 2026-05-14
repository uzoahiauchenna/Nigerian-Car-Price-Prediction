import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigerian Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
    }
    .main { background-color: #0D0D0D; color: #F5F5F5; }
    .stApp { background-color: #0D0D0D; }

    .metric-card {
        background: #1A1A1A;
        border: 1px solid #2A2A2A;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #00E5A0;
    }
    .metric-sub {
        font-size: 11px;
        color: #555;
        margin-top: 4px;
    }
    .price-result {
        background: linear-gradient(135deg, #00E5A0 0%, #00B8D4 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .price-result h2 {
        color: #0D0D0D !important;
        font-size: 2.5rem !important;
        margin: 0 !important;
    }
    .price-result p {
        color: #0D0D0D;
        margin: 0.3rem 0 0;
        font-size: 14px;
    }
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00E5A0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem;
        border-left: 3px solid #00E5A0;
        padding-left: 10px;
    }
    .cnn-badge {
        background: #1A1A1A;
        border: 1px solid #00E5A0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 13px;
        color: #00E5A0;
        margin-bottom: 1rem;
    }
    .warning-badge {
        background: #1A1A1A;
        border: 1px solid #FFB347;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 13px;
        color: #FFB347;
        margin-bottom: 1rem;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #AAA !important;
        font-size: 13px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00E5A0, #00B8D4);
        color: #0D0D0D;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 15px;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .sidebar-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: #00E5A0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_price_model():
    model   = joblib.load('car_price_model2.pkl')
    le_dict = joblib.load('encoders2.pkl')
    features = joblib.load('features2.pkl')
    return model, le_dict, features


@st.cache_resource
def load_cnn_model():
    try:
        from tensorflow.keras.models import load_model
        cnn   = load_model('best_cnn_model.keras')
        names = json.load(open('class_names.json'))
        return cnn, names
    except Exception:
        return None, None

price_model, le_dict, features = load_price_model()
cnn_model, class_names = load_cnn_model()
cnn_ready = cnn_model is not None


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🚗 Car Price Predictor💲 </div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Car Detection & Price", "📊 Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#555; line-height:1.8;'>
    <b style='color:#888'>Models loaded:</b><br>
    ✅ Price Predictor<br>
    """ + ("✅ CNN Classifier" if cnn_ready else "⏳ CNN (coming soon)") + """
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Car Detection & Price Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Car Detection & Price":

    st.markdown("# Car Detection & Price Prediction")
    st.markdown("<p style='color:#666; margin-top:-10px;'>Upload a car image or select manually to predict its market price in Nigeria</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Image upload & Make detection ───────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">Step 1 — Car Image</div>', unsafe_allow_html=True)

        predicted_make = None

        if cnn_ready:
            st.markdown('<div class="cnn-badge">✅ CNN model active — upload an image to auto-detect the car make</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload car image", type=['jpg', 'jpeg', 'png'])

            if uploaded:
                img = Image.open(uploaded).convert('RGB')
                st.image(img, use_column_width=True, caption="Uploaded image")

                # preprocess
                img_resized = img.resize((224, 224))
                img_array  = np.array(img_resized) / 255.0
                img_array  = np.expand_dims(img_array, axis=0)

                # predict
                with st.spinner("Detecting car make..."):
                    preds         = cnn_model.predict(img_array)
                    idx           = np.argmax(preds)
                    confidence    = preds[0][idx] * 100
                    predicted_make = class_names[idx]

                st.markdown(f"""
                <div class="cnn-badge">
                    🎯 Detected: <b>{predicted_make}</b> &nbsp;|&nbsp; Confidence: <b>{confidence:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

                # top 3 predictions
                top3_idx = np.argsort(preds[0])[::-1][:3]
                st.markdown("**Top 3 predictions:**")
                for i in top3_idx:
                    st.progress(int(preds[0][i] * 100), text=f"{class_names[i]}: {preds[0][i]*100:.1f}%")

        else:
            st.markdown('<div class="warning-badge">⏳ CNN model not loaded yet — select Make manually below</div>', unsafe_allow_html=True)
            st.image("https://via.placeholder.com/400x250/1A1A1A/555?text=Upload+Car+Image", use_column_width=True)
            predicted_make = st.selectbox(
                "Select Car Make manually",
                sorted(le_dict['Make'].classes_.tolist())
            )

    # ── RIGHT: Car details inputs ─────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">Step 2 — Car Details</div>', unsafe_allow_html=True)

        # if CNN detected make, show it locked — else let user pick
        if cnn_ready and predicted_make:
            st.info(f"Make auto-detected: **{predicted_make}**")
        elif not cnn_ready:
            pass  # already selected above
        else:
            predicted_make = st.selectbox(
                "Car Make",
                sorted(le_dict['Make'].classes_.tolist())
            )
        col_a,col_b = st.columns(2)
        with col_a:
            model_name = st.selectbox(
                "Car Model",
                sorted(le_dict['Model'].classes_.tolist())
            )
        with col_b:
           bought_condition = st.selectbox(
                'Bougth Condition',
                le_dict['Bought Condition'].classes_.tolist()
            ) 
        col_c, col_d= st.columns(2)
        with col_c:
            selling_condition = st.selectbox(
                'Selling Condition',
                le_dict['Selling Condition'].classes_.tolist()
            )
        with col_d:
            year = st.selectbox(
                "Year of Manufacture",
                list(range(2026, 1979, -1))
            )
        col_e, col_f = st.columns(2)
        with col_e:
            condition = st.selectbox(
                "Condition",
                le_dict['Condition'].classes_.tolist()
            )
        with col_f:
            mileage = st.number_input(
                "Mileage (km)",
                min_value=0,
                max_value=5000000,
                value=50000,
                step=5000
            )
        col_g, col_h = st.columns(2)   
        with col_g:
            engine = st.number_input(
                "Engine Size (cc)",
                min_value=800,
                max_value=6500,
                value=2000,
                step=100
            )
        with col_h:
            car_body = st.selectbox(
                "Car Body Type",
                sorted(le_dict['Car body'].classes_.tolist())
            )
        col_i, col_j = st.columns(2)    
        with col_i:
            colour = st.selectbox(
                "Colour",
                sorted(le_dict['Colour'].classes_.tolist())
            ) 
        with col_j:
            gear = st.selectbox(
                "Gear Type",
                le_dict['gear type'].classes_.tolist()
            )

             
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict button ─────────────────────────────────────────────────────
        if st.button("🚗 Predict Price"):
            if predicted_make is None:
                st.warning("Please upload an image or select a Make first.")
            else:

                new_car = pd.DataFrame({
                    'gear type'  : [gear],
                    'Make'       : [predicted_make],
                    'Model'      : [model_name],
                    'Colour'     : [colour],
                    'Condition'  : [condition],
                    'Car body'   : [car_body],
                    'Mileage'    : [mileage],
                    'Engine Size': [engine],
                    'Year of manufacture': [year],
                    'Selling Condition': [selling_condition],
                    'Bought Condition' : [bought_condition],
                })

                cat_cols = ['gear type', 'Make', 'Model',
                            'Colour', 'Condition', 'Car body', 'Bought Condition', 'Selling Condition']

                error = False
                for col in cat_cols:
                    try:
                        new_car[col] = le_dict[col].transform(new_car[col])
                    except ValueError as e:
                        st.error(f"Unrecognised value in **{col}**: {e}. Please select a different option.")
                        error = True
                        break

                if not error:
                    
                    # reorder columns to match training order exactly
                    new_car = new_car[features]
                    predicted_log   = price_model.predict(new_car)[0]
                    predicted_price = np.expm1(predicted_log)

                    st.markdown(f"""
                    <div class="price-result">
                        <p>Estimated Market Price</p>
                        <h2>₦{predicted_price:,.0f}</h2>
                        <p>{year} {predicted_make} {model_name} · {condition} · {mileage:,} km</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # price range estimate
                    low  = predicted_price * 0.85
                    high = predicted_price * 1.15
                    st.markdown(f"<p style='color:#666; font-size:13px; text-align:center;'>Estimated range: ₦{low:,.0f} — ₦{high:,.0f}</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("# Model Performance")
    st.markdown("<p style='color:#666; margin-top:-10px;'>Evaluation results for the Nigerian Car Price Prediction models</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Price model metrics ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price Prediction Model — Gradient Boosting</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test R²</div>
            <div class="metric-value">79.5%</div>
            <div class="metric-sub">variance explained</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test MAE</div>
            <div class="metric-value">₦785,950</div>
            <div class="metric-sub">avg prediction error</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test RMSE</div>
            <div class="metric-value">₦1,395,489</div>
            <div class="metric-sub">root mean sq error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    results_data = {
        'Model'      : ['Gradient Boosting', 'XGBoost', 'Random Forest', 'LGBMRegressor'],
        'Train R²'   : [0.9110, 0.8990, 0.8414, 0.8606],
        'Test R²'    : [0.7176, 0.7001, 0.6845, 0.6780],
        'Test MAE'   : ['₦1,032,191', '₦1,063,366', '₦1,082,252', '₦1,087,014'],
        'Test RMSE'  : ['₦2,185,083', '₦2,251,443', '₦2,309,573', '₦2,332,937'],
        'Winner'     : ['✅', '', '', '']
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance chart ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

    fi_data = {
        'Feature'   : ['Year of manufacture', 'Engine Size', 'Mileage', 'Model',
                        'Make', 'Colour', 'Condition', 'Car body', 'Selling Condition',
                        'Bought Condition', 'Seats', 'Number of Cylinders', 'gear type', 'fuel type'],
        'Importance': [0.4274, 0.2835, 0.0899, 0.0641,
                       0.0526, 0.0206, 0.0199, 0.0174, 0.0130,
                       0.0057, 0.0018, 0.0016, 0.0015, 0.0010]
    }
    fi_df = pd.DataFrame(fi_data).sort_values('Importance')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1A1A1A')
    ax.set_facecolor('#1A1A1A')
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                   color=['#00E5A0' if v > 0.1 else '#00B8D4' if v > 0.05 else '#2A5A4A'
                          for v in fi_df['Importance']])
    ax.set_xlabel('Importance Score', color='#888', fontsize=10)
    ax.set_title('Feature Importance — Gradient Boosting', color='#F5F5F5',
                 fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#888', labelsize=9)
    ax.spines[:].set_color('#2A2A2A')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── CNN performance (if ready) ─────────────────────────────────────────────
    st.markdown('<div class="section-header">CNN Image Classifier</div>', unsafe_allow_html=True)

    if cnn_ready:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">52%</div>
                <div class="metric-sub">image classification</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Architecture</div>
                <div class="metric-value" style='font-size:18px;'>EfficientNetB0</div>
                <div class="metric-sub">transfer learning</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Classes</div>
                <div class="metric-value">{}</div>
                <div class="metric-sub">car makes</div>
            </div>""".format(len(class_names)), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-badge">
            ⏳ CNN model is still being trained. Performance metrics will appear here once the model is ready.
        </div>
        """, unsafe_allow_html=True)
