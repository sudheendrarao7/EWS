import streamlit as st
import numpy as np
import joblib
import pandas as pd
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image, ImageDraw, ImageFont
import os

# Load model
model = joblib.load("rf_model.pkl")

st.set_page_config(page_title="EWS Default Predictor", layout="centered")
st.title("📊 Early Warning System for Default Prediction")
st.markdown("Enter the financial indicators below:")

# Input fields
cris_score = st.number_input("CRISIL Score", 0.0, 10.0, 5.0, 0.1)
net_profit_margin = st.number_input("Net Profit Margin", -1.0, 1.0, 0.05, 0.01)
icr = st.number_input("Interest Coverage Ratio (ICR)", 0.0, 100.0, 2.0, 0.1)
crilc_flag = st.selectbox("CRILC Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
sectoral_index = st.number_input("CRISIL Sectoral Index", 0.0, 10.0, 2.0, 0.1)
fund_based_limits = st.number_input("Fund Based Limits (%)", 0.0, 100.0, 50.0, 1.0)
tol_tnw = st.number_input("TOL/TNW", 0.0, 100.0, 3.0, 0.1)
tol_adj_tnw = st.number_input("TOL/Adj_TNW", 0.0, 100.0, 2.0, 0.1)
order_book_nw = st.number_input("Order Book / Net Worth", 0.0, 20.0, 3.0, 0.1)

input_data = np.array([[cris_score, net_profit_margin, icr, crilc_flag,
                        sectoral_index, fund_based_limits, tol_tnw,
                        tol_adj_tnw, order_book_nw]])

input_df = pd.DataFrame(input_data, columns=[
    'CRISIL_Score', 'Net_Profit_Margin', 'ICR', 'CRILC_Flag',
    'CRISIL_Sectoral_Index', 'Fund_Based_Limits', 'TOL_TNW',
    'TOL_Adj_TNW', 'Order_Book_Net_Worth'
])

if "result" not in st.session_state:
    st.session_state.result = None

# Prediction Button
if st.button("Predict Default Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    label = "Default" if prediction == 1 else "Non-Default"
    emoji = "⚠️" if prediction == 1 else "✅"
    
    # Store result in session state
    st.session_state.result = {
        "label": label,
        "prob": prob,
        "input_df": input_df
    }

# Show results if available
if st.session_state.result:
    result = st.session_state.result
    st.markdown("---")
    st.subheader("📌 Result")
    st.markdown(f"{'⚠️' if result['label'] == 'Default' else '✅'} **Prediction:** {result['label']}")
    st.markdown(f"📈 **Risk Score:** `{result['prob']:.2f}`")
    st.markdown("### 🔍 Input Summary")
    st.dataframe(result["input_df"])

    # Report export section
    st.markdown("### 🖨️ Download Prediction Report")
    export_format = st.selectbox("Choose file format", ["PDF", "Image (PNG)"])

    if st.button("Generate Report"):
        if export_format == "PDF":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                c = canvas.Canvas(tmp_file.name, pagesize=letter)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, 750, "EWS Default Prediction Report")
                c.setFont("Helvetica", 12)
                c.drawString(100, 720, f"Prediction: {result['label']}")
                c.drawString(100, 700, f"Risk Score: {result['prob']:.2f}")
                y_pos = 660
                for key, val in zip(result["input_df"].columns, input_data[0]):
                    c.drawString(100, y_pos, f"{key}: {val}")
                    y_pos -= 20
                c.save()
                with open(tmp_file.name, "rb") as f:
                    st.download_button("📄 Download PDF", data=f.read(),
                                       file_name="EWS_Report.pdf", mime="application/pdf")

        elif export_format == "Image (PNG)":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                img = Image.new("RGB", (600, 400), color="white")
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((20, 20), "EWS Default Prediction Report", font=font, fill="black")
                draw.text((20, 60), f"Prediction: {result['label']}", font=font, fill="black")
                draw.text((20, 80), f"Risk Score: {result['prob']:.2f}", font=font, fill="black")
                y = 120
                for key, val in zip(result["input_df"].columns, input_data[0]):
                    draw.text((20, y), f"{key}: {val}", font=font, fill="black")
                    y += 20
                img.save(tmp_file.name)
                with open(tmp_file.name, "rb") as f:
                    st.download_button("🖼️ Download PNG", data=f.read(),
                                       file_name="EWS_Report.png", mime="image/png")

# Feature importance
with st.expander("📈 View Model Info & Feature Importance"):
    importances = model.feature_importances_
    feature_names = [
        'CRISIL_Score', 'Net_Profit_Margin', 'ICR', 'CRILC_Flag',
        'CRISIL_Sectoral_Index', 'Fund_Based_Limits', 'TOL/TNW',
        'TOL/Adj_TNW', 'Order_Book_Net_Worth'
    ]
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    st.bar_chart(imp_df.set_index("Feature"))

st.markdown("---")
st.caption("Made by Navya and Sudheendra for the EWS Default Risk in SBI")
