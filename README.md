# Early Warning System for banking sector (SBI)

A Streamlit web application for predicting the default risk of a company using financial indicators and a pre-trained Random Forest model.

---

## Features

- **User Input:** Enter key financial indicators via an interactive form.
- **Prediction:** Get instant default risk prediction and risk score.
- **Report Export:** Download prediction reports as PDF or PNG.
- **Feature Importance:** Visualize which features most influence the model.

---

## Installation

1. **Clone or Download this Repository**

2. **Install Dependencies**

   First, ensure you have Python 3.7+ installed.

   Then, install all required packages using:

   ```sh
   pip install -r requirements.txt
   ```

3. **Place the Model File**

   Ensure `rf_model.pkl` (the trained Random Forest model) is in the same directory as `app.py`.

4. **Start the App**

   ```sh
   streamlit run app.py
   ```

---

## Usage

- Enter the required financial indicators.
- Click **Predict Default Risk**.
- View the prediction, risk score, and input summary.
- Download the report as PDF or PNG.
- Expand the feature importance section for model insights.

---

## Authors

Made by Navya and Sudheendra for the EWS Default Risk in SBI.

---

## License

For internal use only.
