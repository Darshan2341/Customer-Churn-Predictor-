# 🔮 Customer Churn Predictor

> An end-to-end Machine Learning web app that predicts customer churn — built for freshers to showcase real Data Science & ML skills.

## 🚀 Live Demo
👉 [Try it here](https://your-app-link.streamlit.app) ← replace after deploying

---

## 📸 App Preview

| Tab | Description |
|---|---|
| 📋 Data | View raw data & statistics |
| 📊 EDA | Interactive charts & correlation heatmap |
| 🤖 ML Model | Train model, see accuracy, ROC curve, confusion matrix |
| 🎯 Predict | Enter customer details → get instant churn prediction |
| 📥 Export | Download Excel workbook or Word report |

---

## ✨ Key Features

- 🤖 **3 ML Models** — Random Forest, Gradient Boosting, Logistic Regression
- 📊 **Interactive EDA** — Pie charts, histograms, box plots, heatmaps
- 📈 **Model Metrics** — Accuracy, ROC-AUC, Precision, Recall, F1 Score
- 🎯 **Live Prediction** — Gauge chart with churn risk percentage
- 📊 **Excel Export** — 5 formatted sheets with styled tables
- 📝 **Word Export** — Professional 5-section analysis report
- 🔍 **Auto Insights** — Business recommendations generated from data
- 📂 **Any CSV** — Auto-detects comma, semicolon, tab separators
- 💡 **Sample Data** — Built-in dataset, no upload needed to get started

---

## 📊 Excel Export — 5 Sheets

| Sheet | Contents |
|---|---|
| 📋 Raw Data | Full dataset with formatted table |
| 📊 Statistics | Mean, std, min, max for all columns |
| 🤖 Model Performance | Accuracy, AUC, Precision, Recall, F1 |
| 🏆 Feature Importance | Ranked list of features driving churn |
| 🎯 Predictions | Actual vs Predicted with correct/wrong flag |

---

## 📝 Word Report — 5 Sections

1. Model Performance Summary (with status ratings)
2. Top 10 Feature Importance table
3. Dataset Overview
4. Summary Statistics table
5. Data Sample (first 10 rows)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web app framework |
| Scikit-learn | ML models & metrics |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Plotly | Interactive charts |
| OpenPyXL | Excel file generation |
| Python-Docx | Word document generation |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Darshan2341/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📂 Project Structure

```
customer-churn-predictor/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🧠 ML Models Explained

| Model | Best For | Speed |
|---|---|---|
| Random Forest | Balanced accuracy & interpretability | Fast |
| Gradient Boosting | Highest accuracy | Medium |
| Logistic Regression | Simple baseline, explainable | Very Fast |

---

## 📊 Sample Dataset Features

The built-in sample data includes 600 customers with:

- `Age` — Customer age
- `Gender` — Male / Female
- `Tenure` — Months with company
- `MonthlyCharges` — Monthly bill amount
- `TotalCharges` — Total amount paid
- `Contract` — Month-to-month / One year / Two year
- `InternetService` — DSL / Fiber optic / No
- `TechSupport` — Yes / No
- `PaymentMethod` — Electronic check / Mailed check / etc.
- `NumSupportCalls` — Number of support calls made
- `Churn` — Target variable (Yes / No)

---

## 🎯 What This Project Demonstrates

For recruiters and interviewers, this project shows:

✅ End-to-end ML pipeline (data → model → deploy)  
✅ Data preprocessing & feature engineering  
✅ Model evaluation with multiple metrics  
✅ Interactive data visualization  
✅ File export (Excel + Word)  
✅ Clean UI/UX design  
✅ Python best practices  

---

## 👨‍💻 Author

**Darshan Pokale**
- 🐙 GitHub: [@Darshan2341](https://github.com/Darshan2341)
- 💼 LinkedIn: [Darshan Pokale](https://linkedin.com/in/your-profile) ← update this

---

## ⭐ Support

If this project helped you, please give it a ⭐ on GitHub — it helps a lot!
