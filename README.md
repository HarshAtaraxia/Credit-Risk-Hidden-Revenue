# LendingClub Credit Risk Analysis: Unlocking $460M in Hidden Revenue

## 📌 Project Overview
This project leverages machine learning to optimize credit underwriting for a portfolio of **2.26 Million loan records**. The objective was to identify "Hidden Gems"—borrowers misclassified as high-risk by traditional grading systems who possess the behavioral profile of prime borrowers.

### Key Results
* **Annual Revenue Opportunity:** $460,071,296
* **Identified Loan Volume:** $2.46 Billion
* **Model Performance:** 91% Accuracy | 83% Recall on defaults

---

## 🛠️ Technical Workflow

### 1. Data Processing
* **Scale:** Processed 2.2M+ rows using memory downcasting (float64 to float32) to optimize cloud resource usage.
* **Leakage Prevention:** Scrubbed all post-origination variables to ensure the model only uses data available at the time of application.

### 2. Machine Learning
* **Algorithm:** XGBoost Classifier (Histogram-based).
* **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to improve default detection.
* **Feature Engineering:** Utilized Weight of Evidence (WoE) and Information Value (IV) to rank predictive power.

### 3. Strategy & Insights
* **Segmentation:** Targeted Grade D/E borrowers with risk scores lower than the Grade B (Prime) average.
* **Survival Analysis:** Implemented the Cox Proportional Hazards model to predict "Time-to-Default" and assess long-term profitability.

---

## 📊 Visualizations & Dashboards
The project includes a **Power BI Dashboard** that tracks:
* Revenue opportunity by loan grade.
* Risk-Reward distribution (Hidden Gems vs. Prime).
* Model sensitivity and Recall performance.

---

## 📁 Repository Structure
* `core_engine/`: End-to-end Python modeling scripts.
* `docs/`: Technical project reports and strategic white papers (PDF).
* `visuals/`: Screenshots of KPI dashboards and risk plots.
* `requirements.txt`: List of Python dependencies.

---

## 👤 Author
**Harsh Chaturvedi** Data Strategist | PGDM Business Analytics  
[LinkedIn](https://www.linkedin.com/in/harsh71) | [GitHub](https://github.com/HarshAtaraxia)
