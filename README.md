# Insurance Claims Fraud Detection & Risk Analysis
This project simulates a real-world fraud investigation workflow used by risk analytics teams.

## Project Overview
This project focuses on detecting potential fraud in insurance claims using a combination of **Python-based analysis** and **SQL-driven business logic**.

The objective was to simulate a real-world fraud detection workflow by combining behavioral indicators, financial signals, and risk segmentation to identify suspicious claims efficiently.

---

## Objectives
- Identify potentially fraudulent insurance claims  
- Build a rule-based fraud scoring system  
- Segment claims into risk categories (Low, Medium, High)  
- Prioritize high-risk claims for investigation  
- Simulate real-world uncertainty in fraud detection  

---

## Dataset
- Synthetic dataset designed to mimic real-world insurance scenarios  
- Key features include:
  - Claim Amount  
  - Incident Date & Claim Date  
  - Policy Holder ID  
  - Claim Type  
  - Insured Declared Value (IDV)  

---

## Project Structure

```
insurance-fraud-detection-analysis/
│
├── data/
│   └── claims_data_v2.csv
│
├── python/
│   └── fraud_analysis.ipynb
│
├── sql/
│   └── fraud_analysis.sql
│
├── outputs/
│   └── (charts, results, screenshots)
│
└── README.md
```

## Approach

### Data Preparation (Python)
- Handled missing values and inconsistencies  
- Created structured dataset for analysis  

## Workflow Integration

- Input: Raw claims dataset with customer, claim, and vehicle details  
- Processing: Data cleaning, feature engineering, and fraud scoring using SQL and Python  
- Output: Risk-segmented claims and prioritized high-risk cases for investigation

### Feature Engineering
- **Reporting Lag** → Delay between incident and claim  
- **Claim Ratio** → Claim Amount vs IDV  
- **Claim Frequency** → Multiple claims per customer  

---

## Fraud Detection Logic

Developed rule-based indicators:

- Late Reporting  
- High Claim Amount  
- High Claim-to-IDV Ratio  
- Frequent Claims  
- Suspicious Patterns  

These signals were combined into:

### Fraud Score
A composite score representing likelihood of fraud  

### Risk Segmentation
- Low Risk  
- Medium Risk  
- High Risk  

---

## SQL Analysis 

SQL was used to simulate real-world analytics:

- KPI Dashboard (Total Claims, Fraud %, High-Risk %)  
- Risk Segmentation Distribution  
- Vehicle Age vs Fraud Behavior  
- Garage-wise Fraud Concentration  
- High-Value Claim Risk Analysis  
- Investigation Queue (Top suspicious claims using ranking)

---

## Key Insights

- Fraud is highly concentrated in a **small subset (~3%) of claims**  
- High claim-to-IDV ratio is a strong fraud indicator  
- Delayed reporting increases fraud likelihood  
- Certain garages show higher fraud concentration  
- Risk segmentation effectively isolates high-impact cases  

---

## Real-World Simulation

To mimic real-world complexity:

- Introduced controlled noise in fraud scoring  
- Simulated false positives & false negatives  
- Avoided overly deterministic patterns  

---

## Business Impact

- Reduced investigation scope to a focused ~3% high-risk segment, minimizing manual review effort
- Enabled targeted fraud detection instead of manual review  
- Highlighted key fraud drivers for decision-making  
- Improved efficiency in prioritizing suspicious claims  

---

## Tools & Technologies

- Python (Pandas, NumPy, Matplotlib, Seaborn)  
- SQL (MySQL)  
- Jupyter Notebook  

---

## Key Learning

This project demonstrates how combining **data analysis, business rules, and realistic assumptions** can create an effective fraud detection system aligned with real-world operations.

---

## Author
Akshaya V.S.  
📅 Mar 2026
