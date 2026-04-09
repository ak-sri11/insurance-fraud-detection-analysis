-- ============================================================
-- Project Title:
-- Insurance Claims Fraud Detection & Risk Segmentation (SQL)

-- ============================================================
-- Problem Statement:
-- Insurance companies face significant financial losses due to fraudulent claims.
-- Detecting fraud is challenging because it does not depend on a single factor,
-- but rather a combination of behavioral and financial patterns.

-- ============================================================
-- Objective:
-- Identify fraudulent claim patterns using behavioral indicators
-- Build a rule-based fraud scoring model
-- Segment claims into Low, Medium, and High risk categories
-- Enable prioritization of high-risk claims for investigation

-- ============================================================
Create Database fraud_analysis;
USE fraud_analysis;

select count(*) from claims_data_v2;
-- ============================================================
-- Create New Table
CREATE TABLE claims_analysis AS
SELECT 
    *,
    
    -- 1. Reporting Delay
    DATEDIFF(Intimation_Date, Loss_Date) AS Reporting_Lag,
    
    -- 2. Delay Flag
    CASE 
        WHEN DATEDIFF(Intimation_Date, Loss_Date) > 7 THEN 1 
        ELSE 0 
    END AS Delay_Flag,
    
    -- 3. High Claim Flag
    CASE 
        WHEN Claim_Amount / IDV > 0.75 THEN 1 
        ELSE 0 
    END AS High_Claim_Flag

FROM claims_data_v2;
-- ============================================================
-- Fraud Scoring & Risk Segmentation

-- Add Fraud Score + Risk Signal
ALTER TABLE claims_analysis
ADD COLUMN Fraud_Score INT,
ADD COLUMN Risk_Level VARCHAR(10);
-- Update Fraud Score
SET SQL_SAFE_UPDATES = 0;
UPDATE claims_analysis
SET Fraud_Score = 
    Delay_Flag +
    High_Claim_Flag +
    CASE WHEN Vehicle_Age > 5 THEN 1 ELSE 0 END;
-- Update Risk Level
UPDATE claims_analysis
SET Risk_Level = 
    CASE 
        WHEN Fraud_Score <= 1 THEN 'Low'
        WHEN Fraud_Score = 2 THEN 'Medium'
        ELSE 'High'
    END;

SELECT Claim_Number, Fraud_Score, Risk_Level
FROM claims_analysis
LIMIT 10;
-- ============================================================
-- KPI Dashboard & Fraud Rate Analysis

SELECT 
    COUNT(*) AS Total_Claims,
    SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) AS Fraud_Claims,
    ROUND(
        SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Fraud_Percentage
FROM claims_analysis;

/* 
Overall Fraud Concentration: 
Only 3.33% of total claims are high-risk, indicating that fraud is a rare but critical event.
Effective Risk Filtering:
Out of 1500 claims, only 50 are flagged as high-risk, showing that your model is not over-flagging.
Business Impact Insight:
A small fraction (3.33%) of claims drives potential fraud exposure, meaning investigation teams can focus resources efficiently on a limited set of cases.

*/
-- ============================================================
-- Risk Segmentation Analysis

SELECT 
    Risk_Level,
    COUNT(*) AS Total_Claims,
    SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) AS High_Risk_Count,
    ROUND(
        SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) 
        * 100.0 / SUM(COUNT(*)) OVER(), 2
    ) AS Pct_of_All_Claims
FROM claims_analysis
GROUP BY Risk_Level
ORDER BY 
    CASE 
        WHEN Risk_Level = 'Low' THEN 1
        WHEN Risk_Level = 'Medium' THEN 2
        ELSE 3
    END;
    
/*
Only 3.33% of total claims are classified as high-risk, indicating fraud is rare but concentrated.
The model effectively isolates risky cases, enabling focused and efficient investigation.
*/
-- ============================================================
-- Customer Claim Behavior Analysis

SELECT 
    Vehicle_Age,
    COUNT(*) AS Total_Claims,
    SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) AS Fraud_Claims,
    ROUND(
        SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Fraud_Percentage
FROM claims_analysis
GROUP BY Vehicle_Age
ORDER BY Vehicle_Age;

/*Fraud risk increases as vehicle age rises, especially beyond 5 years.
Older vehicles show higher fraud percentages compared to newer ones.
This suggests aging assets are more vulnerable to suspicious claims.
*/
-- ============================================================
-- Garage Risk Analysis

SELECT 
    Garage_ID,
    COUNT(*) AS Total_Claims,
    SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) AS Fraud_Claims,
    ROUND(
        SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Fraud_Percentage
FROM claims_analysis
GROUP BY Garage_ID
HAVING COUNT(*) > 10
ORDER BY Fraud_Percentage DESC;

/*Certain garages show significantly higher fraud percentages, indicating possible suspicious patterns or weak verification controls.
A small set of garages contributes disproportionately to fraud risk.
These garages should be prioritized for audit and investigation.
*/
-- ============================================================
-- High Value Claim Risk Analysis

SELECT 
    CASE 
        WHEN Claim_Amount / IDV <= 0.25 THEN 'Low'
        WHEN Claim_Amount / IDV <= 0.50 THEN 'Medium'
        WHEN Claim_Amount / IDV <= 0.75 THEN 'High'
        ELSE 'Very High'
    END AS Claim_Bucket,
    
    COUNT(*) AS Total_Claims,
    SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) AS Fraud_Claims,
    ROUND(
        SUM(CASE WHEN Risk_Level = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Fraud_Percentage
FROM claims_analysis
GROUP BY Claim_Bucket
ORDER BY Fraud_Percentage DESC;

/* Fraud is concentrated in the “Very High” claim bucket, indicating high claim-to-IDV ratio is a strong fraud signal.
Lower buckets show negligible fraud, proving this metric is effective for risk identification.
*/
-- ============================================================
-- Top Suspicious Claims (Investigation Queue)
-- “Which claims should be investigated immediately?”

WITH ranked_claims AS (
    SELECT 
        Claim_Number,
        Customer_ID,
        Claim_Amount,
        IDV,
        Fraud_Score,
        Risk_Level,
        ROW_NUMBER() OVER (
            ORDER BY Fraud_Score DESC, Claim_Amount DESC
        ) AS rank_no
	FROM claims_analysis
    WHERE Risk_Level = 'High'
)
SELECT *
FROM ranked_claims
WHERE rank_no <= 20;

/* Top-ranked claims show consistently high fraud scores and high claim amounts, making them priority investigation cases.
Ranking helps focus investigation on the most critical claims first, improving efficiency.
*/
-- ============================================================
-- FINAL SUMMARY
/*
In this project, I developed a SQL-based fraud detection system for insurance claims using a rule-based scoring approach.
I engineered key behavioral and financial indicators such as reporting delay, claim-to-IDV ratio, and vehicle age to identify suspicious patterns.
By combining these signals, I created a Fraud Score to segment claims into Low, Medium, and High risk categories.
The analysis demonstrated strong fraud concentration in high-risk segments, validating the effectiveness of the model.
I further built analytical layers including KPI monitoring, risk segmentation, garage-level risk analysis, and claim value-based risk assessment.
Finally, I designed an investigation query to prioritize high-risk claims, simulating real-world fraud investigation workflows.
*/
-- ============================================================
/*
Project: Insurance Claims Fraud Detection
Created By: Akshaya
Date: Mar 2026
*/

