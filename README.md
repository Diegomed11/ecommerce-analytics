# 🚀 E-Commerce Analytics Suite & Demand Forecasting

A full-stack Business Intelligence application that transforms raw transactional data into actionable strategic insights. This project implements an end-to-end data pipeline: from **SQL extraction** to **Machine Learning modeling** and **Interactive Visualization**.

## 📋 Project Overview
This tool was designed to solve two main business problems in the retail sector:
1.  **Inventory Optimization:** Predicting daily sales to prevent stock-outs or overstocking.
2.  **Customer Retention:** Identifying VIP and at-risk customers using behavioral segmentation.

## 🛠️ Tech Stack & Tools
* **Data Engineering:** PostgreSQL, SQL (Window Functions, CTEs, Aggregations), SQLAlchemy.
* **Machine Learning:** Python, Scikit-Learn (Random Forest Regressor, K-Means Clustering).
* **Data Analysis:** Pandas, NumPy, RFM Analysis (Recency, Frequency, Monetary).
* **Visualization:** Streamlit (Web App), Plotly (Interactive Charts), Seaborn.

## 🧠 Key Features

### 1. Demand Forecasting (Time-Series)
* Implemented a **Random Forest Regressor** to predict future daily sales.
* **Feature Engineering:** Created Lag features (t-1, t-7) and Rolling Windows (7-day moving averages) to capture weekly seasonality and trends.
* **Result:** Visual comparison between predicted values and real historical data.

### 2. Customer Segmentation (Unsupervised Learning)
* Applied **K-Means Clustering** to segment customers based on purchasing behavior.
* **RFM Methodology:** Calculated Recency, Frequency, and Monetary value for each user via complex SQL queries.
* **Insights:** Identification of "VIP Whales" (High Spend, High Frequency) vs. "Churn Risk" users.

### 3. Interactive Dashboard
* Real-time filtering and parameter tuning (e.g., changing the number of clusters or forecast days).
* Dynamic plots using **Plotly** for granular data exploration.

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ecommerce-analytics.git](https://github.com/your-username/ecommerce-analytics.git)
    cd ecommerce-analytics
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Database Configuration:**
    * Ensure you have PostgreSQL installed.
    * Import the Olist E-Commerce dataset (or restore the `.sql` dump provided).
    * Update the connection string in `secrets.toml` or environment variables.

4.  **Run the Application:**
    ```bash
    streamlit run app_pro.py
    ```

* **Forecast Accuracy:** The model successfully captures weekly sales peaks.
* **Cluster Analysis:** Detected 3 distinct customer personas, allowing for targeted marketing campaigns.

---
*Developed by Diego Medina Medina - Data Science Student at ESCOM IPN*
