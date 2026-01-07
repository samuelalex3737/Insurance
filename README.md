# 🏥 Advanced Insurance Claims Analytics Dashboard

## Overview

This is a comprehensive Streamlit dashboard for analyzing insurance death claims data with embedded Machine Learning algorithms including Classification, Clustering, Regression, and Association Rule Mining.

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the dashboard:**
```bash
streamlit run app.py
```

3. **Open in browser:** Navigate to `http://localhost:8501`

## 📊 Dashboard Features

### 1. Executive Overview
- **Key Metrics**: Total claims, approval rates, total liability
- **Interactive Sunburst Drill-Down**: Click to explore Status → Gender → Age Group
- **Waterfall Charts**: Visualize claim approval breakdown
- **Treemap**: Claims by Zone and Status with drill-down
- **Sankey Diagram**: Claim flow visualization (Medical Type → Early/Non-Early → Status)
- **Nested Donut Charts**: Multi-level pie chart visualization

### 2. Classification Analysis (Approved vs Repudiated)
**Models Available:**
- Random Forest
- Gradient Boosting
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes
- AdaBoost

**Features:**
- Confusion Matrix with detailed breakdown
- ROC and Precision-Recall Curves
- Feature Importance Analysis
- **Error Analysis with Deep Drill-Down**:
  - False Positive analysis (why claims were wrongly approved)
  - False Negative analysis (why claims were wrongly rejected)
  - Segment-wise error breakdown
- Learning Curves for diagnosing overfitting/underfitting
- Threshold adjustment for precision-recall tradeoff
- Model comparison across all algorithms

### 3. Clustering Analysis
**Algorithms:**
- K-Means (with Elbow method for optimal K)
- DBSCAN (density-based clustering)
- Hierarchical/Agglomerative Clustering

**Features:**
- PCA visualization (2D and 3D)
- Silhouette score analysis
- Cluster profiling (demographics, approval rates)
- Box plots for cluster characteristics

### 4. Regression Analysis (Sum Assured Prediction)
**Models:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

**Features:**
- R², RMSE, MAE metrics
- Actual vs Predicted scatter plots
- Residuals analysis
- Feature importance/coefficients
- Prediction waterfall breakdown

### 5. Association Rule Mining
**Algorithm:** Apriori

**Features:**
- Adjustable minimum support and confidence
- Interactive scatter plot (Support vs Confidence vs Lift)
- Top rules by lift
- Rules predicting claim outcomes highlighted
- Antecedent → Consequent rule display

### 6. Geographic Analysis
**Visualizations:**
- India map with state-wise metrics
- Region-wise breakdown (North, South, East, West, etc.)
- Sunburst drill-down: Region → State → Claim Status
- State performance heatmap

### 7. Deep Drill-Down Analysis
**Purpose:** Understand why model performance varies

**Features:**
- Segment-wise performance analysis
- False Positive deep dive with root cause analysis
- False Negative deep dive
- Model confidence analysis and calibration plot
- What-If Analysis: Adjust features and see prediction changes

## 📈 Key Insights from Data

Based on the analysis:
- **Total Claims:** 1,790
- **Approval Rate:** ~68%
- **Age Range:** 3 to 82 years
- **Key Factors for Claim Outcome:**
  - Early vs Non-Early claims
  - Medical vs Non-Medical policies
  - Geographic zone
  - Age of policy holder

## 🔧 Technical Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly (interactive charts)
- **ML:** Scikit-learn
- **Association Rules:** MLxtend
- **Data Processing:** Pandas, NumPy

## 📁 File Structure

```
insurance_dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Insurance.csv       # Dataset
└── README.md          # This file
```

## 💡 Tips for Use

1. **Classification**: Start with Random Forest, then compare with other models
2. **Drill-Down**: Click on any segment in sunburst/treemap to explore further
3. **Error Analysis**: Use the Deep Drill-Down tab to understand misclassifications
4. **Threshold Tuning**: Adjust the classification threshold to balance precision/recall
5. **Association Rules**: Lower minimum support for rare but important patterns

## 🎯 Business Use Cases

1. **Risk Assessment**: Identify high-risk claim patterns
2. **Fraud Detection**: Spot unusual claim characteristics
3. **Process Optimization**: Find zones/segments needing attention
4. **Policy Pricing**: Understand factors affecting claim outcomes
5. **Customer Segmentation**: Group policies by risk profiles

---
Built with ❤️ using Streamlit and Python
