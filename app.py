"""
Advanced Insurance Claims Analytics Dashboard
=============================================
Features:
- Classification (Approved/Repudiated prediction) with drill-down analysis
- Clustering (K-Means, DBSCAN, Hierarchical)
- Regression (Sum Assured prediction)
- Association Rule Mining (Apriori)
- Interactive drill-down charts
- Waterfall charts
- Geographic visualizations
- Confusion matrix deep analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score, roc_curve, auc,
                            silhouette_score, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json

# Page Configuration
st.set_page_config(
    page_title="Insurance Claims Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1e88e5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the insurance dataset"""
    df = pd.read_csv('/mnt/user-data/uploads/Insurance.csv')
    
    # Clean numeric columns (remove commas)
    df['SUM_ASSURED'] = df['SUM_ASSURED'].astype(str).str.replace(',', '').astype(float)
    df['PI_ANNUAL_INCOME'] = df['PI_ANNUAL_INCOME'].astype(str).str.replace(',', '').astype(float)
    
    # Create binary target variable
    df['CLAIM_STATUS'] = df['POLICY_STATUS'].apply(lambda x: 1 if 'Approved' in str(x) else 0)
    df['CLAIM_STATUS_TEXT'] = df['CLAIM_STATUS'].map({1: 'Approved', 0: 'Repudiated'})
    
    # Create age groups
    df['AGE_GROUP'] = pd.cut(df['PI_AGE'], bins=[0, 30, 45, 60, 75, 100], 
                            labels=['Young (0-30)', 'Middle (31-45)', 'Senior (46-60)', 
                                   'Elderly (61-75)', 'Very Elderly (76+)'])
    
    # Create income groups
    df['INCOME_GROUP'] = pd.cut(df['PI_ANNUAL_INCOME'], 
                                bins=[-1, 0, 100000, 300000, 500000, float('inf')],
                                labels=['No Income', 'Low (<1L)', 'Medium (1-3L)', 
                                       'High (3-5L)', 'Very High (>5L)'])
    
    # Create sum assured groups
    df['SUM_ASSURED_GROUP'] = pd.cut(df['SUM_ASSURED'],
                                     bins=[0, 100000, 300000, 500000, 1000000, float('inf')],
                                     labels=['<1L', '1-3L', '3-5L', '5-10L', '>10L'])
    
    # Region mapping for geographic visualization
    state_region_map = {
        'Himachal Pradesh': 'North', 'Punjab': 'North', 'Haryana': 'North',
        'Jammu And Kashmir': 'North', 'Delhi': 'North', 'Uttarakhand': 'North',
        'Uttar Pradesh': 'North', 'Chandigarh': 'North', 'Rajasthan': 'North',
        'Maharashtra': 'West', 'Gujarat': 'West', 'Goa': 'West',
        'Karnataka': 'South', 'Kerala': 'South', 'Tamilnadu': 'South',
        'Andhra Pradesh': 'South', 'Telangana': 'South',
        'West Bengal': 'East', 'Bihar': 'East', 'Jharkhand': 'East',
        'Orissa': 'East', 'Assam': 'Northeast', 'Chhattisgarh': 'Central',
        'Madhya Pradesh': 'Central'
    }
    df['REGION'] = df['PI_STATE'].map(state_region_map).fillna('Other')
    
    return df


def create_encoded_features(df):
    """Create encoded features for ML models"""
    df_encoded = df.copy()
    
    categorical_cols = ['PI_GENDER', 'ZONE', 'PAYMENT_MODE', 'EARLY_NON', 
                       'PI_OCCUPATION', 'MEDICAL_NONMED', 'PI_STATE', 'REGION']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + '_ENCODED'] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded, label_encoders


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Insurance Claims Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_and_preprocess_data()
    df_encoded, label_encoders = create_encoded_features(df)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/insurance.png", width=80)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select Analysis Module",
        ["📊 Executive Overview", 
         "🎯 Classification Analysis",
         "🔮 Clustering Analysis",
         "📈 Regression Analysis",
         "🔗 Association Rules",
         "🗺️ Geographic Analysis",
         "📉 Deep Drill-Down Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.info(f"""
    **Total Records:** {len(df):,}
    **Features:** {len(df.columns)}
    **Approved Claims:** {df['CLAIM_STATUS'].sum():,}
    **Repudiated Claims:** {(df['CLAIM_STATUS']==0).sum():,}
    """)
    
    if page == "📊 Executive Overview":
        executive_overview(df)
    elif page == "🎯 Classification Analysis":
        classification_analysis(df, df_encoded)
    elif page == "🔮 Clustering Analysis":
        clustering_analysis(df, df_encoded)
    elif page == "📈 Regression Analysis":
        regression_analysis(df, df_encoded)
    elif page == "🔗 Association Rules":
        association_rule_mining(df)
    elif page == "🗺️ Geographic Analysis":
        geographic_analysis(df)
    elif page == "📉 Deep Drill-Down Analysis":
        deep_drilldown_analysis(df, df_encoded)


def executive_overview(df):
    """Executive Overview Dashboard"""
    st.header("📊 Executive Overview Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_claims = len(df)
    approved = df['CLAIM_STATUS'].sum()
    repudiated = total_claims - approved
    approval_rate = (approved / total_claims) * 100
    total_sum_assured = df['SUM_ASSURED'].sum()
    
    with col1:
        st.metric("Total Claims", f"{total_claims:,}", delta=None)
    with col2:
        st.metric("Approved Claims", f"{approved:,}", delta=f"{approval_rate:.1f}%")
    with col3:
        st.metric("Repudiated Claims", f"{repudiated:,}", delta=f"{100-approval_rate:.1f}%")
    with col4:
        st.metric("Avg Sum Assured", f"₹{df['SUM_ASSURED'].mean():,.0f}")
    with col5:
        st.metric("Total Liability", f"₹{total_sum_assured/10000000:.2f} Cr")
    
    st.markdown("---")
    
    # Row 1: Sunburst Drill-Down and Waterfall
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Interactive Drill-Down: Claims by Status → Gender → Age Group")
        
        # Create hierarchical data for sunburst
        sunburst_df = df.groupby(['CLAIM_STATUS_TEXT', 'PI_GENDER', 'AGE_GROUP']).size().reset_index(name='Count')
        
        fig_sunburst = px.sunburst(
            sunburst_df,
            path=['CLAIM_STATUS_TEXT', 'PI_GENDER', 'AGE_GROUP'],
            values='Count',
            color='Count',
            color_continuous_scale='RdYlGn',
            title='Click to Drill Down: Status → Gender → Age'
        )
        fig_sunburst.update_layout(height=500)
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with col2:
        st.subheader("📊 Waterfall Chart: Claim Approval Breakdown")
        
        # Create waterfall data
        categories = ['Total Claims', 'Approved - Medical', 'Approved - Non-Medical', 
                     'Repudiated - Medical', 'Repudiated - Non-Medical', 'Final']
        
        approved_medical = len(df[(df['CLAIM_STATUS']==1) & (df['MEDICAL_NONMED']=='MEDICAL')])
        approved_nonmed = len(df[(df['CLAIM_STATUS']==1) & (df['MEDICAL_NONMED']=='NON MEDICAL')])
        repudiated_medical = len(df[(df['CLAIM_STATUS']==0) & (df['MEDICAL_NONMED']=='MEDICAL')])
        repudiated_nonmed = len(df[(df['CLAIM_STATUS']==0) & (df['MEDICAL_NONMED']=='NON MEDICAL')])
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Claims",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=categories,
            y=[total_claims, approved_medical, approved_nonmed, -repudiated_medical, -repudiated_nonmed, 0],
            textposition="outside",
            text=[f"{total_claims:,}", f"+{approved_medical}", f"+{approved_nonmed}", 
                  f"-{repudiated_medical}", f"-{repudiated_nonmed}", f"{approved:,}"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}}
        ))
        fig_waterfall.update_layout(title="Waterfall: From Total to Approved", height=500)
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Row 2: Treemap and Sankey
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌳 Treemap: Claims by Zone and Status")
        treemap_df = df.groupby(['ZONE', 'CLAIM_STATUS_TEXT']).agg({
            'POLICY_NO': 'count',
            'SUM_ASSURED': 'sum'
        }).reset_index()
        treemap_df.columns = ['Zone', 'Status', 'Count', 'Total Sum Assured']
        
        fig_treemap = px.treemap(
            treemap_df,
            path=['Zone', 'Status'],
            values='Count',
            color='Total Sum Assured',
            color_continuous_scale='Viridis',
            title='Treemap: Click to Explore Zones'
        )
        fig_treemap.update_layout(height=450)
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    with col2:
        st.subheader("🔄 Sankey Diagram: Claim Flow")
        
        # Create Sankey data
        sankey_data = df.groupby(['MEDICAL_NONMED', 'EARLY_NON', 'CLAIM_STATUS_TEXT']).size().reset_index(name='Count')
        
        # Define nodes
        all_nodes = list(sankey_data['MEDICAL_NONMED'].unique()) + \
                   list(sankey_data['EARLY_NON'].unique()) + \
                   list(sankey_data['CLAIM_STATUS_TEXT'].unique())
        all_nodes = list(dict.fromkeys(all_nodes))  # Remove duplicates while preserving order
        
        # Create source, target, value lists
        source = []
        target = []
        value = []
        
        for _, row in sankey_data.iterrows():
            source.append(all_nodes.index(row['MEDICAL_NONMED']))
            target.append(all_nodes.index(row['EARLY_NON']))
            value.append(row['Count'])
            
            source.append(all_nodes.index(row['EARLY_NON']))
            target.append(all_nodes.index(row['CLAIM_STATUS_TEXT']))
            value.append(row['Count'])
        
        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=["#3498db", "#9b59b6", "#f39c12", "#e74c3c", "#2ecc71", "#1abc9c"][:len(all_nodes)]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(200,200,200,0.5)"
            )
        ))
        fig_sankey.update_layout(title="Claim Flow: Medical Type → Early/Non-Early → Status", height=450)
        st.plotly_chart(fig_sankey, use_container_width=True)
    
    # Row 3: Time/Age Analysis with drill-down
    st.subheader("📈 Age Distribution Analysis with Drill-Down")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Nested Pie Chart (Donut with multiple rings)
        fig_nested = go.Figure()
        
        # Inner ring - Claim Status
        status_counts = df['CLAIM_STATUS_TEXT'].value_counts()
        fig_nested.add_trace(go.Pie(
            values=status_counts.values,
            labels=status_counts.index,
            hole=0.6,
            domain={'x': [0.2, 0.8], 'y': [0.2, 0.8]},
            marker_colors=['#2ecc71', '#e74c3c'],
            name='Status',
            textinfo='label+percent'
        ))
        
        # Outer ring - Gender breakdown
        gender_status = df.groupby(['CLAIM_STATUS_TEXT', 'PI_GENDER']).size().reset_index(name='Count')
        outer_labels = [f"{row['CLAIM_STATUS_TEXT']}-{row['PI_GENDER']}" for _, row in gender_status.iterrows()]
        outer_colors = ['#27ae60', '#229954', '#c0392b', '#a93226']
        
        fig_nested.add_trace(go.Pie(
            values=gender_status['Count'].values,
            labels=outer_labels,
            hole=0.75,
            domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
            marker_colors=outer_colors,
            name='Gender',
            textinfo='label+percent'
        ))
        
        fig_nested.update_layout(
            title='Nested Donut: Status (Inner) → Gender (Outer)',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_nested, use_container_width=True)
    
    with col2:
        # Age distribution by status
        fig_age = go.Figure()
        
        for status in ['Approved', 'Repudiated']:
            subset = df[df['CLAIM_STATUS_TEXT'] == status]
            fig_age.add_trace(go.Violin(
                x=[status] * len(subset),
                y=subset['PI_AGE'],
                name=status,
                box_visible=True,
                meanline_visible=True,
                fillcolor='#2ecc71' if status == 'Approved' else '#e74c3c',
                opacity=0.7
            ))
        
        fig_age.update_layout(
            title='Age Distribution by Claim Status',
            yaxis_title='Age',
            height=400
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_age_approved = df[df['CLAIM_STATUS']==1]['PI_AGE'].mean()
        avg_age_repudiated = df[df['CLAIM_STATUS']==0]['PI_AGE'].mean()
        st.markdown(f"""
        <div class="insight-box">
        <h4>📊 Age Analysis</h4>
        <p>Average age of <b>approved claims: {avg_age_approved:.1f} years</b></p>
        <p>Average age of <b>repudiated claims: {avg_age_repudiated:.1f} years</b></p>
        <p>Difference: {abs(avg_age_approved - avg_age_repudiated):.1f} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        top_zone = df.groupby('ZONE')['CLAIM_STATUS'].mean().idxmax()
        top_zone_rate = df.groupby('ZONE')['CLAIM_STATUS'].mean().max() * 100
        st.markdown(f"""
        <div class="insight-box">
        <h4>🏢 Zone Performance</h4>
        <p>Best performing zone: <b>{top_zone}</b></p>
        <p>Approval rate: <b>{top_zone_rate:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        early_approval = df[df['EARLY_NON']=='EARLY']['CLAIM_STATUS'].mean() * 100
        non_early_approval = df[df['EARLY_NON']=='NON EARLY']['CLAIM_STATUS'].mean() * 100
        st.markdown(f"""
        <div class="insight-box">
        <h4>⏰ Early vs Non-Early Claims</h4>
        <p>Early claim approval rate: <b>{early_approval:.1f}%</b></p>
        <p>Non-Early approval rate: <b>{non_early_approval:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)


def classification_analysis(df, df_encoded):
    """Classification Analysis with Deep Drill-Down"""
    st.header("🎯 Classification Analysis: Claim Approval Prediction")
    
    # Feature Selection
    st.sidebar.markdown("### Model Configuration")
    
    feature_cols = ['PI_AGE', 'SUM_ASSURED', 'PI_ANNUAL_INCOME', 
                   'PI_GENDER_ENCODED', 'ZONE_ENCODED', 'PAYMENT_MODE_ENCODED',
                   'EARLY_NON_ENCODED', 'MEDICAL_NONMED_ENCODED', 'REGION_ENCODED']
    
    available_features = [col for col in feature_cols if col in df_encoded.columns]
    
    selected_features = st.sidebar.multiselect(
        "Select Features",
        available_features,
        default=available_features[:6]
    )
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", 
         "Decision Tree", "KNN", "SVM", "Naive Bayes", "AdaBoost"]
    )
    
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features to proceed.")
        return
    
    # Prepare data
    X = df_encoded[selected_features].fillna(0)
    y = df_encoded['CLAIM_STATUS']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    model = models[model_choice]
    
    # Train model
    with st.spinner(f"Training {model_choice}..."):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display Metrics
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Accuracy", f"{accuracy:.2%}", 
                delta="Good" if accuracy > 0.75 else "Needs Improvement")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Confusion Matrix", "📈 ROC & PR Curves", "🎯 Feature Importance",
        "🔍 Error Analysis", "📉 Learning Curves"
    ])
    
    with tab1:
        st.subheader("Confusion Matrix Analysis")
        
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix Heatmap
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Repudiated', 'Approved'],
                y=['Repudiated', 'Approved'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig_cm.update_layout(title="Confusion Matrix", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Confusion Matrix Breakdown
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown("### Detailed Breakdown")
            
            metrics_df = pd.DataFrame({
                'Metric': ['True Negatives (TN)', 'False Positives (FP)', 
                          'False Negatives (FN)', 'True Positives (TP)',
                          'False Positive Rate', 'False Negative Rate',
                          'Specificity', 'Sensitivity'],
                'Value': [tn, fp, fn, tp, 
                         f"{fp/(fp+tn):.2%}", f"{fn/(fn+tp):.2%}",
                         f"{tn/(tn+fp):.2%}", f"{tp/(tp+fn):.2%}"],
                'Interpretation': [
                    'Correctly predicted Repudiated',
                    '⚠️ Predicted Approved but actually Repudiated',
                    '⚠️ Predicted Repudiated but actually Approved',
                    'Correctly predicted Approved',
                    'Rate of false alarms',
                    'Rate of missed approvals',
                    'True Negative Rate',
                    'True Positive Rate (Recall)'
                ]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visual breakdown
            fig_breakdown = go.Figure(go.Waterfall(
                name="Predictions",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=['Total Test', 'True Positives', 'True Negatives', 'Errors', 'Correct'],
                y=[len(y_test), tp, tn, -(fp+fn), 0],
                text=[len(y_test), f"+{tp}", f"+{tn}", f"-{fp+fn}", tp+tn],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                totals={"marker": {"color": "#3498db"}}
            ))
            fig_breakdown.update_layout(title="Prediction Waterfall", height=350)
            st.plotly_chart(fig_breakdown, use_container_width=True)
    
    with tab2:
        if y_pred_proba is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                            name=f'ROC (AUC = {roc_auc:.3f})',
                                            fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.3)'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                            name='Random', line=dict(dash='dash', color='gray')))
                fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                                     yaxis_title='True Positive Rate', height=400)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                # Precision-Recall Curve
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
                avg_precision = average_precision_score(y_test, y_pred_proba)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines',
                                           name=f'PR (AP = {avg_precision:.3f})',
                                           fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.3)'))
                fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall',
                                    yaxis_title='Precision', height=400)
                st.plotly_chart(fig_pr, use_container_width=True)
            
            # Threshold Analysis
            st.subheader("🎚️ Threshold Analysis")
            threshold = st.slider("Adjust Classification Threshold", 0.0, 1.0, 0.5, 0.01)
            
            y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
            cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Adjusted Accuracy", f"{accuracy_score(y_test, y_pred_adjusted):.2%}")
            col2.metric("Adjusted Precision", f"{precision_score(y_test, y_pred_adjusted, zero_division=0):.2%}")
            col3.metric("Adjusted Recall", f"{recall_score(y_test, y_pred_adjusted, zero_division=0):.2%}")
            col4.metric("Adjusted F1", f"{f1_score(y_test, y_pred_adjusted, zero_division=0):.2%}")
        else:
            st.info("ROC and PR curves are not available for this model type.")
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='Viridis',
                                title='Feature Importance')
                fig_imp.update_layout(height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col2:
                # Feature importance pie
                fig_pie = px.pie(importance_df, values='Importance', names='Feature',
                                title='Feature Importance Distribution')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        elif hasattr(model, 'coef_'):
            coef_df = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', ascending=True)
            
            fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                             color='Coefficient', color_continuous_scale='RdBu',
                             title='Feature Coefficients')
            fig_coef.update_layout(height=400)
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    with tab4:
        st.subheader("🔍 Error Analysis: Understanding Misclassifications")
        
        # Get misclassified samples
        test_indices = X_test.index
        error_mask = y_test != y_pred
        
        error_df = df.iloc[test_indices[error_mask]].copy()
        error_df['Predicted'] = ['Approved' if p == 1 else 'Repudiated' for p in y_pred[error_mask]]
        error_df['Actual'] = ['Approved' if a == 1 else 'Repudiated' for a in y_test[error_mask].values]
        error_df['Error_Type'] = error_df.apply(
            lambda x: 'False Positive' if x['Predicted'] == 'Approved' and x['Actual'] == 'Repudiated' 
            else 'False Negative', axis=1
        )
        
        if len(error_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Error distribution by type
                error_counts = error_df['Error_Type'].value_counts()
                fig_error_type = px.pie(values=error_counts.values, names=error_counts.index,
                                       title='Error Type Distribution',
                                       color_discrete_map={'False Positive': '#e74c3c', 'False Negative': '#f39c12'})
                st.plotly_chart(fig_error_type, use_container_width=True)
            
            with col2:
                # Error by age group
                error_age = error_df.groupby(['Error_Type', 'AGE_GROUP']).size().reset_index(name='Count')
                fig_error_age = px.bar(error_age, x='AGE_GROUP', y='Count', color='Error_Type',
                                      title='Errors by Age Group', barmode='group')
                st.plotly_chart(fig_error_age, use_container_width=True)
            
            # Drill-down into false positives
            st.markdown("### 🔴 False Positives Analysis (Predicted Approved, Actually Repudiated)")
            fp_df = error_df[error_df['Error_Type'] == 'False Positive']
            
            if len(fp_df) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fp_zone = fp_df['ZONE'].value_counts().head(5)
                    fig_fp_zone = px.bar(x=fp_zone.index, y=fp_zone.values, 
                                        title='FP by Zone (Top 5)')
                    st.plotly_chart(fig_fp_zone, use_container_width=True)
                
                with col2:
                    fp_early = fp_df['EARLY_NON'].value_counts()
                    fig_fp_early = px.pie(values=fp_early.values, names=fp_early.index,
                                         title='FP by Early/Non-Early')
                    st.plotly_chart(fig_fp_early, use_container_width=True)
                
                with col3:
                    fp_medical = fp_df['MEDICAL_NONMED'].value_counts()
                    fig_fp_med = px.pie(values=fp_medical.values, names=fp_medical.index,
                                       title='FP by Medical Type')
                    st.plotly_chart(fig_fp_med, use_container_width=True)
                
                st.markdown("**Characteristics of False Positives:**")
                st.dataframe(fp_df[['POLICY_NO', 'PI_AGE', 'SUM_ASSURED', 'ZONE', 'EARLY_NON', 
                                   'REASON_FOR_CLAIM', 'Actual']].head(10), use_container_width=True)
            
            # Drill-down into false negatives
            st.markdown("### 🟡 False Negatives Analysis (Predicted Repudiated, Actually Approved)")
            fn_df = error_df[error_df['Error_Type'] == 'False Negative']
            
            if len(fn_df) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fn_zone = fn_df['ZONE'].value_counts().head(5)
                    fig_fn_zone = px.bar(x=fn_zone.index, y=fn_zone.values,
                                        title='FN by Zone (Top 5)')
                    st.plotly_chart(fig_fn_zone, use_container_width=True)
                
                with col2:
                    fn_reason = fn_df['REASON_FOR_CLAIM'].value_counts().head(5)
                    fig_fn_reason = px.bar(x=fn_reason.index, y=fn_reason.values,
                                          title='FN by Claim Reason')
                    st.plotly_chart(fig_fn_reason, use_container_width=True)
                
                with col3:
                    st.markdown("**FN Statistics:**")
                    st.write(f"- Avg Age: {fn_df['PI_AGE'].mean():.1f}")
                    st.write(f"- Avg Sum Assured: ₹{fn_df['SUM_ASSURED'].mean():,.0f}")
                    st.write(f"- Most Common Zone: {fn_df['ZONE'].mode().iloc[0] if len(fn_df['ZONE'].mode()) > 0 else 'N/A'}")
        else:
            st.success("No misclassifications found! The model perfectly predicted all test samples.")
    
    with tab5:
        st.subheader("📉 Learning Curves Analysis")
        
        with st.spinner("Generating learning curves..."):
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy'
            )
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            test_mean = test_scores.mean(axis=1)
            test_std = test_scores.std(axis=1)
            
            fig_lc = go.Figure()
            
            # Training score
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=train_mean, mode='lines+markers',
                name='Training Score', line=dict(color='#3498db'),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            # Cross-validation score
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=test_mean, mode='lines+markers',
                name='Cross-Validation Score', line=dict(color='#e74c3c'),
                error_y=dict(type='data', array=test_std, visible=True)
            ))
            
            fig_lc.update_layout(
                title='Learning Curves',
                xaxis_title='Training Set Size',
                yaxis_title='Accuracy Score',
                height=450
            )
            st.plotly_chart(fig_lc, use_container_width=True)
            
            # Interpretation
            gap = train_mean[-1] - test_mean[-1]
            if gap > 0.1:
                st.warning(f"""
                **Diagnosis: High Variance (Overfitting)**
                - Gap between training and validation: {gap:.2%}
                - Suggestions: Reduce model complexity, add regularization, or get more data
                """)
            elif test_mean[-1] < 0.7:
                st.warning(f"""
                **Diagnosis: High Bias (Underfitting)**
                - Both scores are relatively low
                - Suggestions: Add more features, use more complex model, or reduce regularization
                """)
            else:
                st.success(f"""
                **Diagnosis: Good Fit**
                - Training and validation scores are close
                - Model generalizes well to unseen data
                """)
    
    # Model Comparison
    st.markdown("---")
    st.subheader("🏆 Model Comparison")
    
    if st.button("Compare All Models"):
        with st.spinner("Training all models..."):
            results = []
            for name, clf in models.items():
                try:
                    clf.fit(X_train_scaled, y_train)
                    pred = clf.predict(X_test_scaled)
                    results.append({
                        'Model': name,
                        'Accuracy': accuracy_score(y_test, pred),
                        'Precision': precision_score(y_test, pred),
                        'Recall': recall_score(y_test, pred),
                        'F1 Score': f1_score(y_test, pred)
                    })
                except Exception as e:
                    st.warning(f"Error training {name}: {str(e)}")
            
            results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
            
            with col2:
                fig_compare = px.bar(results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                                    x='Model', y='Score', color='Metric', barmode='group',
                                    title='Model Performance Comparison')
                fig_compare.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_compare, use_container_width=True)


def clustering_analysis(df, df_encoded):
    """Clustering Analysis"""
    st.header("🔮 Clustering Analysis: Policy Segmentation")
    
    # Feature selection for clustering
    st.sidebar.markdown("### Clustering Configuration")
    
    numeric_features = ['PI_AGE', 'SUM_ASSURED', 'PI_ANNUAL_INCOME']
    encoded_features = [col for col in df_encoded.columns if col.endswith('_ENCODED')]
    all_features = numeric_features + encoded_features
    
    selected_features = st.sidebar.multiselect(
        "Select Features for Clustering",
        all_features,
        default=['PI_AGE', 'SUM_ASSURED', 'PI_ANNUAL_INCOME']
    )
    
    algorithm = st.sidebar.selectbox(
        "Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical"]
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        return
    
    # Prepare data
    X = df_encoded[selected_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == "K-Means":
        st.subheader("K-Means Clustering")
        
        # Elbow method
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Optimal K Selection (Elbow Method)")
            
            inertias = []
            silhouettes = []
            K_range = range(2, 11)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
            
            fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
            fig_elbow.add_trace(
                go.Scatter(x=list(K_range), y=inertias, name="Inertia", mode='lines+markers'),
                secondary_y=False
            )
            fig_elbow.add_trace(
                go.Scatter(x=list(K_range), y=silhouettes, name="Silhouette", mode='lines+markers'),
                secondary_y=True
            )
            fig_elbow.update_layout(title="Elbow Method & Silhouette Score", height=350)
            fig_elbow.update_xaxes(title_text="Number of Clusters")
            fig_elbow.update_yaxes(title_text="Inertia", secondary_y=False)
            fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        with col2:
            n_clusters = st.slider("Select Number of Clusters", 2, 10, 4)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            df_clustered = df.copy()
            df_clustered['Cluster'] = clusters
            
            st.markdown(f"### Cluster Distribution")
            cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
            fig_dist = px.pie(values=cluster_counts.values, names=[f'Cluster {i}' for i in cluster_counts.index],
                             title=f'Distribution across {n_clusters} Clusters')
            st.plotly_chart(fig_dist, use_container_width=True)
    
    elif algorithm == "DBSCAN":
        st.subheader("DBSCAN Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 20, 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        
        with col2:
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            st.metric("Number of Clusters", n_clusters)
            st.metric("Noise Points", n_noise)
    
    else:  # Hierarchical
        st.subheader("Hierarchical Clustering")
        
        n_clusters = st.slider("Select Number of Clusters", 2, 10, 4)
        linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clusters = hierarchical.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
    
    # Visualization
    st.markdown("---")
    st.subheader("📊 Cluster Visualization")
    
    # PCA for visualization
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_clustered['PCA1'] = X_pca[:, 0]
        df_clustered['PCA2'] = X_pca[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(df_clustered, x='PCA1', y='PCA2', color='Cluster',
                                    title='Clusters in PCA Space',
                                    color_continuous_scale='Viridis',
                                    hover_data=['PI_AGE', 'SUM_ASSURED', 'CLAIM_STATUS_TEXT'])
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # 3D if we have 3 components
            if X_scaled.shape[1] >= 3:
                pca_3d = PCA(n_components=3)
                X_pca_3d = pca_3d.fit_transform(X_scaled)
                df_clustered['PCA3'] = X_pca_3d[:, 2]
                
                fig_3d = px.scatter_3d(df_clustered, x='PCA1', y='PCA2', z='PCA3',
                                       color='Cluster', title='3D Cluster View',
                                       hover_data=['PI_AGE', 'SUM_ASSURED'])
                fig_3d.update_layout(height=400)
                st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig_scatter = px.scatter(df_clustered, x=selected_features[0], y=selected_features[1],
                                color='Cluster', title='Cluster Visualization')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Cluster Profiles
    st.subheader("📋 Cluster Profiles")
    
    cluster_profiles = df_clustered.groupby('Cluster').agg({
        'PI_AGE': 'mean',
        'SUM_ASSURED': 'mean',
        'PI_ANNUAL_INCOME': 'mean',
        'CLAIM_STATUS': ['mean', 'count'],
        'POLICY_NO': 'count'
    }).round(2)
    
    cluster_profiles.columns = ['Avg Age', 'Avg Sum Assured', 'Avg Income', 
                                'Approval Rate', 'Total Claims', 'Count']
    cluster_profiles['Approval Rate'] = (cluster_profiles['Approval Rate'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(cluster_profiles, use_container_width=True)
    
    # Cluster characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.box(df_clustered, x='Cluster', y='PI_AGE', color='Cluster',
                        title='Age Distribution by Cluster')
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        fig_sum = px.box(df_clustered, x='Cluster', y='SUM_ASSURED', color='Cluster',
                        title='Sum Assured Distribution by Cluster')
        st.plotly_chart(fig_sum, use_container_width=True)
    
    # Approval rate by cluster
    cluster_approval = df_clustered.groupby('Cluster')['CLAIM_STATUS'].mean() * 100
    
    fig_approval = px.bar(x=cluster_approval.index, y=cluster_approval.values,
                         title='Claim Approval Rate by Cluster (%)',
                         labels={'x': 'Cluster', 'y': 'Approval Rate (%)'},
                         color=cluster_approval.values,
                         color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_approval, use_container_width=True)


def regression_analysis(df, df_encoded):
    """Regression Analysis for Sum Assured Prediction"""
    st.header("📈 Regression Analysis: Sum Assured Prediction")
    
    # Feature selection
    st.sidebar.markdown("### Regression Configuration")
    
    feature_cols = ['PI_AGE', 'PI_ANNUAL_INCOME', 'PI_GENDER_ENCODED', 
                   'ZONE_ENCODED', 'PAYMENT_MODE_ENCODED', 'EARLY_NON_ENCODED',
                   'MEDICAL_NONMED_ENCODED', 'CLAIM_STATUS']
    
    available_features = [col for col in feature_cols if col in df_encoded.columns]
    
    selected_features = st.sidebar.multiselect(
        "Select Features",
        available_features,
        default=['PI_AGE', 'PI_ANNUAL_INCOME', 'CLAIM_STATUS']
    )
    
    model_choice = st.sidebar.selectbox(
        "Select Regression Model",
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest Regressor"]
    )
    
    if len(selected_features) < 1:
        st.warning("Please select at least 1 feature.")
        return
    
    # Prepare data
    X = df_encoded[selected_features].fillna(0)
    y = df_encoded['SUM_ASSURED']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    from sklearn.ensemble import RandomForestRegressor
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    model = models[model_choice]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"₹{rmse:,.0f}")
    col3.metric("MAE", f"₹{mae:,.0f}")
    col4.metric("MSE", f"{mse:,.0f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted
        fig_scatter = px.scatter(x=y_test, y=y_pred, 
                                labels={'x': 'Actual Sum Assured', 'y': 'Predicted Sum Assured'},
                                title='Actual vs Predicted Sum Assured')
        fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                        y=[y_test.min(), y_test.max()],
                                        mode='lines', name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Residuals
        residuals = y_test - y_pred
        fig_residuals = px.histogram(residuals, nbins=50, 
                                    title='Residuals Distribution',
                                    labels={'value': 'Residual', 'count': 'Frequency'})
        fig_residuals.update_layout(height=400)
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis',
                        title='Feature Importance for Sum Assured Prediction')
        st.plotly_chart(fig_imp, use_container_width=True)
    
    elif hasattr(model, 'coef_'):
        st.subheader("📊 Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=True)
        
        fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                         color='Coefficient', color_continuous_scale='RdBu',
                         title='Feature Coefficients')
        st.plotly_chart(fig_coef, use_container_width=True)
    
    # Prediction waterfall
    st.subheader("📊 Prediction Breakdown (Sample)")
    
    sample_idx = st.slider("Select Sample Index", 0, len(X_test)-1, 0)
    sample = X_test.iloc[sample_idx:sample_idx+1]
    sample_scaled = scaler.transform(sample)
    
    if hasattr(model, 'coef_'):
        contributions = sample_scaled[0] * model.coef_
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(selected_features) + ["total"],
            x=["Base"] + selected_features + ["Prediction"],
            y=[model.intercept_] + list(contributions) + [0],
            text=[f"₹{model.intercept_:,.0f}"] + [f"₹{c:,.0f}" for c in contributions] + 
                 [f"₹{y_pred[sample_idx]:,.0f}"],
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        fig_waterfall.update_layout(title="Prediction Waterfall Breakdown", height=400)
        st.plotly_chart(fig_waterfall, use_container_width=True)


def association_rule_mining(df):
    """Association Rule Mining"""
    st.header("🔗 Association Rule Mining")
    
    st.markdown("""
    Discover hidden patterns and relationships in insurance claims data.
    Find which combinations of factors tend to occur together.
    """)
    
    # Prepare data for association rules
    # Create categorical bins
    df_assoc = df.copy()
    df_assoc['Age_Category'] = pd.cut(df['PI_AGE'], bins=[0, 40, 60, 100], 
                                      labels=['Young', 'Middle', 'Senior'])
    df_assoc['Sum_Category'] = pd.cut(df['SUM_ASSURED'], bins=[0, 200000, 500000, float('inf')],
                                      labels=['Low_Sum', 'Medium_Sum', 'High_Sum'])
    
    # Select columns for association mining
    assoc_cols = ['PI_GENDER', 'EARLY_NON', 'MEDICAL_NONMED', 'CLAIM_STATUS_TEXT',
                  'Age_Category', 'Sum_Category']
    
    selected_cols = st.multiselect(
        "Select Columns for Association Mining",
        assoc_cols,
        default=['PI_GENDER', 'EARLY_NON', 'MEDICAL_NONMED', 'CLAIM_STATUS_TEXT']
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
    
    # Create transactions
    transactions = []
    for _, row in df_assoc[selected_cols].iterrows():
        transaction = [f"{col}={str(row[col])}" for col in selected_cols if pd.notna(row[col])]
        transactions.append(transaction)
    
    # One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    with st.spinner("Mining association rules..."):
        try:
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                st.warning("No frequent itemsets found. Try lowering the minimum support.")
                return
            
            # Generate rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if len(rules) == 0:
                st.warning("No rules found. Try lowering the minimum confidence.")
                return
            
            # Display results
            st.subheader(f"📋 Found {len(rules)} Association Rules")
            
            # Format rules for display
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Sort by lift
            rules_sorted = rules.sort_values('lift', ascending=False)
            
            # Display top rules
            st.dataframe(
                rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20).style.format({
                    'support': '{:.3f}',
                    'confidence': '{:.3f}',
                    'lift': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Support vs Confidence scatter
                fig_scatter = px.scatter(rules_sorted.head(50), x='support', y='confidence',
                                        size='lift', color='lift',
                                        hover_data=['antecedents', 'consequents'],
                                        title='Support vs Confidence (size=lift)',
                                        color_continuous_scale='Viridis')
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Top rules by lift
                top_rules = rules_sorted.head(10)
                top_rules['rule'] = top_rules['antecedents'] + ' → ' + top_rules['consequents']
                
                fig_lift = px.bar(top_rules, x='lift', y='rule', orientation='h',
                                 color='confidence', color_continuous_scale='RdYlGn',
                                 title='Top 10 Rules by Lift')
                fig_lift.update_layout(height=400)
                st.plotly_chart(fig_lift, use_container_width=True)
            
            # Key insights
            st.subheader("💡 Key Insights from Association Rules")
            
            # Find rules related to claim status
            approval_rules = rules_sorted[rules_sorted['consequents'].str.contains('Approved|Repudiat')]
            
            if len(approval_rules) > 0:
                st.markdown("### Rules Predicting Claim Outcomes:")
                for _, rule in approval_rules.head(5).iterrows():
                    if 'Approved' in rule['consequents']:
                        st.success(f"✅ **{rule['antecedents']}** → **{rule['consequents']}** "
                                 f"(Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})")
                    else:
                        st.error(f"❌ **{rule['antecedents']}** → **{rule['consequents']}** "
                               f"(Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})")
        
        except Exception as e:
            st.error(f"Error in association rule mining: {str(e)}")
            st.info("Try adjusting the minimum support or selecting different columns.")


def geographic_analysis(df):
    """Geographic Analysis with India Map"""
    st.header("🗺️ Geographic Analysis")
    
    # State-wise analysis
    state_stats = df.groupby('PI_STATE').agg({
        'POLICY_NO': 'count',
        'CLAIM_STATUS': 'mean',
        'SUM_ASSURED': ['sum', 'mean']
    }).reset_index()
    state_stats.columns = ['State', 'Total Claims', 'Approval Rate', 'Total Sum Assured', 'Avg Sum Assured']
    state_stats['Approval Rate'] = state_stats['Approval Rate'] * 100
    
    # India state coordinates (approximate centroids)
    state_coords = {
        'Himachal Pradesh': (31.1048, 77.1734),
        'Punjab': (31.1471, 75.3412),
        'Haryana': (29.0588, 76.0856),
        'Jammu And Kashmir': (33.7782, 76.5762),
        'Delhi': (28.7041, 77.1025),
        'Uttarakhand': (30.0668, 79.0193),
        'Uttar Pradesh': (26.8467, 80.9462),
        'Rajasthan': (27.0238, 74.2179),
        'Maharashtra': (19.7515, 75.7139),
        'Gujarat': (22.2587, 71.1924),
        'Karnataka': (15.3173, 75.7139),
        'Kerala': (10.8505, 76.2711),
        'Tamilnadu': (11.1271, 78.6569),
        'Andhra Pradesh': (15.9129, 79.7400),
        'Telangana': (18.1124, 79.0193),
        'West Bengal': (22.9868, 87.8550),
        'Bihar': (25.0961, 85.3131),
        'Jharkhand': (23.6102, 85.2799),
        'Orissa': (20.9517, 85.0985),
        'Assam': (26.2006, 92.9376),
        'Chhattisgarh': (21.2787, 81.8661),
        'Madhya Pradesh': (22.9734, 78.6569),
        'Chandigarh': (30.7333, 76.7794),
        'Goa': (15.2993, 74.1240)
    }
    
    # Add coordinates
    state_stats['lat'] = state_stats['State'].map(lambda x: state_coords.get(x, (20, 78))[0])
    state_stats['lon'] = state_stats['State'].map(lambda x: state_coords.get(x, (20, 78))[1])
    
    # Metric selection
    metric = st.selectbox(
        "Select Metric to Visualize",
        ['Total Claims', 'Approval Rate', 'Total Sum Assured', 'Avg Sum Assured']
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Map visualization
        fig_map = px.scatter_geo(
            state_stats,
            lat='lat',
            lon='lon',
            size=metric,
            color=metric,
            hover_name='State',
            hover_data=['Total Claims', 'Approval Rate', 'Avg Sum Assured'],
            title=f'Geographic Distribution: {metric}',
            color_continuous_scale='Viridis',
            scope='asia'
        )
        
        fig_map.update_geos(
            visible=True,
            resolution=50,
            showcountries=True,
            countrycolor="Black",
            showsubunits=True,
            subunitcolor="Gray",
            center=dict(lat=22, lon=78),
            projection_scale=4
        )
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        # Top states table
        st.markdown(f"### Top States by {metric}")
        top_states = state_stats.nlargest(10, metric)[['State', metric]]
        st.dataframe(top_states, use_container_width=True)
    
    # Region-wise analysis
    st.markdown("---")
    st.subheader("📊 Region-wise Analysis")
    
    region_stats = df.groupby('REGION').agg({
        'POLICY_NO': 'count',
        'CLAIM_STATUS': 'mean',
        'SUM_ASSURED': 'sum'
    }).reset_index()
    region_stats.columns = ['Region', 'Claims', 'Approval Rate', 'Total Sum Assured']
    region_stats['Approval Rate'] = region_stats['Approval Rate'] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_region_claims = px.pie(region_stats, values='Claims', names='Region',
                                   title='Claims by Region', hole=0.4)
        st.plotly_chart(fig_region_claims, use_container_width=True)
    
    with col2:
        fig_region_approval = px.bar(region_stats, x='Region', y='Approval Rate',
                                     color='Approval Rate', color_continuous_scale='RdYlGn',
                                     title='Approval Rate by Region (%)')
        st.plotly_chart(fig_region_approval, use_container_width=True)
    
    with col3:
        fig_region_sum = px.bar(region_stats, x='Region', y='Total Sum Assured',
                               color='Total Sum Assured', color_continuous_scale='Viridis',
                               title='Total Liability by Region')
        st.plotly_chart(fig_region_sum, use_container_width=True)
    
    # Sunburst for geographic drill-down
    st.subheader("🎯 Geographic Drill-Down: Region → State → Status")
    
    geo_hierarchy = df.groupby(['REGION', 'PI_STATE', 'CLAIM_STATUS_TEXT']).size().reset_index(name='Count')
    
    fig_sunburst = px.sunburst(
        geo_hierarchy,
        path=['REGION', 'PI_STATE', 'CLAIM_STATUS_TEXT'],
        values='Count',
        color='Count',
        color_continuous_scale='RdYlGn',
        title='Click to Drill Down: Region → State → Claim Status'
    )
    fig_sunburst.update_layout(height=600)
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # State comparison heatmap
    st.subheader("🔥 State-wise Metrics Heatmap")
    
    state_metrics = df.groupby('PI_STATE').agg({
        'POLICY_NO': 'count',
        'CLAIM_STATUS': 'mean',
        'SUM_ASSURED': 'mean',
        'PI_AGE': 'mean'
    }).reset_index()
    state_metrics.columns = ['State', 'Claims', 'Approval Rate', 'Avg Sum', 'Avg Age']
    
    # Normalize for heatmap
    state_metrics_norm = state_metrics.copy()
    for col in ['Claims', 'Approval Rate', 'Avg Sum', 'Avg Age']:
        state_metrics_norm[col] = (state_metrics_norm[col] - state_metrics_norm[col].min()) / \
                                  (state_metrics_norm[col].max() - state_metrics_norm[col].min())
    
    heatmap_data = state_metrics_norm.set_index('State')[['Claims', 'Approval Rate', 'Avg Sum', 'Avg Age']]
    
    fig_heatmap = px.imshow(heatmap_data.T, 
                           labels=dict(x="State", y="Metric", color="Normalized Value"),
                           title='State Performance Heatmap (Normalized)',
                           color_continuous_scale='RdYlGn',
                           aspect='auto')
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)


def deep_drilldown_analysis(df, df_encoded):
    """Deep Drill-Down Analysis for Understanding Model Performance"""
    st.header("📉 Deep Drill-Down Analysis")
    
    st.markdown("""
    This section provides comprehensive drill-down capabilities to understand:
    - Why accuracy might be low
    - Why false positives/negatives are high
    - Which segments need attention
    """)
    
    # Train a reference model
    feature_cols = ['PI_AGE', 'SUM_ASSURED', 'PI_ANNUAL_INCOME', 
                   'PI_GENDER_ENCODED', 'ZONE_ENCODED', 'PAYMENT_MODE_ENCODED',
                   'EARLY_NON_ENCODED', 'MEDICAL_NONMED_ENCODED']
    available_features = [col for col in feature_cols if col in df_encoded.columns]
    
    X = df_encoded[available_features].fillna(0)
    y = df_encoded['CLAIM_STATUS']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Create analysis dataframe
    test_indices = X_test.index
    analysis_df = df.iloc[test_indices].copy()
    analysis_df['Predicted'] = y_pred
    analysis_df['Predicted_Proba'] = y_pred_proba
    analysis_df['Actual'] = y_test.values
    analysis_df['Correct'] = (analysis_df['Predicted'] == analysis_df['Actual']).astype(int)
    analysis_df['Error_Type'] = analysis_df.apply(
        lambda x: 'Correct' if x['Correct'] == 1 
        else ('False Positive' if x['Predicted'] == 1 else 'False Negative'), axis=1
    )
    
    # Tabs for different drill-downs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Segment Performance", "🔴 False Positive Deep Dive", 
        "🟡 False Negative Deep Dive", "📊 Confidence Analysis", "🔄 What-If Analysis"
    ])
    
    with tab1:
        st.subheader("Segment-wise Model Performance")
        
        segment_col = st.selectbox(
            "Select Segment Dimension",
            ['AGE_GROUP', 'ZONE', 'PI_GENDER', 'EARLY_NON', 'MEDICAL_NONMED', 'REGION', 'INCOME_GROUP']
        )
        
        # Calculate segment performance
        segment_perf = analysis_df.groupby(segment_col).agg({
            'Correct': ['sum', 'count'],
            'Predicted_Proba': 'mean'
        }).reset_index()
        segment_perf.columns = [segment_col, 'Correct Predictions', 'Total', 'Avg Confidence']
        segment_perf['Accuracy'] = segment_perf['Correct Predictions'] / segment_perf['Total'] * 100
        segment_perf['Error Rate'] = 100 - segment_perf['Accuracy']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_seg_acc = px.bar(segment_perf, x=segment_col, y='Accuracy',
                                color='Accuracy', color_continuous_scale='RdYlGn',
                                title=f'Accuracy by {segment_col}')
            fig_seg_acc.add_hline(y=segment_perf['Accuracy'].mean(), line_dash="dash",
                                 annotation_text=f"Avg: {segment_perf['Accuracy'].mean():.1f}%")
            st.plotly_chart(fig_seg_acc, use_container_width=True)
        
        with col2:
            # Error composition by segment
            error_by_seg = analysis_df[analysis_df['Correct']==0].groupby([segment_col, 'Error_Type']).size().reset_index(name='Count')
            fig_error_seg = px.bar(error_by_seg, x=segment_col, y='Count', color='Error_Type',
                                  title=f'Error Composition by {segment_col}',
                                  color_discrete_map={'False Positive': '#e74c3c', 'False Negative': '#f39c12'})
            st.plotly_chart(fig_error_seg, use_container_width=True)
        
        # Detailed segment table
        st.markdown("### Detailed Segment Analysis")
        st.dataframe(segment_perf.round(2), use_container_width=True)
        
        # Identify problematic segments
        problematic = segment_perf[segment_perf['Accuracy'] < segment_perf['Accuracy'].mean() - 5]
        if len(problematic) > 0:
            st.warning(f"⚠️ **Problematic Segments** (Accuracy below average by >5%):")
            for _, row in problematic.iterrows():
                st.write(f"- **{row[segment_col]}**: {row['Accuracy']:.1f}% accuracy ({row['Total']} samples)")
    
    with tab2:
        st.subheader("🔴 False Positive Analysis")
        st.markdown("*Cases where model predicted Approved but actual was Repudiated*")
        
        fp_df = analysis_df[analysis_df['Error_Type'] == 'False Positive']
        
        if len(fp_df) > 0:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total False Positives", len(fp_df))
                st.metric("FP Rate", f"{len(fp_df)/len(analysis_df)*100:.1f}%")
                st.metric("Avg Confidence in FP", f"{fp_df['Predicted_Proba'].mean():.2%}")
            
            with col2:
                # Characteristics of false positives
                st.markdown("**Key Characteristics:**")
                st.write(f"- Average Age: {fp_df['PI_AGE'].mean():.1f} years")
                st.write(f"- Average Sum Assured: ₹{fp_df['SUM_ASSURED'].mean():,.0f}")
                st.write(f"- Most common Zone: {fp_df['ZONE'].mode().iloc[0] if len(fp_df['ZONE'].mode()) > 0 else 'N/A'}")
                st.write(f"- Most common Reason: {fp_df['REASON_FOR_CLAIM'].mode().iloc[0] if len(fp_df['REASON_FOR_CLAIM'].mode()) > 0 else 'N/A'}")
            
            # Drill-down visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sunburst for FP drill-down
                fp_hierarchy = fp_df.groupby(['EARLY_NON', 'MEDICAL_NONMED', 'PI_GENDER']).size().reset_index(name='Count')
                fig_fp_sun = px.sunburst(fp_hierarchy, path=['EARLY_NON', 'MEDICAL_NONMED', 'PI_GENDER'],
                                        values='Count', title='FP Drill-Down: Early → Medical → Gender')
                st.plotly_chart(fig_fp_sun, use_container_width=True)
            
            with col2:
                # Confidence distribution of FPs
                fig_fp_conf = px.histogram(fp_df, x='Predicted_Proba', nbins=20,
                                          title='Confidence Distribution of False Positives',
                                          labels={'Predicted_Proba': 'Model Confidence'})
                fig_fp_conf.add_vline(x=0.5, line_dash="dash", annotation_text="Threshold")
                st.plotly_chart(fig_fp_conf, use_container_width=True)
            
            # Detailed FP samples
            st.markdown("### Sample False Positive Cases")
            display_cols = ['POLICY_NO', 'PI_AGE', 'SUM_ASSURED', 'ZONE', 'EARLY_NON', 
                          'MEDICAL_NONMED', 'REASON_FOR_CLAIM', 'Predicted_Proba']
            st.dataframe(fp_df[display_cols].head(10), use_container_width=True)
            
            # Root cause analysis
            st.markdown("### 🔍 Root Cause Analysis")
            
            # Compare FP characteristics to correctly classified
            correct_approved = analysis_df[(analysis_df['Actual']==1) & (analysis_df['Correct']==1)]
            
            comparison_metrics = pd.DataFrame({
                'Metric': ['Average Age', 'Avg Sum Assured', 'Early Claim %', 'Medical Policy %'],
                'False Positives': [
                    fp_df['PI_AGE'].mean(),
                    fp_df['SUM_ASSURED'].mean(),
                    (fp_df['EARLY_NON']=='EARLY').mean() * 100,
                    (fp_df['MEDICAL_NONMED']=='MEDICAL').mean() * 100
                ],
                'Correct Approved': [
                    correct_approved['PI_AGE'].mean(),
                    correct_approved['SUM_ASSURED'].mean(),
                    (correct_approved['EARLY_NON']=='EARLY').mean() * 100,
                    (correct_approved['MEDICAL_NONMED']=='MEDICAL').mean() * 100
                ]
            })
            comparison_metrics['Difference'] = comparison_metrics['False Positives'] - comparison_metrics['Correct Approved']
            st.dataframe(comparison_metrics.round(2), use_container_width=True)
        else:
            st.success("✅ No False Positives found! Great model performance.")
    
    with tab3:
        st.subheader("🟡 False Negative Analysis")
        st.markdown("*Cases where model predicted Repudiated but actual was Approved*")
        
        fn_df = analysis_df[analysis_df['Error_Type'] == 'False Negative']
        
        if len(fn_df) > 0:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total False Negatives", len(fn_df))
                st.metric("FN Rate", f"{len(fn_df)/len(analysis_df)*100:.1f}%")
                st.metric("Avg Confidence in FN", f"{fn_df['Predicted_Proba'].mean():.2%}")
            
            with col2:
                st.markdown("**Key Characteristics:**")
                st.write(f"- Average Age: {fn_df['PI_AGE'].mean():.1f} years")
                st.write(f"- Average Sum Assured: ₹{fn_df['SUM_ASSURED'].mean():,.0f}")
                st.write(f"- Most common Zone: {fn_df['ZONE'].mode().iloc[0] if len(fn_df['ZONE'].mode()) > 0 else 'N/A'}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fn_hierarchy = fn_df.groupby(['ZONE']).size().reset_index(name='Count').nlargest(10, 'Count')
                fig_fn_zone = px.bar(fn_hierarchy, x='ZONE', y='Count',
                                    title='False Negatives by Zone (Top 10)')
                st.plotly_chart(fig_fn_zone, use_container_width=True)
            
            with col2:
                fn_reason = fn_df['REASON_FOR_CLAIM'].value_counts().head(10)
                fig_fn_reason = px.pie(values=fn_reason.values, names=fn_reason.index,
                                      title='FN by Claim Reason (Top 10)')
                st.plotly_chart(fig_fn_reason, use_container_width=True)
            
            st.markdown("### Sample False Negative Cases")
            st.dataframe(fn_df[['POLICY_NO', 'PI_AGE', 'SUM_ASSURED', 'ZONE', 
                               'REASON_FOR_CLAIM', 'Predicted_Proba']].head(10), use_container_width=True)
        else:
            st.success("✅ No False Negatives found!")
    
    with tab4:
        st.subheader("📊 Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution by actual outcome
            fig_conf_dist = px.histogram(analysis_df, x='Predicted_Proba', color='Error_Type',
                                        nbins=30, barmode='overlay', opacity=0.7,
                                        title='Confidence Distribution by Prediction Outcome',
                                        color_discrete_map={'Correct': '#2ecc71', 
                                                          'False Positive': '#e74c3c',
                                                          'False Negative': '#f39c12'})
            st.plotly_chart(fig_conf_dist, use_container_width=True)
        
        with col2:
            # Accuracy by confidence bins
            analysis_df['Confidence_Bin'] = pd.cut(analysis_df['Predicted_Proba'], 
                                                   bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                                                   labels=['0-30%', '30-50%', '50-70%', '70-90%', '90-100%'])
            
            conf_accuracy = analysis_df.groupby('Confidence_Bin').agg({
                'Correct': 'mean',
                'Predicted_Proba': 'count'
            }).reset_index()
            conf_accuracy.columns = ['Confidence Range', 'Accuracy', 'Sample Count']
            conf_accuracy['Accuracy'] = conf_accuracy['Accuracy'] * 100
            
            fig_conf_acc = px.bar(conf_accuracy, x='Confidence Range', y='Accuracy',
                                 text='Sample Count', title='Accuracy by Confidence Level',
                                 color='Accuracy', color_continuous_scale='RdYlGn')
            fig_conf_acc.update_traces(textposition='outside')
            st.plotly_chart(fig_conf_acc, use_container_width=True)
        
        # Calibration analysis
        st.markdown("### Model Calibration")
        
        # Group by predicted probability bins and calculate actual positive rate
        analysis_df['Prob_Bin'] = pd.cut(analysis_df['Predicted_Proba'], bins=10)
        calibration = analysis_df.groupby('Prob_Bin').agg({
            'Predicted_Proba': 'mean',
            'Actual': 'mean',
            'POLICY_NO': 'count'
        }).reset_index()
        calibration.columns = ['Bin', 'Mean Predicted', 'Mean Actual', 'Count']
        
        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(x=calibration['Mean Predicted'], y=calibration['Mean Actual'],
                                       mode='markers+lines', name='Model',
                                       marker=dict(size=calibration['Count']/10)))
        fig_calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                                       line=dict(dash='dash', color='gray')))
        fig_calib.update_layout(title='Calibration Plot', xaxis_title='Mean Predicted Probability',
                               yaxis_title='Actual Positive Rate', height=400)
        st.plotly_chart(fig_calib, use_container_width=True)
    
    with tab5:
        st.subheader("🔄 What-If Analysis")
        st.markdown("Explore how changing features affects predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Adjust Features")
            age = st.slider("Age", 20, 90, 50)
            sum_assured = st.number_input("Sum Assured (₹)", 50000, 10000000, 500000, step=50000)
            annual_income = st.number_input("Annual Income (₹)", 0, 10000000, 200000, step=50000)
            gender = st.selectbox("Gender", ['M', 'F'])
            early_non = st.selectbox("Early/Non-Early", ['EARLY', 'NON EARLY'])
            medical = st.selectbox("Medical Type", ['MEDICAL', 'NON MEDICAL'])
        
        with col2:
            st.markdown("### Prediction Result")
            
            # Prepare input
            sample_input = pd.DataFrame({
                'PI_AGE': [age],
                'SUM_ASSURED': [sum_assured],
                'PI_ANNUAL_INCOME': [annual_income],
                'PI_GENDER_ENCODED': [1 if gender == 'M' else 0],
                'ZONE_ENCODED': [0],  # Default
                'PAYMENT_MODE_ENCODED': [0],  # Default
                'EARLY_NON_ENCODED': [1 if early_non == 'EARLY' else 0],
                'MEDICAL_NONMED_ENCODED': [1 if medical == 'MEDICAL' else 0]
            })
            
            sample_scaled = scaler.transform(sample_input)
            prediction = model.predict(sample_scaled)[0]
            probability = model.predict_proba(sample_scaled)[0][1]
            
            if prediction == 1:
                st.success(f"**Prediction: APPROVED**")
            else:
                st.error(f"**Prediction: REPUDIATED**")
            
            st.metric("Approval Probability", f"{probability:.1%}")
            
            # Show feature contribution
            if hasattr(model, 'feature_importances_'):
                st.markdown("### Feature Influence")
                
                importances = model.feature_importances_
                feature_names = available_features
                
                contributions = []
                for i, (name, imp) in enumerate(zip(feature_names, importances)):
                    val = sample_scaled[0][i]
                    contributions.append({
                        'Feature': name,
                        'Importance': imp,
                        'Value': val,
                        'Contribution': imp * val
                    })
                
                contrib_df = pd.DataFrame(contributions).sort_values('Contribution', ascending=True)
                
                fig_contrib = px.bar(contrib_df, x='Contribution', y='Feature', orientation='h',
                                    color='Contribution', color_continuous_scale='RdBu',
                                    title='Feature Contribution to Prediction')
                st.plotly_chart(fig_contrib, use_container_width=True)


if __name__ == "__main__":
    main()
