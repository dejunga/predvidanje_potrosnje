# Enhanced Streamlit App for Consumption Forecasting
import sys, os
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.model import train_and_forecast_arima

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="📈 Consumption Forecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #e2f3ff;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .info-box h4 {
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    .info-box ul {
        margin: 0.5rem 0;
        padding-left: 1.2rem;
    }
    .info-box li {
        color: #34495e;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_data(df):
    """Comprehensive data validation"""
    issues = []
    
    # Check for required columns (after processing, Datum becomes index)
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("❌ Missing or invalid date index")
    
    if 'Potrošnja' not in df.columns:
        issues.append("❌ Missing required column: 'Potrošnja'")
    
    # Check for missing values
    if df.isnull().any().any():
        issues.append(f"⚠️ Found {df.isnull().sum().sum()} missing values")
    
    # Check for negative values (only if Potrošnja column exists)
    if 'Potrošnja' in df.columns and (df['Potrošnja'] < 0).any():
        issues.append("⚠️ Found negative consumption values")
    
    # Check for duplicated dates
    if df.index.duplicated().any():
        issues.append("⚠️ Found duplicate dates")
    
    # Check data frequency
    if len(df) > 1:
        date_diff = df.index.to_series().diff().dropna()
        if len(date_diff) > 0 and not (date_diff == pd.Timedelta(days=1)).all():
            issues.append("⚠️ Data is not daily frequency or has gaps")
    
    return issues

def calculate_metrics(actual, predicted):
    """Calculate comprehensive performance metrics"""
    # Remove NaN values
    mask = ~(pd.isna(actual) | pd.isna(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {}
    
    metrics = {}
    metrics['RMSE'] = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    metrics['MAE'] = mean_absolute_error(actual_clean, predicted_clean)
    
    # Safe MAPE calculation
    nonzero_mask = actual_clean != 0
    if nonzero_mask.sum() > 0:
        metrics['MAPE'] = np.mean(np.abs((actual_clean[nonzero_mask] - predicted_clean[nonzero_mask]) / actual_clean[nonzero_mask])) * 100
    else:
        metrics['MAPE'] = np.nan
    
    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    metrics['R2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Direction accuracy
    if len(actual_clean) > 1:
        actual_direction = np.sign(np.diff(actual_clean))
        predicted_direction = np.sign(np.diff(predicted_clean))
        metrics['Direction_Accuracy'] = np.mean(actual_direction == predicted_direction) * 100
    
    return metrics

def create_comprehensive_plot(train, valid, forecast, forecast_ci=None):
    """Create an enhanced interactive plot"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Consumption Forecast', 'Forecast Detail (Zoomed)'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main plot
    fig.add_trace(
        go.Scatter(x=train.index, y=train['Potrošnja'], 
                  name='Training Data', line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    
    if valid is not None and len(valid) > 0:
        fig.add_trace(
            go.Scatter(x=valid.index, y=valid['Potrošnja'], 
                      name='Validation Data', line=dict(color='#F18F01', width=2)),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values, 
                  name='Forecast', line=dict(color='#C73E1D', width=3)),
        row=1, col=1
    )
    
    # Add confidence intervals if available
    if forecast_ci is not None:
        fig.add_trace(
            go.Scatter(x=forecast.index, y=forecast_ci.iloc[:, 1],
                      fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=forecast.index, y=forecast_ci.iloc[:, 0],
                      fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                      name='95% Confidence', fillcolor='rgba(199, 62, 29, 0.2)'),
            row=1, col=1
        )
    
    # Zoomed forecast view
    if valid is not None and len(valid) > 0:
        # Show last part of validation + forecast
        zoom_start = valid.index[-10] if len(valid) >= 10 else valid.index[0]
        
        fig.add_trace(
            go.Scatter(x=valid.index, y=valid['Potrošnja'], 
                      name='Actual', line=dict(color='#F18F01', width=2)),
            row=2, col=1
        )
        
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values, 
                  name='Forecast Detail', line=dict(color='#C73E1D', width=3)),
        row=2, col=1
    )
    
    fig.update_layout(
        title="📈 Consumption Forecasting Results",
        showlegend=True,
        height=800,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Consumption")
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Consumption Forecast Pro</h1>
    <p>Professional ARIMA-based consumption forecasting with advanced analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("🔧 Configuration")

# Sample data options
st.sidebar.subheader("📊 Sample Data")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🎲 Random Sample", help="Load a random sample dataset"):
        st.session_state.use_sample = True
        st.session_state.random_sample = True

with col2:
    if st.button("📋 Choose Sample", help="Select a specific sample dataset"):
        st.session_state.use_sample = True
        st.session_state.choose_sample = True



# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Data Upload", 
    "🎯 Model Configuration", 
    "📊 Results & Analytics", 
    "📋 Model Diagnostics"
])

# ============================================================================
# TAB 1: DATA UPLOAD
# ============================================================================
with tab1:
    st.header("📁 Data Upload & Validation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload or sample data
        if st.session_state.get('use_sample', False):
            # Load sample descriptions
            try:
                import json
                import random
                import os
                
                with open('data/sample_descriptions.json', 'r', encoding='utf-8') as f:
                    sample_descriptions = json.load(f)
                
                # Get available sample files
                sample_files = [f for f in sample_descriptions.keys() if os.path.exists(f'data/{f}')]
                
                if not sample_files:
                    st.error("❌ No sample files found")
                    st.session_state.use_sample = False
                    df = None
                else:
                    selected_file = None
                    
                    # Handle random sample selection
                    if st.session_state.get('random_sample', False):
                        selected_file = random.choice(sample_files)
                        st.session_state.selected_sample = selected_file
                        st.session_state.random_sample = False  # Reset flag
                    
                    # Handle manual sample selection
                    elif st.session_state.get('choose_sample', False):
                        # Don't reset the flag immediately, keep showing selection until something is chosen
                        
                        # Show sample selection interface
                        st.markdown("**🎯 Choose Sample Dataset:**")
                        
                        for i, file in enumerate(sample_files):
                            sample_info = sample_descriptions[file]
                            
                            # Create expandable card for each sample
                            with st.expander(f"{sample_info['name']} - {sample_info['difficulty']}"):
                                st.markdown(f"**Description:** {sample_info['description']}")
                                st.markdown("**Characteristics:**")
                                for char in sample_info['characteristics']:
                                    st.markdown(f"• {char}")
                                
                                if st.button(f"📊 Load {sample_info['name']}", key=f"load_{i}"):
                                    st.session_state.selected_sample = file
                                    st.session_state.choose_sample = False  # Reset flag after selection
                                    st.rerun()
                        
                        # Don't load data while choosing
                        selected_file = None
                    
                    # Use existing selection if available
                    elif 'selected_sample' in st.session_state:
                        selected_file = st.session_state.selected_sample
                    
                    # Load the selected file if we have one
                    if selected_file:
                        sample_info = sample_descriptions[selected_file]
                        st.info(f"🔄 Loading: {sample_info['name']}")
                        
                        df = pd.read_csv(f'data/{selected_file}', skipinitialspace=True)
                        df.columns = df.columns.str.strip().str.capitalize().str.replace("Potrosnja", "Potrošnja")
                        df["Datum"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d", exact=True)
                        df.set_index("Datum", inplace=True)
                        
                        # Show sample info
                        st.success(f"✅ {sample_info['name']} loaded successfully!")
                        st.markdown(f"**📋 Dataset Info:** {sample_info['description']}")
                        
                        # Show characteristics as badges
                        chars_text = " • ".join([f"🏷️ {char}" for char in sample_info['characteristics']])
                        st.markdown(f"**🎯 Characteristics:** {chars_text}")
                        
                        # Difficulty indicator
                        difficulty_colors = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
                        st.markdown(f"**🎚️ Forecasting Difficulty:** {difficulty_colors.get(sample_info['difficulty'], '⚪')} {sample_info['difficulty']}")
                    else:
                        df = None
                        
            except Exception as e:
                st.error(f"❌ Error loading sample data: {e}")
                st.session_state.use_sample = False
                df = None
        else:
            uploaded_file = st.file_uploader(
                "📤 Upload CSV file with columns 'Datum' and 'Potrošnja'", 
                type=["csv"],
                help="File should contain daily consumption data with dates in YYYY-MM-DD format"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, skipinitialspace=True)
                    df.columns = df.columns.str.strip().str.capitalize().str.replace("Potrosnja", "Potrošnja")
                    df["Datum"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d", exact=True)
                    df.set_index("Datum", inplace=True)
                    st.success("✅ File uploaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")
                    df = None
            else:
                df = None
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Data Requirements</h4>
            <ul>
                <li>CSV format</li>
                <li>Columns: 'Datum', 'Potrošnja'</li>
                <li>Date format: YYYY-MM-DD</li>
                <li>Daily frequency</li>
                <li>Minimum 30 days</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if df is not None:
        # Data validation
        st.subheader("🔍 Data Validation")
        issues = validate_data(df)
        
        if issues:
            for issue in issues:
                if "❌" in issue:
                    st.error(issue)
                else:
                    st.warning(issue)
        else:
            st.success("✅ Data validation passed!")
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Days", len(df))
        with col2:
            st.metric("📅 Date Range", f"{(df.index.max() - df.index.min()).days} days")
        with col3:
            st.metric("📈 Avg Consumption", f"{df['Potrošnja'].mean():.1f}")
        with col4:
            st.metric("📋 Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.subheader("📋 Data Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**First & Last 5 rows:**")
            preview_df = pd.concat([df.head(), df.tail()])
            st.dataframe(preview_df, use_container_width=True)
        
        with col2:
            st.markdown("**Statistical Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Basic visualization
        st.subheader("📈 Data Visualization")
        fig = px.line(df.reset_index(), x='Datum', y='Potrošnja', 
                     title="Historical Consumption Data")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: MODEL CONFIGURATION
# ============================================================================
with tab2:
    st.header("🎯 Model Configuration")
    
    if 'df' not in locals() or df is None:
        st.warning("⚠️ Please upload data in the Data Upload tab first.")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Train/Validation Split")
        split_ratio = st.slider("Training data percentage", 60, 95, 80) / 100
        
        split_index = int(len(df) * split_ratio)
        train = df.iloc[:split_index].copy()
        valid = df.iloc[split_index:].copy() if split_index < len(df) else pd.DataFrame()
        
        st.info(f"📊 Split: {len(train)} training days, {len(valid)} validation days")
        
        st.subheader("🔮 Forecast Parameters")
        periods = st.number_input(
            "Days to forecast", 
            min_value=1, max_value=60, value=7,
            help="Number of future days to predict"
        )
        
    with col2:
        st.subheader("🔧 ARIMA Parameters")
        
        # Auto ARIMA option
        use_auto_arima = st.checkbox("🤖 Auto ARIMA (recommended)", value=True)
        
        if not use_auto_arima:
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                p = st.number_input("p (AR)", 0, 5, 1, help="Autoregressive order")
            with col2b:
                d = st.number_input("d (I)", 0, 2, 1, help="Differencing order")
            with col2c:
                q = st.number_input("q (MA)", 0, 5, 1, help="Moving average order")
            order = (p, d, q)
        else:
            st.info("🤖 Auto ARIMA will find optimal parameters automatically")
            order = None
    
    # Model fitting
    fit_model = st.button("🚀 Train Model", type="primary")
    
    if fit_model:
        if len(train) < 10:
            st.error("❌ Insufficient training data (minimum 10 days required)")
            st.stop()
        
        with st.spinner("🔄 Training ARIMA model..."):
            try:
                if use_auto_arima:
                    # Simple grid search for best parameters
                    best_aic = float('inf')
                    best_order = None
                    best_result = None
                    
                    progress_bar = st.progress(0)
                    total_combinations = 3 * 2 * 3  # p: 0-2, d: 0-1, q: 0-2
                    current = 0
                    
                    for p in range(3):
                        for d in range(2):
                            for q in range(3):
                                try:
                                    result = train_and_forecast_arima(
                                        train, order=(p, d, q), periods=periods,
                                        valid=valid["Potrošnja"] if len(valid) > 0 else None
                                    )
                                    if result['aic'] < best_aic:
                                        best_aic = result['aic']
                                        best_order = (p, d, q)
                                        best_result = result
                                except:
                                    pass
                                current += 1
                                progress_bar.progress(current / total_combinations)
                    
                    if best_result is None:
                        st.error("❌ Could not find suitable ARIMA parameters")
                        st.stop()
                    
                    st.success(f"✅ Best model found: ARIMA{best_order} (AIC: {best_aic:.2f})")
                    order = best_order
                    model_result = best_result
                else:
                    model_result = train_and_forecast_arima(
                        train, order=order, periods=periods,
                        valid=valid["Potrošnja"] if len(valid) > 0 else None
                    )
                
                # Store results in session state
                st.session_state.model_result = model_result
                st.session_state.model_order = order
                st.session_state.train_data = train
                st.session_state.valid_data = valid
                st.session_state.periods = periods
                
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
                st.stop()

# ============================================================================
# TAB 3: RESULTS & ANALYTICS
# ============================================================================
with tab3:
    st.header("📊 Results & Analytics")
    
    if 'model_result' not in st.session_state:
        st.warning("⚠️ Please train a model first in the Model Configuration tab.")
        st.stop()
    
    result = st.session_state.model_result
    order = st.session_state.model_order
    train = st.session_state.train_data
    valid = st.session_state.valid_data
    periods = st.session_state.periods
    
    forecast = result["forecast"]
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    if len(valid) > 0:
        # Calculate metrics on validation set
        val_forecast = forecast[:len(valid)]
        metrics = calculate_metrics(valid['Potrošnja'], val_forecast)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 RMSE", f"{metrics.get('RMSE', 0):.2f}")
        with col2:
            st.metric("📊 MAE", f"{metrics.get('MAE', 0):.2f}")
        with col3:
            if not np.isnan(metrics.get('MAPE', np.nan)):
                st.metric("📋 MAPE", f"{metrics.get('MAPE', 0):.1f}%")
            else:
                st.metric("📋 MAPE", "N/A")
        with col4:
            st.metric("📈 R²", f"{metrics.get('R2', 0):.3f}")
            
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Direction Accuracy", f"{metrics.get('Direction_Accuracy', 0):.1f}%")
        with col2:
            st.metric("🔧 AIC", f"{result['aic']:.2f}")
        with col3:
            st.metric("📊 Model", f"ARIMA{order}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔧 AIC", f"{result['aic']:.2f}")
        with col2:
            st.metric("📊 Model", f"ARIMA{order}")
    
    # Comprehensive visualization
    st.subheader("📈 Forecast Visualization")
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Get confidence intervals
        model = ARIMA(train['Potrošnja'], order=order).fit()
        forecast_result = model.get_forecast(steps=periods)
        forecast_ci = forecast_result.conf_int()
        
        fig = create_comprehensive_plot(train, valid, forecast, forecast_ci)
    except:
        fig = create_comprehensive_plot(train, valid, forecast)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.subheader("📋 Detailed Forecast")
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['Date', 'Predicted_Consumption']
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(forecast_df, use_container_width=True)
    
    with col2:
        # Summary statistics
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Forecast Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("📈 Mean", f"{forecast.mean():.2f}")
        st.metric("📊 Std Dev", f"{forecast.std():.2f}")
        st.metric("📋 Min", f"{forecast.min():.2f}")
        st.metric("📈 Max", f"{forecast.max():.2f}")
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast CSV
        csv_data = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Model summary report
        if len(valid) > 0:
            report = f"""
ARIMA Forecast Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model: ARIMA{order}
AIC: {result['aic']:.2f}

Performance Metrics:
- RMSE: {metrics.get('RMSE', 0):.2f}
- MAE: {metrics.get('MAE', 0):.2f}
- MAPE: {metrics.get('MAPE', 0):.1f}%
- R²: {metrics.get('R2', 0):.3f}
- Direction Accuracy: {metrics.get('Direction_Accuracy', 0):.1f}%

Training Period: {train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')}
Validation Period: {valid.index.min().strftime('%Y-%m-%d')} to {valid.index.max().strftime('%Y-%m-%d')}
Forecast Period: {forecast.index.min().strftime('%Y-%m-%d')} to {forecast.index.max().strftime('%Y-%m-%d')}
"""
        else:
            report = f"""
ARIMA Forecast Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model: ARIMA{order}
AIC: {result['aic']:.2f}

Training Period: {train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')}
Forecast Period: {forecast.index.min().strftime('%Y-%m-%d')} to {forecast.index.max().strftime('%Y-%m-%d')}
"""
        
        st.download_button(
            label="📄 Download Report",
            data=report.encode('utf-8'),
            file_name=f"arima_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================================================
# TAB 4: MODEL DIAGNOSTICS
# ============================================================================
with tab4:
    st.header("📋 Model Diagnostics")
    
    if 'model_result' not in st.session_state:
        st.warning("⚠️ Please train a model first in the Model Configuration tab.")
        st.stop()
    
    train = st.session_state.train_data
    order = st.session_state.model_order
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.stats.stattools import jarque_bera
        from scipy import stats
        
        # Fit model for diagnostics
        model = ARIMA(train['Potrošnja'], order=order).fit()
        residuals = model.resid
        
        st.subheader("🔍 Residual Analysis")
        
        # Residual statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Mean", f"{residuals.mean():.4f}")
        with col2:
            st.metric("📈 Std Dev", f"{residuals.std():.4f}")
        with col3:
            st.metric("📋 Skewness", f"{residuals.skew():.4f}")
        with col4:
            st.metric("📊 Kurtosis", f"{residuals.kurtosis():.4f}")
        
        # Statistical tests
        st.subheader("🧪 Statistical Tests")
        
        # Ljung-Box test
        max_lags = min(10, len(residuals) // 4)
        if max_lags > 0:
            lb_test = acorr_ljungbox(residuals, lags=max_lags, return_df=True)
            lb_pvalue = lb_test.iloc[-1]['lb_pvalue']
            
            if lb_pvalue > 0.05:
                st.success(f"✅ Ljung-Box Test: No significant autocorrelation (p = {lb_pvalue:.4f})")
            else:
                st.warning(f"⚠️ Ljung-Box Test: Significant autocorrelation detected (p = {lb_pvalue:.4f})")
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
        if jb_pvalue > 0.05:
            st.success(f"✅ Jarque-Bera Test: Residuals are normally distributed (p = {jb_pvalue:.4f})")
        else:
            st.warning(f"⚠️ Jarque-Bera Test: Residuals are not normally distributed (p = {jb_pvalue:.4f})")
        
        # Diagnostic plots
        st.subheader("📊 Diagnostic Plots")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Time', 'Residuals Distribution', 
                          'Q-Q Plot', 'ACF of Residuals'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs time
        fig.add_trace(
            go.Scatter(x=train.index, y=residuals, mode='lines', name='Residuals'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=20, name='Distribution'),
            row=1, col=2
        )
        
        # Q-Q plot
        qq_data = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0], 
                      mode='lines', name='Normal Line'),
            row=2, col=1
        )
        
        # ACF plot
        from statsmodels.tsa.stattools import acf
        lags = min(20, len(residuals)//2)
        acf_values = acf(residuals, nlags=lags)
        
        fig.add_trace(
            go.Bar(x=list(range(lags+1)), y=acf_values, name='ACF'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model summary
        st.subheader("📊 Model Summary")
        st.text(str(model.summary()))
        
    except Exception as e:
        st.error(f"❌ Error generating diagnostics: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    📈 Consumption Forecast Pro | Built with Streamlit & ARIMA | 
    <a href='#' style='color: #667eea;'>Documentation</a> | 
    <a href='#' style='color: #667eea;'>Support</a>
</div>
""", unsafe_allow_html=True)