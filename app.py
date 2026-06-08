# ==============================================================================
# PRODUCTION INDUSTRIAL STREAMLIT APPLICATION: app.py
# Use Case: Customs Clearance Time Prediction
# Reference Standard: app(2).py Integration Mapping Suite
# Execution: streamlit run app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import catboost
import plotly.express as px
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import sys

# ==============================================================================
# STEP 1: DEFINE STRUCTURAL PREPROCESSOR BLUEPRINT FOR JOBLIB DESERIALIZATION
# ==============================================================================
class DynamicLogisticsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_threshold=10, target_col='clearance_duration_hours', drop_cols=None):
        self.categorical_threshold = categorical_threshold
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols else ['shipment_id']
        self.imputation_values_ = {}
        self.low_cardinality_cols_ = []
        self.high_cardinality_cols_ = []
        self.one_hot_categories_ = {}
        self.numeric_cols_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in X_out.columns:
            if 'timestamp' in col:
                datetime_series = pd.to_datetime(X_out[col], errors='coerce')
                X_out[f'{col}_hour'] = datetime_series.dt.hour.fillna(12).astype(int)
                X_out[f'{col}_dayofweek'] = datetime_series.dt.dayofweek.fillna(0).astype(int)
                X_out[f'{col}_is_weekend'] = datetime_series.dt.dayofweek.isin([4, 5]).astype(int)
                X_out[f'{col}_is_solar_holiday'] = (
                    ((datetime_series.dt.month == 9) & (datetime_series.dt.day == 23)) | 
                    ((datetime_series.dt.month == 2) & (datetime_series.dt.day == 22))
                ).astype(int)
                X_out.drop(columns=[col], inplace=True)

        for col in self.numeric_cols_:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.imputation_values_.get(col, 0))
                
        if 'ambient_temperature_celsius' in X_out.columns and 'shc_code' in X_out.columns:
            is_sensitive_cargo = X_out['shc_code'].isin(['PER', 'COL', 'CRT'])
            is_extreme_heat = X_out['ambient_temperature_celsius'] > 43.0
            X_out['interaction_thermal_shock_risk'] = (is_sensitive_cargo & is_extreme_heat).astype(int)

        if 'pre_arrival_filing_hours' in X_out.columns:
            X_out['interaction_late_filing_flag'] = (X_out['pre_arrival_filing_hours'] < 0.0).astype(int)

        for col in self.low_cardinality_cols_:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna("UNKNOWN")
                for category in self.one_hot_categories_.get(col, []):
                    X_out[f"ohe_{col}_{category}"] = (X_out[col] == category).astype(int)
                X_out.drop(columns=[col], inplace=True)

        for col in self.high_cardinality_cols_:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna("UNKNOWN").astype(str)
                X_out[f"encoded_{col}"] = X_out[col].astype('category').cat.codes
                X_out.drop(columns=[col], inplace=True)

        cols_to_drop = [c for c in self.drop_cols if c in X_out.columns]
        X_out.drop(columns=cols_to_drop, inplace=True)
        return X_out

sys.modules['__main__'] = sys.modules[__name__]

# ==============================================================================
# STEP 2: LOAD PIPELINE ASSETS VIA JOBLIB CONTRACT
# ==============================================================================
@st.cache_resource
def load_production_assets():
    model_obj = joblib.load("winning_clearance_model.pkl")
    prep_obj = joblib.load("custom_preprocessor_pipeline.pkl")
    return model_obj, prep_obj

model, preprocessor = load_production_assets()

try:
    test_data = pd.read_csv("customs_clearance_test.csv")
except Exception:
    test_data = pd.DataFrame()

# ==============================================================================
# STEP 3: APPLICATION VISUAL STYLESHEET
# ==============================================================================
st.set_page_config(page_title="Customs Clearance Time Prediction", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1rem;}
.main {background-color:#f5f7fb;}
[data-testid="stMetric"]{
    background:white;padding:20px;border-radius:18px;
    border-top:5px solid #111827;
}
.result-box{
    background:#dbeafe;padding:25px;border-radius:10px;
    font-size:22px;font-weight:bold;color:#0b4aa2;
}
.result-box-danger{\n    background:#fee2e2;padding:25px;border-radius:10px;
    font-size:22px;font-weight:bold;color:#b91c1c;
}
</style>
""", unsafe_allow_html=True)

st.title("Customs Clearance Time Prediction")
st.markdown("---")

tab1, tab2 = st.tabs([
    "Real-Time Manifest Prediction",
    "Exploratory Data Analysis (EDA) Report"
])

# ==============================================================================
# TAB 1: USER COMPUTE ENTRY INTERFACE
# ==============================================================================
with tab1:
    st.markdown("### Enter Live Flight Manifest Parameters")
    
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        submission_date = st.date_input("Manifest Submission Date")
    with r1_c2:
        submission_time = st.time_input("Manifest Submission Time")
        
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        shc_code = st.selectbox("Special Handling Code (SHC)", options=["COL", "VAL", "GEN", "CRT", "PER"])
        
        # Adaptive dropdown lookup rules
        shc_to_hs_mapping = {
            "PER": [60267540610, 40210100000],  
            "COL": [30049000000, 40210100000],  
            "CRT": [30049000000],               
            "VAL": [85423100000],               
            "GEN": [84713000000, 94032000000]   
        }
        available_hs_options = shc_to_hs_mapping[shc_code]
        
    with r2_c2:
        hs_code = st.selectbox("Saudi Tariff HS Code", options=available_hs_options)

    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        port_loading = st.selectbox("Port of Loading Hub (IATA Airport)", options=["AMS", "BOM", "PVG", "TPE", "CDG", "FRA"])
        port_to_country = {"AMS": "NLD", "BOM": "IND", "PVG": "CHN", "TPE": "TWN", "CDG": "FRA", "FRA": "DEU"}
        origin_country = port_to_country[port_loading]
    with r3_c2:
        hist_avg_hours = st.number_input("Historical 90-Day Clearance Avg (Hours)", value=60.27, step=0.1)

    r4_c1, r4_c2 = st.columns(2)
    with r4_c1:
        fatoorah_passed = st.selectbox("ZATCA Fatoorah XML Validation Passed?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with r4_c2:
        weight_discrepancy = st.slider("Physical vs Declared Weight Discrepancy Ratio", min_value=0.0, max_value=0.10, value=0.012, step=0.001)

    r5_c1, r5_c2 = st.columns(2)
    with r5_c1:
        pre_filing_hours = st.number_input("Pre-Arrival Filing Window (Hours before Landing)", value=12.57, step=0.1)
    with r5_c2:
        ambient_temp = st.number_input("Terminal Ambient Temperature (°C)", value=36.6, step=0.1)

    r6_c1 = st.columns(1)[0]
    with r6_c1:
        visibility_meters = st.number_input("Meteorological Runway Visibility (Meters)", value=1368.0, step=10.0)

    st.markdown("---")
    
    if st.button("Compute Target Customs Dwell Time", use_container_width=True):
        time_string = f"{submission_date.strftime('%Y-%m-%d')}T{submission_time.strftime('%H:%M:%S')}+03:00"
        
        # Build training-compliant background payload
        input_payload = pd.DataFrame({
            'submission_timestamp': [time_string],
            'hs_code': [int(hs_code)],
            'shc_code': [str(shc_code)],
            'origin_country': [str(origin_country)],
            'port_of_loading': [str(port_loading)],
            'importer_cr_id': [4713095212], 
            'is_aeo_certified': [1],          
            'historical_avg_clearance_hours': [float(hist_avg_hours)],
            'fatoorah_validation_passed': [int(fatoorah_passed)],
            'weight_value_discrepancy': [float(weight_discrepancy)],
            'pre_arrival_filing_hours': [float(pre_filing_hours)],
            'ambient_temperature_celsius': [float(ambient_temp)],
            'visibility_meters': [float(visibility_meters)],
            'inspection_track': [0]            
        })
        
        try:
            transformed_features = preprocessor.transform(input_payload)
            features_clean = transformed_features.drop(columns=['clearance_duration_hours'], errors='ignore')
            
            if 'inspection_track' in features_clean.columns:
                features_clean['inspection_track'] = features_clean['inspection_track'].map({'GREEN': 0, 'YELLOW': 1, 'RED': 2}).fillna(features_clean['inspection_track'])
            
            predicted_dwell_hours = float(model.predict(features_clean)[0])
            predicted_dwell_hours_clipped = max(0.5, min(predicted_dwell_hours, 360.0))
            
            st.metric(
                label="Predicted Operational Clearance Wait Time",
                value=f"{predicted_dwell_hours_clipped:.2f} Hours",
                delta="Uncertainty Interval Margin: ±12.71 Hours"
            )
            
            if predicted_dwell_hours_clipped >= 72.0:
                st.error("HIGH DWELL TIME WARNING: Shipment requires manual verification or documentation audit paths.")
            elif predicted_dwell_hours_clipped >= 24.0:
                st.warning("MODERATE EXPEDITION LATENCY: Document queue bottlenecks present.")
            else:
                st.success("RAPID GREEN-CHANNEL ROUTING: Automated release clearance expected within optimal windows.")
                
        except Exception as error:
            st.error(f"Inference System Fault: {str(error)}")

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA HOOK LAYOUT FROM REFERENCE)
# ==============================================================================
with tab2:
    if not test_data.empty:
        
        # ------------------------------------------------------------------
        # CHART 1
        # ------------------------------------------------------------------
        st.subheader("Chart 1")
        c1, c2 = st.columns(2)
        
        with c1:
            fig1_a = px.box(
                test_data,
                x="inspection_track",
                y="clearance_duration_hours",
                title="Graph A: Inspection Track vs Clearance Time"
            )
            st.plotly_chart(fig1_a, use_container_width=True)
            
        with c2:
            country_group = test_data.groupby("origin_country")["clearance_duration_hours"].mean().reset_index()
            country_group = country_group.sort_values("clearance_duration_hours", ascending=False).head(10)
            fig1_b = px.bar(
                country_group,
                x="origin_country",
                y="clearance_duration_hours",
                title="Graph B: Origin Country Analytics"
            )
            st.plotly_chart(fig1_b, use_container_width=True)

        # ------------------------------------------------------------------
        # CHART 2
        # ------------------------------------------------------------------
        st.subheader("Chart 2")
        c3, c4, c5 = st.columns(3)
        
        with c3:
            fig2_a = px.box(
                test_data,
                x="fatoorah_validation_passed",
                y="clearance_duration_hours",
                title="Graph A: Fatoorah Impact"
            )
            st.plotly_chart(fig2_a, use_container_width=True)
            
        with c4:
            fig2_b = px.box(
                test_data,
                x="is_aeo_certified",
                y="clearance_duration_hours",
                title="Graph B: AEO Performance"
            )
            st.plotly_chart(fig2_b, use_container_width=True)
            
        with c5:
            fig2_c = px.scatter(
                test_data,
                x="historical_avg_clearance_hours",
                y="clearance_duration_hours",
                title="Graph C: Historical vs Actual"
            )
            st.plotly_chart(fig2_c, use_container_width=True)
    else:
        st.info("Exploratory data analysis plots are disabled because 'customs_clearance_test.csv' is missing.")