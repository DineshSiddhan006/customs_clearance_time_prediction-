import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from catboost import CatBoostRegressor  
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import os
import sys

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
# STEP 2: PIPELINE ASSET RECOVERY ENGINE
# ==============================================================================
@st.cache_resource
def load_production_pipeline():
    assets_dir = "deployed_assets"
    preprocessor_path = os.path.join(assets_dir, "custom_preprocessor_pipeline.pkl")
    model_path = os.path.join(assets_dir, "winning_clearance_model.pkl")
    
    with open(preprocessor_path, "rb") as f:
        loaded_preprocessor = pickle.load(f)
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
        
    return loaded_preprocessor, loaded_model

@st.cache_data
def load_eda_source_data():
    return pd.read_csv("customs_clearance_train.csv")

preprocessor, model = load_production_pipeline()
raw_train_df = load_eda_source_data()

# ==============================================================================
# STEP 3: APPLICATION UI AND CSS THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Customs Clearance Time Prediction", layout="wide")
st.title("Customs Clearance Time Prediction")
st.markdown("---")

tab1, tab2 = st.tabs([
    "Real-Time Manifest Prediction",
    "Exploratory Data Analysis (EDA) Report"
])

# ==============================================================================
# TAB 1: USER FORM ENTRY INTERFACE WITH COMPREHENSIVE TOOLTIPS
# ==============================================================================
with tab1:
    st.markdown("### Enter Live Flight Manifest Parameters")
    
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        submission_date = st.date_input(
            "Manifest Submission Date",
            help="The scheduled calendar date the flight landing manifest details are logged into the terminal gateway."
        )
    with r1_c2:
        submission_time = st.time_input(
            "Manifest Submission Time",
            help="The exact operational timestamp when the arrival document entry queues open for processing."
        )
        
    r2_c1, r2_c2, r2_c3 = st.columns([1, 1, 2])
    with r2_c1:
        shc_code = st.selectbox(
            "Special Handling Code (SHC)", 
            options=["COL", "VAL", "GEN", "CRT", "PER"],
            help="The special cargo class code indicating protective storage needs (e.g., Cold Chain, High-Value Tech)."
        )
        
        shc_to_heading_mapping = {
            "PER": ["0602", "0402"],  
            "COL": ["3004", "0406"],  
            "CRT": ["3004"],         
            "VAL": ["8542"],         
            "GEN": ["8471", "9403"]   
        }
        available_headings = shc_to_heading_mapping[shc_code]
        
    with r2_c2:
        hs_heading = st.selectbox(
            "HS Heading (First 4 Digits)", 
            options=available_headings,
            help="The standardized international customs tariff heading category locked automatically based on your chosen SHC."
        )
    with r2_c3:
        hs_extension = st.number_input(
            "HS Code Extension (Remaining 8 Digits)", 
            min_value=0, 
            max_value=99999999, 
            value=67540610, 
            step=1,
            format="%d",
            help="The granular variable sub-heading item markers making up the rest of your full 12-digit commodity identification code."
        )
        full_hs_string = f"{hs_heading}{str(hs_extension).zfill(8)}"
        st.caption(f"**Combined Full HS Code Passed to Model Matrix:** `{full_hs_string}`")

    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        port_loading = st.selectbox(
            "Port of Loading Hub (IATA Airport)", 
            options=["AMS", "BOM", "PVG", "TPE", "CDG", "FRA"],
            help="The origin international airport location where the cargo containers were inspected and stowed on board."
        )
        port_to_country = {"AMS": "NLD", "BOM": "IND", "PVG": "CHN", "TPE": "TWN", "CDG": "FRA", "FRA": "DEU"}
        origin_country = port_to_country[port_loading]
    with r3_c2:
        hist_avg_hours = st.number_input(
            "Historical 90-Day Clearance Avg (Hours)", 
            value=60.27, 
            step=0.1, 
            help="The long-term baseline moving average clearance delay recorded for this specific importing enterprise profile."
        )

    r4_c1, r4_c2 = st.columns(2)
    with r4_c1:
        fatoorah_passed = st.selectbox(
            "ZATCA Fatoorah XML Validation Passed?", 
            options=[1, 0], 
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Confirms if the cryptographic structure of the digital invoice successfully cleared automated pre-arrival validation."
        )
    with r4_c2:
        weight_discrepancy = st.slider(
            "Physical vs Declared Weight Discrepancy Ratio", 
            min_value=0.0, 
            max_value=0.10, 
            value=0.012, 
            step=0.001, 
            help="The calculated weight variance between paper declarations and actual scales. Values over 0.05 trigger strict secondary inspection flags."
        )

    r5_c1, r5_c2 = st.columns(2)
    with r5_c1:
        pre_filing_hours = st.number_input(
            "Pre-Arrival Filing Window (Hours before Landing)", 
            value=12.57, 
            step=0.1,
            help="The amount of hours before touchdown that the documentation was complete. Negative numbers mean late filings made post-landing."
        )
    with r5_c2:
        ambient_temp = st.number_input(
            "Terminal Ambient Temperature (°C)", 
            value=36.6, 
            step=0.1,
            help="The real-time ambient thermal reading outside the air terminal hangar corridors."
        )

    r6_c1 = st.columns(1)[0]
    with r6_c1:
        visibility_meters = st.number_input(
            "Meteorological Runway Visibility (Meters)", 
            value=1368.0, 
            step=10.0, 
            help="Runway sight range index. High parameters indicate clear skyways, while drop boundaries signal low visibility tarmac restrictions."
        )

    st.markdown("---")
    
    if st.button("Compute Target Customs Dwell Time", use_container_width=True):
        time_string = f"{submission_date.strftime('%Y-%m-%d')}T{submission_time.strftime('%H:%M:%S')}+03:00"
        
        input_payload = pd.DataFrame({
            'submission_timestamp': [time_string],
            'hs_code': [int(full_hs_string)],  
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
                delta="High Precision Optimization Track Enabled"
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
# STEP 4: EXPLORATORY DATA ANALYSIS (HIGH-CONTRAST STATIC GRIDSPEC)
# ==============================================================================
with tab2:
    eda_df = raw_train_df.copy()
    
    # FIXED HIGH CONTRAST TYPOGRAPHY CONFIGURATION Matrix
    plt.style.use('default')
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'text.color': '#000000',          # Absolute dark text color
        'axes.labelcolor': '#000000',     # Axis labels dark black
        'xtick.color': '#000000',         # X ticks bold dark black
        'ytick.color': '#000000',         # Y ticks bold dark black
        'axes.titlecolor': '#000000',     # Subplot headers crisp black
        'figure.facecolor': '#ffffff',    # Stark white frame context
        'axes.facecolor': '#ffffff',      # Stark white background axes
        'grid.color': '#E5E7EB',          # Light clean background mesh grid lines
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 15
    })
    
    # ------------------------------------------------------------------
    # CHART 1 HIERARCHY: HS CODES VS CLEARANCE DURATION
    # ------------------------------------------------------------------
    st.subheader("Chart 1")
    fig_row1 = plt.figure(figsize=(24, 8))
    gs1 = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])
    
    # Graph A
    ax1 = fig_row1.add_subplot(gs1[0])
    eda_df['hs_chapter'] = eda_df['hs_code'].astype(str).str.zfill(12).str[:2]
    chapter_labels = {
        '04': 'Ch.04: Dairy & Agro', '06': 'Ch.06: Live Plants',
        '30': 'Ch.30: Pharma', '84': 'Ch.84: Industrial Machinery',
        '85': 'Ch.85: Tech & Electronics', '94': 'Ch.94: General Commodities'
    }
    eda_df['hs_chapter_named'] = eda_df['hs_chapter'].map(chapter_labels).fillna(eda_df['hs_chapter'].apply(lambda x: f"Ch.{x}: Other Cargo"))
    chapter_medians = eda_df.groupby('hs_chapter_named')['clearance_duration_hours'].median().sort_values(ascending=False).reset_index()
    sns.barplot(ax=ax1, data=chapter_medians, y='hs_chapter_named', x='clearance_duration_hours', color='#438a5e')
    ax1.set_title("HS Code Chapter vs Clearance Duration", weight='bold', pad=15)
    ax1.set_xlabel("Clearance Duration (Hours)", weight='bold')
    ax1.set_ylabel("HS Chapter Classification", weight='bold')
    ax1.set_xlim(0, 70)
    for i, v in enumerate(chapter_medians['clearance_duration_hours']):
        ax1.text(v + 1, i, f"{v:.1f}h", va='center', ha='left', fontsize=12, weight='bold', color='#000000')
        
    # Graph B
    ax2 = fig_row1.add_subplot(gs1[1])
    eda_df['hs_heading'] = eda_df['hs_code'].astype(str).str.zfill(12).str[:4]
    heading_medians = eda_df.groupby('hs_heading')['clearance_duration_hours'].median().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(ax=ax2, data=heading_medians, y='hs_heading', x='clearance_duration_hours', hue='hs_heading', palette="YlOrRd_r", legend=False)
    ax2.set_title("Top 10 High-Delay HS Headings vs Clearance Duration", weight='bold', pad=15)
    ax2.set_xlabel("Clearance Duration (Hours)", weight='bold')
    ax2.set_ylabel("HS Heading Category Code", weight='bold')
    ax2.set_xlim(0, 70)
    for i, v in enumerate(heading_medians['clearance_duration_hours']):
        ax2.text(v + 1, i, f"{v:.1f}h", va='center', ha='left', fontsize=12, weight='bold', color='#000000')
        
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_row1, transparent=False)
    plt.close()
    st.markdown("---")

    # ------------------------------------------------------------------
    # CHART 2 HIERARCHY: INTERNATIONAL FREIGHT LANE ANALYSIS
    # ------------------------------------------------------------------
    st.subheader("Chart 2")
    fig_row2 = plt.figure(figsize=(24, 8))
    gs2 = gridspec.GridSpec(1, 2)
    
    # Graph A
    ax3 = fig_row2.add_subplot(gs2[0])
    country_labels = {
        'IND': 'India (IND)', 'DEU': 'Germany (DEU)', 'NLD': 'Netherlands (NLD)',
        'FRA': 'France (FRA)', 'TWN': 'Taiwan (TWN)', 'CHN': 'China (CHN)'
    }
    eda_df['origin_country_named'] = eda_df['origin_country'].map(country_labels).fillna(eda_df['origin_country'])
    country_medians = eda_df.groupby('origin_country_named')['clearance_duration_hours'].median().sort_values(ascending=False).reset_index()
    sns.barplot(ax=ax3, data=country_medians, y='origin_country_named', x='clearance_duration_hours', color='#1e3d59')
    ax3.set_title("Country of Origin vs Clearance Duration", weight='bold', pad=15)
    ax3.set_xlabel("Clearance Duration (Hours)", weight='bold')
    ax3.set_ylabel("Country of Origin", weight='bold')
    ax3.set_xlim(0, 70)
    for i, v in enumerate(country_medians['clearance_duration_hours']):
        ax3.text(v + 1, i, f"{v:.1f}h", va='center', ha='left', fontsize=12, weight='bold', color='#000000')
        
    # Graph B
    ax4 = fig_row2.add_subplot(gs2[1])
    port_medians = eda_df.groupby('port_of_loading')['clearance_duration_hours'].median().sort_values(ascending=False).reset_index()
    sns.barplot(ax=ax4, data=port_medians, y='port_of_loading', x='clearance_duration_hours', palette="Blues_r", hue='port_of_loading', legend=False)
    ax4.set_title("Departure Port vs Clearance Duration", weight='bold', pad=15)
    ax4.set_xlabel("Clearance Duration (Hours)", weight='bold')
    ax4.set_ylabel("Port of Loading Airport Code", weight='bold')
    ax4.set_xlim(0, 70)
    for i, v in enumerate(port_medians['clearance_duration_hours']):
        ax4.text(v + 1, i, f"{v:.1f}h", va='center', ha='left', fontsize=12, weight='bold', color='#000000')
        
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_row2, transparent=False)
    plt.close()
    st.markdown("---")

    # ------------------------------------------------------------------
    # CHART 3 HIERARCHY: REGULATORY PROGRAM EFFECTIVENESS
    # ------------------------------------------------------------------
    st.subheader("Chart 3")
    fig_row3 = plt.figure(figsize=(24, 8))
    gs3 = gridspec.GridSpec(1, 2)
    
    # Graph A
    ax5 = fig_row3.add_subplot(gs3[0])
    aeo_map = {1: "Gold Tier AEO Certified (1)", 0: "Standard Importer (0)"}
    eda_df['aeo_status_named'] = eda_df['is_aeo_certified'].map(aeo_map)
    aeo_colors = {"Gold Tier AEO Certified (1)": "#2ec4b6", "Standard Importer (0)": "#1e3d59"}
    sns.boxplot(ax=ax5, data=eda_df, y='aeo_status_named', x='clearance_duration_hours', hue='aeo_status_named', palette=aeo_colors, width=0.4, showfliers=False, legend=False)
    ax5.set_title("Authorized Economic Operator (AEO) Status vs Clearance Duration", weight='bold', pad=15)
    ax5.set_xlabel("Clearance Duration (Hours)", weight='bold')
    ax5.set_ylabel("Importer Profile", weight='bold')
    ax5.set_xlim(0, 70)
    for tick, label in enumerate(eda_df['aeo_status_named'].unique()):
        median_val = eda_df[eda_df['aeo_status_named'] == label]['clearance_duration_hours'].median()
        ax5.text(median_val, tick - 0.28, f"Median: {median_val:.1f}h", ha='center', va='bottom', fontsize=12, weight='bold', color='#000000')
        
    # Graph B
    ax6 = fig_row3.add_subplot(gs3[1])
    fatoorah_map = {1: "ZATCA Compliant XML E-Invoice (1)", 0: "Legacy PDF/Manual Manifest (0)"}
    eda_df['fatoorah_status_named'] = eda_df['fatoorah_validation_passed'].map(fatoorah_map)
    fatoorah_colors = {"ZATCA Compliant XML E-Invoice (1)": "#2ec4b6", "Legacy PDF/Manual Manifest (0)": "#e63946"}
    sns.boxplot(ax=ax6, data=eda_df, y='fatoorah_status_named', x='clearance_duration_hours', hue='fatoorah_status_named', palette=fatoorah_colors, width=0.4, showfliers=False, legend=False)
    ax6.set_title("Digital E-Invoicing Gateway Validation vs Clearance Duration", weight='bold', pad=15)
    ax6.set_xlabel("Clearance Duration (Hours)", weight='bold')
    ax6.set_ylabel("Documentation Profile", weight='bold')
    ax6.set_xlim(0, 70)
    for tick, label in enumerate(eda_df['fatoorah_status_named'].unique()):
        median_val = eda_df[eda_df['fatoorah_status_named'] == label]['clearance_duration_hours'].median()
        ax6.text(median_val, tick - 0.28, f"Median: {median_val:.1f}h", ha='center', va='bottom', fontsize=12, weight='bold', color='#000000')
        
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_row3, transparent=False)
    plt.close()
    st.markdown("---")

    # ------------------------------------------------------------------
    # CHART 4 HIERARCHY: RISK THRESHOLDS AND PROFILE CORRELATIONS
    # ------------------------------------------------------------------
    st.subheader("Chart 4")
    fig_row4 = plt.figure(figsize=(24, 8))
    gs4 = gridspec.GridSpec(1, 2)
    
    # Graph A
    ax7 = fig_row4.add_subplot(gs4[0])
    sns.scatterplot(ax=ax7, data=eda_df, x='historical_avg_clearance_hours', y='clearance_duration_hours', alpha=0.5, color='#1e3d59', edgecolor='none')
    sns.regplot(ax=ax7, data=eda_df, x='historical_avg_clearance_hours', y='clearance_duration_hours', scatter=False, color='#e63946', line_kws={"linewidth": 3.0, "linestyle": "--"})
    ax7.set_title("Historical 90-Day Importer Average vs Current Clearance Duration", weight='bold', pad=15)
    ax7.set_xlabel("Historical 90-Day Rolling Clearance Mean (Hours)", weight='bold')
    ax7.set_ylabel("Current Clearance Duration (Hours)", weight='bold')
    ax7.set_xlim(0, 168)
    ax7.set_ylim(0, 250)
    
    # Graph B
    ax8 = fig_row4.add_subplot(gs4[1])
    sns.scatterplot(ax=ax8, data=eda_df, x='weight_value_discrepancy', y='clearance_duration_hours', alpha=0.5, color='#1e3d59', edgecolor='none')
    sns.regplot(ax=ax8, data=eda_df, x='weight_value_discrepancy', y='clearance_duration_hours', scatter=False, color='#e63946', line_kws={"linewidth": 3.0, "linestyle": "--"})
    ax8.set_title("Physical Weight Discrepancy Ratio vs Clearance Duration", weight='bold', pad=15)
    ax8.set_xlabel("Weight Discrepancy Ratio (Percentage)", weight='bold')
    ax8.set_ylabel("Clearance Duration (Hours)", weight='bold')
    ax8.set_xlim(0, 0.10)
    ax8.set_ylim(0, 250)
    ax8.axvline(x=0.05, color='#e63946', linestyle=':', alpha=0.9, linewidth=2.5)
    ax8.text(0.052, 220, "ZATCA 5% Fraud Alert Gate", color='#e63946', weight='bold', fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_row4, transparent=False)
    plt.close()
