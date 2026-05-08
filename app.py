import streamlit as st
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import joblib
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Wind Turbine ML Diagnostics", layout="wide")

# Load the ML Model
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

model = load_model()

st.title("Wind Turbine Predictive Maintenance Dashboard")
st.write("Upload a .mat vibration file to analyze the High-Speed Shaft health.")

uploaded_file = st.file_uploader("Upload .mat file", type=['mat'])

if uploaded_file is not None:
    st.info("File uploaded successfully. Analyzing signal...")
    
    data = sio.loadmat(uploaded_file)
    
    try:
        # Extract AN7 signal
        signal = data['AN7'].flatten()
        sample_rate = 40000
        chunk = signal[:sample_rate] # First 1-second chunk for MVP analysis
        
        # ==========================================
        # 1. EXTRACT FEATURES & PREDICT FIRST
        # ==========================================
        # Time Domain
        rms = np.sqrt(np.mean(chunk**2))
        kurt = kurtosis(chunk)
        skewness = skew(chunk)
        peak_to_peak = np.max(chunk) - np.min(chunk)
        crest_factor = np.max(np.abs(chunk)) / rms if rms > 0 else 0 
        
        # Frequency Domain (FFT)
        yf = np.abs(fft(chunk))
        xf = fftfreq(sample_rate, 1 / sample_rate)
        positive_freqs = xf[1:sample_rate//2]
        positive_mags = yf[1:sample_rate//2]
        dominant_freq = positive_freqs[np.argmax(positive_mags)]
        spectral_energy = np.sum(positive_mags**2) / len(positive_mags)
        
        # ML Prediction
        features_df = pd.DataFrame([{
            'RMS': rms, 
            'Kurtosis': kurt, 
            'Skewness': skewness,
            'Peak2Peak': peak_to_peak,
            'CrestFactor': crest_factor,
            'Dominant_Freq': dominant_freq,
            'Spectral_Energy': spectral_energy
        }])
        
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = np.max(probabilities) * 100
        
        # ==========================================
        # 2. DISPLAY AI DIAGNOSTIC & METRICS
        # ==========================================
        st.subheader("AI Diagnostic Engine")
        
        if prediction == 1:
            st.error(f"🚨 FAULT DETECTED (Confidence: {confidence:.1f}%)")
            st.write("**Analysis:** The machine learning model detected anomalous vibration signatures indicative of gearbox damage.")
        else:
            st.success(f"✅ SYSTEM HEALTHY (Confidence: {confidence:.1f}%)")
            st.write("**Analysis:** Vibration levels are consistent with normal operating parameters.")

        st.subheader("Extracted DSP Features (1-Second Snapshot)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMS (Energy)", f"{rms:.4f}")
        col2.metric("Kurtosis", f"{kurt:.4f}")
        col3.metric("Dominant Freq", f"{dominant_freq:.1f} Hz")
        col4.metric("Spectral Energy", f"{spectral_energy:.4f}")

        # ==========================================
        # 3. INTERACTIVE PLOT (STATIC SNAPSHOT)
        # ==========================================
        st.subheader("Initial Vibration Signature")
        
        render_step = 50 
        x_vals = np.arange(len(chunk))[::render_step]
        y_vals = chunk[::render_step]
        
        # Baseline physics threshold
        absolute_threshold = 4.0 
        
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=x_vals, y=y_vals, mode='lines', name='Vibration Wave',
            line=dict(color='#00d4ff', width=1), opacity=0.8
        ))
        
        # THE FIX: Only draw red spikes if the ML model predicted a fault!
        if prediction == 1:
            fault_x = [x for x, y in zip(x_vals, y_vals) if abs(y) > absolute_threshold]
            fault_y = [y for y in y_vals if abs(y) > absolute_threshold]
            if fault_x:
                fig.add_trace(go.Scattergl(
                    x=fault_x, y=fault_y, mode='markers', name='Anomalous Impacts',
                    marker=dict(color='#ff3333', size=6, symbol='x')
                ))
            
        fig.update_layout(
            template="plotly_dark", xaxis_title="Time (Samples)", yaxis_title="Acceleration (m/s²)",
            margin=dict(l=0, r=0, t=30, b=0), height=400, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 4. LIVE TELEMETRY SIMULATOR
        # ==========================================
        st.subheader("Live Telemetry Feed (60 Seconds)")
        st.write("Simulate a live data stream of the entire 2.4 million point dataset.")
        
        if st.button("▶️ Start Live Feed"):
            plot_placeholder = st.empty()
            
            fps = 10  
            window_seconds = 2  
            
            for current_sec in range(0, 60):
                start_idx = max(0, (current_sec - window_seconds) * sample_rate)
                end_idx = current_sec * sample_rate
                
                if end_idx == 0:
                    continue
                
                window_signal = signal[start_idx:end_idx]
                x_vals_live = np.arange(start_idx, end_idx)[::render_step]
                y_vals_live = window_signal[::render_step]
                
                fig_live = go.Figure()
                fig_live.add_trace(go.Scattergl(
                    x=x_vals_live, y=y_vals_live, mode='lines', 
                    line=dict(color='#00d4ff', width=1.5), opacity=0.8
                ))
                
                # Apply the same logic fix to the live feed
                if prediction == 1:
                    fault_x_live = [x for x, y in zip(x_vals_live, y_vals_live) if abs(y) > absolute_threshold]
                    fault_y_live = [y for y in y_vals_live if abs(y) > absolute_threshold]
                    if fault_x_live:
                        fig_live.add_trace(go.Scattergl(
                            x=fault_x_live, y=fault_y_live, mode='markers', 
                            marker=dict(color='#ff3333', size=8, symbol='x')
                        ))
                
                fig_live.update_layout(
                    template="plotly_dark", xaxis_title="Total Samples Processed", yaxis_title="Acceleration (m/s²)", 
                    yaxis=dict(range=[-15, 15]), # Locked Y-axis for stable animation
                    margin=dict(l=0, r=0, t=30, b=0), height=400, showlegend=False
                )
                
                plot_placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(1 / fps)
            
            st.success("Telemetry feed complete.")

    except KeyError:
        st.error("Error: Could not find sensor 'AN7' in the uploaded file. Please verify the dataset.")