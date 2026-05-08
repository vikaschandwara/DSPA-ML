import streamlit as st
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import joblib
import matplotlib.pyplot as plt
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
        # Extract AN7 and take the first 1-second chunk for the MVP
        signal = data['AN7'].flatten()
        sample_rate = 40000
        chunk = signal[:sample_rate] 
        
        # # Plot the raw signal
        # st.subheader("Raw Vibration Signal (1-Second Snapshot)")
        # fig, ax = plt.subplots(figsize=(10, 3))
        # ax.plot(chunk, color='#1f77b4', alpha=0.8)
        # ax.set_ylabel("Acceleration (m/s^2)")
        # ax.set_xlabel("Samples")
        # st.pyplot(fig)

        
        # --- HIGH-PERFORMANCE INTERACTIVE PLOTTING ---
        st.subheader("Interactive Vibration Signature")
        
        # 1. Downsample the 2.4 Million points so the browser doesn't crash
        # Taking every 50th point. It preserves the exact visual shape of the wave!
        render_step = 50 
        x_vals = np.arange(len(signal))[::render_step]
        y_vals = signal[::render_step]
        
        # 2. Define what a "Fault Point" looks like visually
        # We will flag any impact spike that exceeds 3x the normal RMS energy
        rms_full = np.sqrt(np.mean(signal**2))
        impact_threshold = rms_full * 3
        
        # 3. Build the Plotly Graph
        fig = go.Figure()
        
        # Add the main blue wave (using Scattergl for high performance rendering)
        fig.add_trace(go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Vibration Wave',
            line=dict(color='#00d4ff', width=1), # A sleek, modern cyan
            opacity=0.8
        ))
        
        # Find the violent spikes and paint them red
        fault_x = [x for x, y in zip(x_vals, y_vals) if abs(y) > impact_threshold]
        fault_y = [y for y in y_vals if abs(y) > impact_threshold]
        
        if fault_x:
            fig.add_trace(go.Scattergl(
                x=fault_x,
                y=fault_y,
                mode='markers',
                name='Anomalous Impacts',
                marker=dict(color='#ff3333', size=6, symbol='x') # Bright red X's
            ))
            
        # Make the UI look incredibly premium
        fig.update_layout(
            template="plotly_dark", # Dark mode instantly makes it look pro
            xaxis_title="Time (Samples)",
            yaxis_title="Acceleration (m/s²)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)



        
        # --- FEATURE EXTRACTION ---
        # 1. Time Domain
        rms = np.sqrt(np.mean(chunk**2))
        kurt = kurtosis(chunk)
        skewness = skew(chunk)
        peak_to_peak = np.max(chunk) - np.min(chunk)
        crest_factor = np.max(np.abs(chunk)) / rms if rms > 0 else 0 
        
        # 2. Frequency Domain (FFT)
        yf = np.abs(fft(chunk))
        xf = fftfreq(sample_rate, 1 / sample_rate)
        positive_freqs = xf[1:sample_rate//2]
        positive_mags = yf[1:sample_rate//2]
        dominant_freq = positive_freqs[np.argmax(positive_mags)]
        spectral_energy = np.sum(positive_mags**2) / len(positive_mags)
        
        # Display Metrics
        st.subheader("Extracted DSP Features")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMS (Energy)", f"{rms:.4f}")
        col2.metric("Kurtosis", f"{kurt:.4f}")
        col3.metric("Dominant Freq", f"{dominant_freq:.1f} Hz")
        col4.metric("Spectral Energy", f"{spectral_energy:.4f}")
        
        # --- MACHINE LEARNING PREDICTION ---
        # Create a DataFrame with the exact same column names used during training
        features_df = pd.DataFrame([{
            'RMS': rms, 
            'Kurtosis': kurt, 
            'Skewness': skewness,
            'Peak2Peak': peak_to_peak,
            'CrestFactor': crest_factor,
            'Dominant_Freq': dominant_freq,
            'Spectral_Energy': spectral_energy
        }])
        
        # Get prediction and confidence probability
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = np.max(probabilities) * 100
        
        st.subheader("AI Diagnostic Engine")
        
        if prediction == 1:
            st.error(f"🚨 FAULT DETECTED (Confidence: {confidence:.1f}%)")
            st.write("**Analysis:** The machine learning model detected anomalous vibration signatures indicative of gearbox damage.")
        else:
            st.success(f"✅ SYSTEM HEALTHY (Confidence: {confidence:.1f}%)")
            st.write("**Analysis:** Vibration levels are consistent with normal operating parameters.")
            
    except KeyError:
        st.error("Error: Could not find sensor 'AN7' in the uploaded file. Please verify the dataset.")
