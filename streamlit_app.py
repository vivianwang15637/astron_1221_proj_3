"""
Streamlit Web App for Cepheid Period-Luminosity Relation Analysis

This app provides an interactive interface to explore Cepheid variable data,
fit the Period-Luminosity relation, and visualize results.

Features:
- Interactive data exploration
- Period-Luminosity relation fitting
- Visualization with residuals
- Distance estimation calculator
- Simple MCP (Model Context Protocol) integration for AI assistance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Page configuration
st.set_page_config(
    page_title="Cepheid Period-Luminosity Relation",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL FUNCTION (Same as notebook)
# ============================================================================

def period_luminosity_model(log10_period, slope, intercept):
    """
    Calculates the predicted magnitude from the Period-Luminosity relation.
    
    Parameters:
    -----------
    log10_period : float or array
        The base-10 logarithm of the period (in days)
    slope : float
        The slope of the line
    intercept : float
        The y-intercept
    
    Returns:
    --------
    magnitude : float or array
        The predicted magnitude value(s)
    """
    return slope * log10_period + intercept

# ============================================================================
# MCP INTEGRATION (Simple, Beginner-Friendly)
# ============================================================================

def simple_mcp_query(prompt, api_key=None, model="gpt-3.5-turbo"):
    """
    Simple MCP (Model Context Protocol) integration for AI assistance.
    
    This is a beginner-friendly wrapper that can connect to OpenAI-compatible APIs.
    For production use, you would use a proper MCP client library.
    
    Parameters:
    -----------
    prompt : str
        The question or prompt to send to the AI
    api_key : str, optional
        API key for the LLM service (stored in environment variable or Streamlit secrets)
    model : str
        Model name to use (default: gpt-3.5-turbo)
    
    Returns:
    --------
    str : Response from the AI, or error message
    """
    try:
        # Try to get API key from Streamlit secrets or environment
        if api_key is None:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", None)
            except:
                api_key = os.environ.get("OPENAI_API_KEY", None)
        
        if api_key is None:
            return "⚠️ API key not found. Please set OPENAI_API_KEY in environment variables or Streamlit secrets."
        
        # Simple OpenAI API call (requires openai package)
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful astronomy assistant. Provide clear, accurate explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            return "⚠️ OpenAI package not installed. Install with: pip install openai"
        except Exception as e:
            return f"⚠️ Error connecting to AI: {str(e)}"
            
    except Exception as e:
        return f"⚠️ MCP Error: {str(e)}"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">⭐ Cepheid Period-Luminosity Relation Analyzer</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>About:</strong> This app analyzes Cepheid variable stars from the OGLE survey 
    to fit the Period-Luminosity relation. Cepheids are "standard candles" used to measure 
    cosmic distances.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["📊 Data Explorer", "🔧 Fit Model", "📈 Visualizations", "🌌 Distance Calculator", "🤖 AI Assistant (MCP)"]
    )
    
    # Load data
    data_file = "ogle_lmc_cepheids.csv"
    if not os.path.exists(data_file):
        st.error(f"❌ Data file '{data_file}' not found. Please run the notebook first to download the data.")
        return
    
    try:
        df = pd.read_csv(data_file)
        df['log10_Period'] = np.log10(df['Period'])
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # ========================================================================
    # PAGE 1: DATA EXPLORER
    # ========================================================================
    if page == "📊 Data Explorer":
        st.markdown('<p class="section-header">Data Explorer</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stars", len(df))
        with col2:
            st.metric("Period Range", f"{df['Period'].min():.2f} - {df['Period'].max():.2f} days")
        with col3:
            st.metric("Magnitude Range", f"{df['I_mag'].min():.2f} - {df['I_mag'].max():.2f} mag")
        
        st.subheader("Dataset Preview")
        st.dataframe(df[['StarID', 'Period', 'I_mag', 'log10_Period']], use_container_width=True)
        
        st.subheader("Basic Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Period Statistics (days):**")
            st.write(df['Period'].describe())
        with col2:
            st.write("**Magnitude Statistics:**")
            st.write(df['I_mag'].describe())
        
        # Quick scatter plot
        st.subheader("Quick Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['log10_Period'], df['I_mag'], alpha=0.6, s=50)
        ax.set_xlabel('log₁₀(Period [days])', fontsize=12)
        ax.set_ylabel('I-band Magnitude', fontsize=12)
        ax.set_title('Cepheid Period-Luminosity Data', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # ========================================================================
    # PAGE 2: FIT MODEL
    # ========================================================================
    elif page == "🔧 Fit Model":
        st.markdown('<p class="section-header">Fit Period-Luminosity Relation</p>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        The Period-Luminosity relation is: **magnitude = slope × log₁₀(period) + intercept**
        
        Click the button below to fit the model to the data.
        """)
        
        if st.button("🚀 Run Fitting", type="primary"):
            with st.spinner("Fitting model..."):
                # Prepare data
                x_data = df['log10_Period'].values
                y_data = df['I_mag'].values
                y_errors = np.full_like(y_data, 0.1)  # Assumed errors
                
                # Initial guess
                initial_slope = (y_data.max() - y_data.min()) / (x_data.max() - x_data.min())
                initial_slope = -abs(initial_slope)
                initial_intercept = y_data.mean()
                initial_guess = [initial_slope, initial_intercept]
                
                # Fit
                try:
                    popt, pcov = curve_fit(
                        period_luminosity_model,
                        x_data,
                        y_data,
                        p0=initial_guess,
                        sigma=y_errors,
                        absolute_sigma=True
                    )
                    
                    fitted_slope = popt[0]
                    fitted_intercept = popt[1]
                    param_errors = np.sqrt(np.diag(pcov))
                    slope_error = param_errors[0]
                    intercept_error = param_errors[1]
                    
                    # Store in session state
                    st.session_state['fitted_slope'] = fitted_slope
                    st.session_state['fitted_intercept'] = fitted_intercept
                    st.session_state['slope_error'] = slope_error
                    st.session_state['intercept_error'] = intercept_error
                    st.session_state['x_data'] = x_data
                    st.session_state['y_data'] = y_data
                    st.session_state['y_errors'] = y_errors
                    
                    # Display results
                    st.success("✅ Fitting completed successfully!")
                    st.subheader("Fitted Parameters")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Slope (a)", f"{fitted_slope:.3f} ± {slope_error:.3f}")
                    with col2:
                        st.metric("Intercept (b)", f"{fitted_intercept:.3f} ± {intercept_error:.3f}")
                    
                    st.write(f"**Fitted Equation:** I_mag = ({fitted_slope:.3f} ± {slope_error:.3f}) × log₁₀(P) + ({fitted_intercept:.3f} ± {intercept_error:.3f})")
                    
                    # Calculate statistics
                    y_model = period_luminosity_model(x_data, fitted_slope, fitted_intercept)
                    residuals = y_data - y_model
                    chi2 = np.sum((residuals / y_errors) ** 2)
                    dof = len(y_data) - 2
                    reduced_chi2 = chi2 / dof
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    st.subheader("Fit Quality Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Reduced χ²", f"{reduced_chi2:.3f}")
                    with col2:
                        st.metric("R²", f"{r_squared:.4f}")
                    with col3:
                        st.metric("RMS Residual", f"{np.sqrt(np.mean(residuals**2)):.4f} mag")
                    
                except Exception as e:
                    st.error(f"❌ Fitting failed: {str(e)}")
        
        else:
            st.info("👆 Click the button above to fit the Period-Luminosity relation.")
    
    # ========================================================================
    # PAGE 3: VISUALIZATIONS
    # ========================================================================
    elif page == "📈 Visualizations":
        st.markdown('<p class="section-header">Visualizations</p>', unsafe_allow_html=True)
        
        if 'fitted_slope' not in st.session_state:
            st.warning("⚠️ Please fit the model first in the 'Fit Model' section.")
        else:
            x_data = st.session_state['x_data']
            y_data = st.session_state['y_data']
            y_errors = st.session_state['y_errors']
            fitted_slope = st.session_state['fitted_slope']
            fitted_intercept = st.session_state['fitted_intercept']
            slope_error = st.session_state['slope_error']
            intercept_error = st.session_state['intercept_error']
            
            # Calculate model and residuals
            y_model = period_luminosity_model(x_data, fitted_slope, fitted_intercept)
            residuals = y_data - y_model
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
            
            # Top panel: Data + Fit
            ax1.errorbar(x_data, y_data, yerr=y_errors, 
                        fmt='o', color='steelblue', markersize=8, 
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        label='OGLE LMC Cepheids', alpha=0.7, zorder=2)
            
            x_model = np.linspace(x_data.min(), x_data.max(), 100)
            y_model_smooth = period_luminosity_model(x_model, fitted_slope, fitted_intercept)
            ax1.plot(x_model, y_model_smooth, 'r-', linewidth=2.5, 
                    label=f'Best fit: I = {fitted_slope:.2f} log₁₀(P) + {fitted_intercept:.2f}', 
                    zorder=3)
            
            ax1.set_xlabel('log₁₀(Period [days])', fontsize=14, fontweight='bold')
            ax1.set_ylabel('I-band Magnitude', fontsize=14, fontweight='bold')
            ax1.set_title('Cepheid Period-Luminosity Relation (LMC)', 
                         fontsize=16, fontweight='bold', pad=15)
            ax1.legend(loc='best', fontsize=12, framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.invert_yaxis()
            
            # Add parameter box
            textstr = f'Slope: {fitted_slope:.3f} ± {slope_error:.3f}\n'
            textstr += f'Intercept: {fitted_intercept:.3f} ± {intercept_error:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
            
            # Bottom panel: Residuals
            ax2.errorbar(x_data, residuals, yerr=y_errors,
                        fmt='o', color='darkgreen', markersize=6,
                        capsize=3, capthick=1, elinewidth=1,
                        alpha=0.7, zorder=2)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                       label='Zero residual', zorder=1)
            ax2.set_xlabel('log₁₀(Period [days])', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Residuals (Obs - Model)', fontsize=14, fontweight='bold')
            ax2.set_title('Residuals Plot', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            rms_residual = np.sqrt(np.mean(residuals**2))
            ax2.axhline(y=rms_residual, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=-rms_residual, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            
            st.pyplot(fig)
            
            # Download button
            st.download_button(
                label="📥 Download Figure",
                data=fig,
                file_name="cepheids_pl_relation.png",
                mime="image/png"
            )
    
    # ========================================================================
    # PAGE 4: DISTANCE CALCULATOR
    # ========================================================================
    elif page == "🌌 Distance Calculator":
        st.markdown('<p class="section-header">Distance Calculator</p>', unsafe_allow_html=True)
        
        st.markdown("""
        Use the Period-Luminosity relation to estimate distances to Cepheid variables.
        Enter a period and apparent magnitude to calculate the distance.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            period = st.number_input("Period (days)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
            apparent_mag = st.number_input("Apparent Magnitude (I-band)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
        
        if 'fitted_slope' in st.session_state:
            fitted_slope = st.session_state['fitted_slope']
            fitted_intercept = st.session_state['fitted_intercept']
            
            if st.button("Calculate Distance", type="primary"):
                log10_period = np.log10(period)
                
                # Predicted apparent magnitude from our fit
                predicted_apparent = period_luminosity_model(log10_period, fitted_slope, fitted_intercept)
                
                # Simplified absolute magnitude calibration (for demonstration)
                absolute_mag = -3.0 * log10_period - 1.5
                
                # Distance modulus
                distance_modulus = apparent_mag - absolute_mag
                
                # Distance in parsecs
                distance_pc = 10 ** ((distance_modulus + 5) / 5)
                distance_kpc = distance_pc / 1000
                
                st.subheader("Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Distance Modulus", f"{distance_modulus:.2f} mag")
                with col2:
                    st.metric("Distance", f"{distance_kpc:.2f} kpc")
                with col3:
                    st.metric("Distance", f"{distance_pc:.0f} pc")
                
                st.info("ℹ️ Note: This uses a simplified absolute magnitude calibration. Real distance measurements require careful calibration.")
        else:
            st.warning("⚠️ Please fit the model first in the 'Fit Model' section.")
    
    # ========================================================================
    # PAGE 5: AI ASSISTANT (MCP)
    # ========================================================================
    elif page == "🤖 AI Assistant (MCP)":
        st.markdown('<p class="section-header">AI Assistant (MCP Integration)</p>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>About MCP:</strong> Model Context Protocol (MCP) is a simple way to integrate 
        AI assistance into applications. This is a beginner-friendly implementation that can 
        connect to OpenAI-compatible APIs.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How to Use:")
        st.markdown("""
        1. Set your API key in environment variable: `export OPENAI_API_KEY='your-key'`
        2. Or add to Streamlit secrets (`.streamlit/secrets.toml`)
        3. Ask questions about Cepheid variables, astronomy, or this project
        """)
        
        # API key input (optional, for testing)
        api_key_input = st.text_input("API Key (optional, if not in environment)", type="password")
        api_key = api_key_input if api_key_input else None
        
        # Example questions
        st.markdown("### Example Questions:")
        example_questions = [
            "What is a Cepheid variable?",
            "How does the Period-Luminosity relation work?",
            "Why are Cepheids called standard candles?",
            "Explain the distance modulus formula."
        ]
        
        selected_example = st.selectbox("Or select an example question:", 
                                       [""] + example_questions)
        
        # User input
        user_question = st.text_area("Ask a question:", 
                                    value=selected_example if selected_example else "",
                                    height=100)
        
        if st.button("🚀 Ask AI", type="primary") and user_question:
            with st.spinner("Thinking..."):
                response = simple_mcp_query(user_question, api_key=api_key)
                st.markdown("### AI Response:")
                st.write(response)
        
        # MCP Documentation
        with st.expander("📚 MCP Integration Details"):
            st.markdown("""
            **What is MCP?**
            - Model Context Protocol is a standardized way to connect applications to AI models
            - This implementation uses a simple wrapper function `simple_mcp_query()`
            
            **How it works:**
            1. Takes a user prompt/question
            2. Connects to OpenAI-compatible API (requires `openai` package)
            3. Returns AI-generated response
            
            **For Production:**
            - Use proper MCP client libraries (e.g., `mcp` package)
            - Add error handling and rate limiting
            - Implement caching for common questions
            - Add context about your specific project/data
            
            **Installation:**
            ```bash
            pip install openai
            ```
            
            **Environment Setup:**
            ```bash
            export OPENAI_API_KEY='your-api-key-here'
            ```
            """)

if __name__ == "__main__":
    main()

