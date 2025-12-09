# Streamlit Web App Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure you have the data file:**
   - Run the notebook first to generate `ogle_lmc_cepheids.csv`
   - Or download it manually

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to that URL manually

## Features

The Streamlit app includes:

- **📊 Data Explorer**: View and explore the Cepheid dataset
- **🔧 Fit Model**: Fit the Period-Luminosity relation interactively
- **📈 Visualizations**: View fitted plots with residuals
- **🌌 Distance Calculator**: Calculate distances using the fitted relation
- **🤖 AI Assistant (MCP)**: Ask questions about Cepheids and astronomy

## MCP Integration

The app includes a simple MCP (Model Context Protocol) integration for AI assistance.

### Setup for AI Assistant:

1. **Install OpenAI package:**
   ```bash
   pip install openai
   ```

2. **Set your API key:**
   
   **Option 1: Environment variable**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   **Option 2: Streamlit secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

3. **Use the AI Assistant:**
   - Navigate to "🤖 AI Assistant (MCP)" in the sidebar
   - Enter your question or select an example
   - Click "Ask AI"

### Note on MCP:

This is a beginner-friendly implementation. For production use, consider:
- Using proper MCP client libraries
- Adding rate limiting
- Implementing caching
- Adding more context about your specific project

## Troubleshooting

- **Data file not found**: Make sure `ogle_lmc_cepheids.csv` exists in the same directory
- **API errors**: Check that your API key is set correctly
- **Import errors**: Make sure all packages in `requirements.txt` are installed

