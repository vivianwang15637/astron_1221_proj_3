"""
Simple MCP Server for Cepheid Period-Luminosity Fitting Pipeline

This server exposes tools for fitting the Period-Luminosity relation
using the MCP (Model Context Protocol) standard.

Usage:
    python mcp_server.py

The server will start and listen for MCP client connections.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Try to import MCP SDK, fallback to simple implementation if not available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP SDK not installed. Install with: pip install mcp")
    print("   Using simple server implementation instead.")

# ============================================================================
# FITTING FUNCTIONS (Core Logic)
# ============================================================================

def period_luminosity_model(log10_period: float, slope: float, intercept: float) -> float:
    """
    Simple linear model: magnitude = slope * log10(period) + intercept
    
    This is the Period-Luminosity relation equation.
    """
    return slope * log10_period + intercept


# Global storage for fitted parameters (simple approach)
_fitted_params = {
    'slope': None,
    'intercept': None,
    'slope_error': None,
    'intercept_error': None,
    'data': None
}


def load_data(filepath: str) -> Dict[str, Any]:
    """
    Load Cepheid data from CSV file.
    
    Returns dictionary with data and status.
    """
    try:
        df = pd.read_csv(filepath)
        df['log10_Period'] = np.log10(df['Period'])
        
        _fitted_params['data'] = {
            'periods': df['Period'].tolist(),
            'magnitudes': df['I_mag'].tolist(),
            'log10_periods': df['log10_Period'].tolist()
        }
        
        return {
            'success': True,
            'message': f'Loaded {len(df)} Cepheids from {filepath}',
            'num_stars': len(df),
            'period_range': [float(df['Period'].min()), float(df['Period'].max())],
            'magnitude_range': [float(df['I_mag'].min()), float(df['I_mag'].max())]
        }
    except Exception as e:
        return {'success': False, 'message': f'Error loading data: {str(e)}'}


def fit_period_luminosity(error_assumption: float = 0.1) -> Dict[str, Any]:
    """
    Fit the Period-Luminosity relation to loaded data.
    
    Parameters:
    -----------
    error_assumption : float
        Assumed measurement uncertainty in magnitudes (default: 0.1)
    
    Returns:
    --------
    Dictionary with fit results and statistics
    """
    if _fitted_params['data'] is None:
        return {'success': False, 'message': 'No data loaded. Load data first.'}
    
    try:
        # Extract data
        x_data = np.array(_fitted_params['data']['log10_periods'])
        y_data = np.array(_fitted_params['data']['magnitudes'])
        y_errors = np.full_like(y_data, error_assumption)
        
        # Initial parameter guess
        initial_slope = (y_data.max() - y_data.min()) / (x_data.max() - x_data.min())
        initial_slope = -abs(initial_slope)  # Negative slope expected
        initial_intercept = y_data.mean()
        initial_guess = [initial_slope, initial_intercept]
        
        # Perform fitting
        popt, pcov = curve_fit(
            period_luminosity_model,
            x_data,
            y_data,
            p0=initial_guess,
            sigma=y_errors,
            absolute_sigma=True
        )
        
        # Extract parameters and uncertainties
        fitted_slope = float(popt[0])
        fitted_intercept = float(popt[1])
        param_errors = np.sqrt(np.diag(pcov))
        slope_error = float(param_errors[0])
        intercept_error = float(param_errors[1])
        
        # Store results
        _fitted_params['slope'] = fitted_slope
        _fitted_params['intercept'] = fitted_intercept
        _fitted_params['slope_error'] = slope_error
        _fitted_params['intercept_error'] = intercept_error
        
        # Calculate statistics
        y_model = period_luminosity_model(x_data, fitted_slope, fitted_intercept)
        residuals = y_data - y_model
        chi2 = float(np.sum((residuals / y_errors) ** 2))
        dof = len(y_data) - 2
        reduced_chi2 = float(chi2 / dof)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
        r_squared = float(1 - (ss_res / ss_tot))
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
        
        return {
            'success': True,
            'message': 'Fitting completed successfully',
            'slope': fitted_slope,
            'intercept': fitted_intercept,
            'slope_error': slope_error,
            'intercept_error': intercept_error,
            'chi_squared': chi2,
            'reduced_chi_squared': reduced_chi2,
            'r_squared': r_squared,
            'rms_residual': rms_residual,
            'degrees_of_freedom': dof,
            'equation': f'I_mag = ({fitted_slope:.3f} ± {slope_error:.3f}) × log₁₀(P) + ({fitted_intercept:.3f} ± {intercept_error:.3f})'
        }
    except Exception as e:
        return {'success': False, 'message': f'Fitting failed: {str(e)}'}


def get_fitted_parameters() -> Dict[str, Any]:
    """Get the currently fitted parameters."""
    if _fitted_params['slope'] is None:
        return {'success': False, 'message': 'No fit available. Run fitting first.'}
    
    return {
        'success': True,
        'slope': _fitted_params['slope'],
        'intercept': _fitted_params['intercept'],
        'slope_error': _fitted_params['slope_error'],
        'intercept_error': _fitted_params['intercept_error'],
        'equation': f'I_mag = ({_fitted_params["slope"]:.3f} ± {_fitted_params["slope_error"]:.3f}) × log₁₀(P) + ({_fitted_params["intercept"]:.3f} ± {_fitted_params["intercept_error"]:.3f})'
    }


def predict_magnitude(period: float) -> Dict[str, Any]:
    """
    Predict magnitude for a given period using fitted parameters.
    
    Parameters:
    -----------
    period : float
        Period in days
    
    Returns:
    --------
    Dictionary with predicted magnitude
    """
    if _fitted_params['slope'] is None:
        return {'success': False, 'message': 'No fit available. Run fitting first.'}
    
    log10_period = np.log10(period)
    predicted_mag = period_luminosity_model(
        log10_period,
        _fitted_params['slope'],
        _fitted_params['intercept']
    )
    
    return {
        'success': True,
        'period': period,
        'log10_period': float(log10_period),
        'predicted_magnitude': float(predicted_mag)
    }


# ============================================================================
# MCP SERVER IMPLEMENTATION
# ============================================================================

if MCP_AVAILABLE:
    # Create MCP server
    server = Server("cepheids-fitting-server")
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List all available tools."""
        return [
            Tool(
                name="load_data",
                description="Load Cepheid data from CSV file. Required before fitting.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to CSV file (e.g., 'ogle_lmc_cepheids.csv')"
                        }
                    },
                    "required": ["filepath"]
                }
            ),
            Tool(
                name="fit_period_luminosity",
                description="Fit the Period-Luminosity relation to loaded data. Returns fitted parameters and statistics.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "error_assumption": {
                            "type": "number",
                            "description": "Assumed measurement uncertainty in magnitudes (default: 0.1)",
                            "default": 0.1
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_fitted_parameters",
                description="Get the currently fitted parameters (slope, intercept, uncertainties).",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="predict_magnitude",
                description="Predict magnitude for a given period using fitted parameters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "number",
                            "description": "Period in days"
                        }
                    },
                    "required": ["period"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "load_data":
                result = load_data(arguments.get("filepath", "ogle_lmc_cepheids.csv"))
            elif name == "fit_period_luminosity":
                error_assumption = arguments.get("error_assumption", 0.1)
                result = fit_period_luminosity(error_assumption)
            elif name == "get_fitted_parameters":
                result = get_fitted_parameters()
            elif name == "predict_magnitude":
                period = arguments.get("period")
                if period is None:
                    result = {'success': False, 'message': 'Period parameter required'}
                else:
                    result = predict_magnitude(period)
            else:
                result = {'success': False, 'message': f'Unknown tool: {name}'}
            
            # Format result as JSON string
            result_str = json.dumps(result, indent=2)
            return [TextContent(type="text", text=result_str)]
            
        except Exception as e:
            error_result = {'success': False, 'message': f'Error: {str(e)}'}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def main():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    if __name__ == "__main__":
        print("🚀 Starting MCP Server for Cepheid Fitting Pipeline...")
        print("   Server will listen for MCP client connections via stdio")
        asyncio.run(main())

else:
    # Simple fallback: command-line interface
    def main_simple():
        """Simple command-line interface when MCP SDK not available."""
        print("=" * 70)
        print("Cepheid Period-Luminosity Fitting Pipeline")
        print("=" * 70)
        print("\nAvailable commands:")
        print("  1. load <filepath>     - Load data from CSV")
        print("  2. fit [error]         - Fit Period-Luminosity relation")
        print("  3. params              - Get fitted parameters")
        print("  4. predict <period>    - Predict magnitude for period")
        print("  5. quit                - Exit")
        print()
        
        while True:
            try:
                cmd = input("> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == "quit" or cmd[0] == "exit":
                    break
                elif cmd[0] == "load" and len(cmd) > 1:
                    result = load_data(cmd[1])
                    print(json.dumps(result, indent=2))
                elif cmd[0] == "fit":
                    error = float(cmd[1]) if len(cmd) > 1 else 0.1
                    result = fit_period_luminosity(error)
                    print(json.dumps(result, indent=2))
                elif cmd[0] == "params":
                    result = get_fitted_parameters()
                    print(json.dumps(result, indent=2))
                elif cmd[0] == "predict" and len(cmd) > 1:
                    period = float(cmd[1])
                    result = predict_magnitude(period)
                    print(json.dumps(result, indent=2))
                else:
                    print("Unknown command. Type 'quit' to exit.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")
    
    if __name__ == "__main__":
        main_simple()

