# MCP Server for Fitting Pipeline

A simple, beginner-friendly MCP (Model Context Protocol) server that exposes tools for fitting the Cepheid Period-Luminosity relation.

## What is This?

This MCP server allows external applications (like AI assistants or other tools) to interact with your fitting pipeline through standardized MCP protocol. Instead of running Python code directly, clients can call tools like "fit_period_luminosity" or "predict_magnitude" through MCP.

## Installation

1. Install the MCP SDK:
```bash
pip install mcp
```

Or if you already have requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run as MCP Server (Recommended)

The server communicates via stdio (standard input/output) following the MCP protocol:

```bash
python mcp_server.py
```

The server will listen for MCP client connections. You'll need an MCP client to interact with it (like Claude Desktop with MCP support, or custom clients).

### Option 2: Simple Command-Line Interface (Fallback)

If MCP SDK is not installed, the server falls back to a simple command-line interface:

```bash
python mcp_server.py
```

Then use commands:
- `load ogle_lmc_cepheids.csv` - Load data
- `fit` - Fit the Period-Luminosity relation
- `fit 0.15` - Fit with custom error assumption (0.15 mag)
- `params` - Get fitted parameters
- `predict 10.0` - Predict magnitude for 10-day period
- `quit` - Exit

## Available Tools

The server exposes these tools:

### 1. `load_data`
Load Cepheid data from CSV file.

**Parameters:**
- `filepath` (string): Path to CSV file (e.g., "ogle_lmc_cepheids.csv")

**Example:**
```json
{
  "filepath": "ogle_lmc_cepheids.csv"
}
```

### 2. `fit_period_luminosity`
Fit the Period-Luminosity relation to loaded data.

**Parameters:**
- `error_assumption` (number, optional): Assumed measurement uncertainty in magnitudes (default: 0.1)

**Returns:**
- Fitted slope and intercept with uncertainties
- Chi-squared, reduced chi-squared, R-squared
- RMS residual

**Example:**
```json
{
  "error_assumption": 0.1
}
```

### 3. `get_fitted_parameters`
Get the currently fitted parameters.

**Returns:**
- Slope, intercept, and their uncertainties
- Fitted equation string

### 4. `predict_magnitude`
Predict magnitude for a given period using fitted parameters.

**Parameters:**
- `period` (number): Period in days

**Example:**
```json
{
  "period": 10.0
}
```

## Example Workflow

1. **Load data:**
   ```json
   {"tool": "load_data", "arguments": {"filepath": "ogle_lmc_cepheids.csv"}}
   ```

2. **Fit the relation:**
   ```json
   {"tool": "fit_period_luminosity", "arguments": {"error_assumption": 0.1}}
   ```

3. **Get parameters:**
   ```json
   {"tool": "get_fitted_parameters"}
   ```

4. **Predict magnitude:**
   ```json
   {"tool": "predict_magnitude", "arguments": {"period": 10.0}}
   ```

## Integration with MCP Clients

### Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "cepheids-fitting": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```

### Custom Clients

The server follows the MCP protocol over stdio. Any MCP-compatible client can connect to it.

## Code Structure

The server is organized simply:

1. **Core Functions** (lines ~30-150): The actual fitting logic
   - `period_luminosity_model()` - Model function
   - `load_data()` - Load CSV data
   - `fit_period_luminosity()` - Perform fitting
   - `get_fitted_parameters()` - Get results
   - `predict_magnitude()` - Make predictions

2. **MCP Server** (lines ~150+): MCP protocol implementation
   - `list_tools()` - Expose available tools
   - `call_tool()` - Handle tool calls
   - `main()` - Run server

3. **Fallback CLI**: Simple command-line interface if MCP SDK unavailable

## Beginner-Friendly Features

- ✅ Clear comments explaining each function
- ✅ Simple error handling
- ✅ JSON responses for easy parsing
- ✅ Fallback CLI if MCP SDK not installed
- ✅ No complex dependencies beyond scipy/numpy/pandas

## Troubleshooting

**"MCP SDK not installed"**
- Install with: `pip install mcp`
- Or use the fallback CLI mode

**"No data loaded"**
- Make sure to call `load_data` before fitting

**"No fit available"**
- Run `fit_period_luminosity` before getting parameters or making predictions

## Notes

- The server stores fitted parameters in memory (global variables)
- Data must be loaded before fitting
- Fitting must be done before getting parameters or predictions
- Simple and efficient - perfect for beginners!

