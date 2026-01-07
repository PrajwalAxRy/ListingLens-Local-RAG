# RHP Analyzer

RHP Analyzer is a local, AI-powered CLI application that processes Indian IPO Red Herring Prospectus (RHP) documents and generates comprehensive investment analysis reports.

## Installation

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv .venv`
3.  Activate the virtual environment:
    *   Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
    *   Windows CMD: `.\.venv\Scripts\activate.bat`
    *   Linux/Mac: `source .venv/bin/activate`
4.  Install runtime dependencies: `pip install -r requirements.txt`
5.  (Recommended for contributors) Install dev tooling: `pip install -r requirements-dev.txt`
6.  Install the project in editable mode so the CLI is discoverable: `pip install -e .`

## CLI Commands

The RHP Analyzer provides two equivalent ways to access the CLI:

- **Console script** (after `pip install -e .`): `rhp-analyzer`
- **Python module**: `python -m rhp_analyzer`

Both provide identical functionality. Examples below use the console script form.

### Global Options

These options work with all commands:

- `--version` / `-v`: Show version and exit
  ```bash
  rhp-analyzer --version
  # Output: RHP Analyzer version 0.1.0
  ```

- `--config` / `-c FILE`: Specify custom configuration file (default: `./config.yaml`)
  ```bash
  rhp-analyzer --config custom.yaml analyze file.pdf
  ```

- `--verbose`: Enable DEBUG level logging (detailed output in `logs/` directory)
  ```bash
  rhp-analyzer --verbose analyze file.pdf
  ```

- `--help`: Display help message and exit
  ```bash
  rhp-analyzer --help
  ```

### analyze - Full RHP Analysis

Analyze an RHP document and generate comprehensive investment report.

**Pipeline**: Processes PDF through: ingestion → analysis → report generation

**Syntax**:
```bash
rhp-analyzer analyze PDF_PATH [OPTIONS]
```

**Arguments**:
- `PDF_PATH` (required): Path to RHP PDF file to analyze

**Options**:
- `--output-dir` / `-o PATH`: Custom output directory (default: from config.yaml)
- `--dry-run`: Validate input without performing actual analysis

**Examples**:
```bash
# Basic analysis
rhp-analyzer analyze path/to/rhp.pdf

# Custom output directory
rhp-analyzer analyze rhp.pdf --output-dir ./reports

# Dry run validation
rhp-analyzer analyze rhp.pdf --dry-run

# Using Python module form
python -m rhp_analyzer analyze path/to/rhp.pdf
```

### validate - Quick PDF Validation

Validate an RHP document without performing full analysis.

**Purpose**: Quick pre-analysis checks to verify PDF readability and structure

**Syntax**:
```bash
rhp-analyzer validate PDF_PATH
```

**Quick Checks Performed**:
- PDF readability and integrity
- Basic structure validation
- Section detection
- Page count verification

**Arguments**:
- `PDF_PATH` (required): Path to RHP PDF file to validate

**Examples**:
```bash
# Quick validation
rhp-analyzer validate path/to/rhp.pdf

# Using Python module form
python -m rhp_analyzer validate path/to/rhp.pdf
```

### config - Display Configuration

Display current configuration settings merged from all sources.

**Configuration Precedence**: Environment variables > `--config` file > `config.yaml` > Built-in defaults

**Syntax**:
```bash
rhp-analyzer config
```

**Output Sections** (5 Rich-formatted tables):
- **Paths Configuration**: input_dir, output_dir, logs_dir, data_dir
- **LLM Configuration**: provider, context_model, reasoning_model, temperature, max_tokens, timeout
- **Ingestion Configuration**: chunk_size, chunk_overlap, min_chunk_size, batch_size
- **Enabled Agents**: architect, forensic, red_flag, governance, legal, summarizer, critic
- **Reporting Configuration**: formats, template, include_appendices

**Examples**:
```bash
# Display all configuration
rhp-analyzer config

# Check config with custom file
rhp-analyzer --config custom.yaml config

# Using Python module form
python -m rhp_analyzer config
```

### Common Workflows

Practical multi-command usage patterns:

```bash
# Validate before analyzing
rhp-analyzer validate path/to/rhp.pdf && rhp-analyzer analyze path/to/rhp.pdf

# Check configuration first
rhp-analyzer config
rhp-analyzer analyze path/to/rhp.pdf

# Custom config with dry run test
rhp-analyzer --config production.yaml analyze rhp.pdf --dry-run
rhp-analyzer --config production.yaml analyze rhp.pdf --output-dir ./output

# Verbose logging for debugging
rhp-analyzer --verbose analyze path/to/rhp.pdf
```

## Configuration

The configuration file (`config.yaml`) defines settings for all components. Use the `rhp-analyzer config` command to view the current merged configuration from all sources (environment variables, custom config file, defaults).

For detailed configuration options, see `config.yaml`.
