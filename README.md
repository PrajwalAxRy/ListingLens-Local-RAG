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

**Configuration Precedence** (highest to lowest):
1. Environment variables (prefix: `RHP_`, use `__` for nesting)
2. Custom config file (via `--config` flag)
3. Default `config.yaml` in project root
4. Built-in defaults in code

**Environment Variable Examples**:
```bash
# Set log level
set RHP_LOG_LEVEL=DEBUG

# Override LLM timeout (Windows)
set RHP_LLM__TIMEOUT=180

# Set HuggingFace token (required for LLM access)
set HF_TOKEN=your_token_here
```

For all configuration options, see `config.yaml` and `config.example.yaml`.

## Development

### Setting Up the Development Environment

```bash
# Clone and navigate to project
git clone <repository-url>
cd listing-lens-RAG-Test

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src/rhp_analyzer --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_config.py -v

# Run with verbose output
pytest tests/ -v
```

### Code Quality

The project uses Ruff for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

Pre-commit hooks are configured - install with:
```bash
pre-commit install
```

### Project Structure

```
src/rhp_analyzer/
├── cli/              # Command-line interface (Typer)
│   ├── main.py       # CLI entry point with global options
│   └── commands/     # Individual CLI commands
├── config/           # Configuration management
│   ├── loader.py     # YAML + env var loading with precedence
│   └── schema.py     # Pydantic models for validation
├── ingestion/        # PDF processing (Phase 2)
├── storage/          # Vector DB and SQL storage (Phase 3)
├── agents/           # LLM analysis agents (Phase 4)
├── reporting/        # Report generation (Phase 5)
└── utils/            # Shared utilities
    ├── log_setup.py  # Loguru configuration
    └── progress.py   # Rich progress display
```

### Logging

Logs are written to the `logs/` directory with daily rotation:
- `logs/YYYY-MM-DD.log` - All logs (INFO and above)
- `logs/errors.log` - Error-only log for quick debugging

Use `--verbose` flag for DEBUG-level console output.

## License

See [LICENSE](LICENSE) for details.
