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

## Usage

```bash
# Analyze an RHP
python -m rhp_analyzer analyze path/to/rhp.pdf
```

## Configuration

See `config.yaml` for configuration options.
