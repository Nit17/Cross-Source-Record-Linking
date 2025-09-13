# Cross-Source Record Linking

A Streamlit app for linking invoice records across two CSV sources using a tiered matching engine (exact ID, canonical ID, and composite rules with tie-breakers).

## Features
- File upload for Source A and Source B
- Semantic column mapping UI
- Configurable tolerances (amount %, date days)
- Tiered matching with suspects and rationale
- Metrics dashboard and CSV export
- Run logs with a clear button

## Requirements
- Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes
- Ensure you map the required fields for both sources before running.
- Amount tolerance is a percentage (e.g., 0.1% default).
