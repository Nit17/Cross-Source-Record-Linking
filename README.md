# Cross-Source Record Linking

A Streamlit app for linking invoice records across two CSV sources using a tiered matching engine (exact ID, canonical ID, and composite rules with tie-breakers).

## Features

- File upload for Source A and Source B
- Semantic column mapping UI
	- Required: invoice ID, email, date, total amount
	- Optional: PO number, tax amount, currency
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

Once running, open:

- Local link: http://localhost:8501
- Cloud link (if deployed): https://cross-source-record-linking.streamlit.app
Direct link - https://cross-source-record-linking-kdgedfpbav8h63szgsstea.streamlit.app/
Project demo -https://drive.google.com/file/d/1M2lM7vBrmJXeRnxvy96QHbAYsZofwuNW/view?usp=drive_link

Tip: On macOS you can also open the local app directly from Terminal:

```bash
open http://localhost:8501
```

If you’re deploying on Streamlit Community Cloud, set the app entry point to `app.py`. After the first deploy, the public URL will be shown and will look like:

```
https://<your-username>-<repo-name>-<branch>.streamlit.app
```

## Notes
- Ensure you map the required fields for both sources before running.
- Amount tolerance is a percentage (e.g., 0.1% default).


