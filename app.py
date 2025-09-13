import streamlit as st
import pandas as pd
import json
from datetime import datetime

from src.utils import autodetect_a, autodetect_b, MappingA, MappingB, RulesConfig, AppPreset
from src.pipeline import run_pipeline

# Optional grid UI; degrade gracefully if not installed
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    HAVE_AGGRID = True
except Exception:
    HAVE_AGGRID = False

st.set_page_config(page_title="Cross-Source Record Linking", layout="wide")
st.title("Cross-Source Record Linking App")

st.sidebar.header("Control Panel")

# File upload widgets
st.sidebar.subheader("Upload Source Files")
source_a_file = st.sidebar.file_uploader("Upload Source A CSV", type=["csv"], key="source_a")
source_b_file = st.sidebar.file_uploader("Upload Source B CSV", type=["csv"], key="source_b")

# Initialize dataframes
if source_a_file is not None:
    df_a = pd.read_csv(source_a_file)
    st.write("Source A Columns:", list(df_a.columns))
else:
    df_a = None

if source_b_file is not None:
    df_b = pd.read_csv(source_b_file)
    st.write("Source B Columns:", list(df_b.columns))
else:
    df_b = None

# Column mapping UI (semantic fields + autodetect)
if df_a is not None and df_b is not None:
    st.subheader("Column Mapping")
    col_left, col_right = st.columns(2)

    # Apply pending autodetected values BEFORE creating widgets
    if st.session_state.get("apply_autodetect", False):
        auto = st.session_state.get("autodetect_values", {})
        for k, v in auto.items():
            st.session_state[k] = v
        st.session_state["apply_autodetect"] = False

    with col_left:
        st.markdown("Source A mappings")
        a_invoice_id = st.selectbox("A: Invoice ID", options=[None] + list(df_a.columns), key="a_invoice_id")
        a_email = st.selectbox("A: Customer Email", options=[None] + list(df_a.columns), key="a_email")
        a_date = st.selectbox("A: Invoice Date", options=[None] + list(df_a.columns), key="a_date")
        a_amount = st.selectbox("A: Total Amount", options=[None] + list(df_a.columns), key="a_amount")
        a_po = st.selectbox("A: PO Number (optional)", options=[None] + list(df_a.columns), key="a_po")

    with col_right:
        st.markdown("Source B mappings")
        b_ref = st.selectbox("B: Reference Code", options=[None] + list(df_b.columns), key="b_ref")
        b_email = st.selectbox("B: Customer Email", options=[None] + list(df_b.columns), key="b_email")
        b_date = st.selectbox("B: Document Date", options=[None] + list(df_b.columns), key="b_date")
        b_amount = st.selectbox("B: Grand Total", options=[None] + list(df_b.columns), key="b_amount")
        b_po = st.selectbox("B: Purchase Order (optional)", options=[None] + list(df_b.columns), key="b_po")

    # Autodetect button (set values and rerun)
    if st.button("Autodetect Mappings"):
        det_a = autodetect_a(df_a)
        det_b = autodetect_b(df_b)
        values = {}
        if det_a:
            values.update({
                "a_invoice_id": det_a.invoice_id,
                "a_email": det_a.customer_email,
                "a_date": det_a.invoice_date,
                "a_amount": det_a.total_amount,
                "a_po": getattr(det_a, 'po_number', None),
            })
        if det_b:
            values.update({
                "b_ref": det_b.ref_code,
                "b_email": det_b.email,
                "b_date": det_b.doc_date,
                "b_amount": det_b.grand_total,
                "b_po": getattr(det_b, 'purchase_order', None),
            })
        st.session_state["autodetect_values"] = values
        st.session_state["apply_autodetect"] = True
        st.rerun()

    mapping_a = MappingA(invoice_id=a_invoice_id, customer_email=a_email, invoice_date=a_date, total_amount=a_amount, po_number=a_po) if all(v is not None for v in [a_invoice_id, a_email, a_date, a_amount]) else None
    mapping_b = MappingB(ref_code=b_ref, email=b_email, doc_date=b_date, grand_total=b_amount, purchase_order=b_po) if all(v is not None for v in [b_ref, b_email, b_date, b_amount]) else None

    # Validate required mappings
    required_a = [a_invoice_id, a_email, a_date, a_amount]
    required_b = [b_ref, b_email, b_date, b_amount]
    if any(v is None for v in required_a + required_b):
        st.warning("Please select all required mappings for both sources.")
        mapping_ready = False
    else:
        mapping_ready = True

    st.info("Proceed to configure matching rules in the sidebar.")
    # --- Rule Configuration UI ---
    st.sidebar.subheader("Matching Rule Configuration")
    amount_tolerance = st.sidebar.number_input("Amount Tolerance (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.01)
    date_tolerance = st.sidebar.number_input("Date Tolerance (days)", min_value=0, max_value=30, value=2, step=1)
    name_sim = st.sidebar.slider("Name Similarity (fuzzy)", 0.0, 1.0, 0.85, 0.01)
    domain_sim = st.sidebar.slider("Domain Similarity (fuzzy)", 0.0, 1.0, 0.9, 0.01)
    rules_config = RulesConfig(amount_tolerance=amount_tolerance/100.0, date_tolerance=int(date_tolerance), name_similarity=name_sim, domain_similarity=domain_sim)

    # Presets save/load
    st.sidebar.subheader("Presets")
    preset_name = st.sidebar.text_input("Preset name", value=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    if st.sidebar.button("Save Preset") and mapping_a and mapping_b:
        preset = AppPreset(mapping_a=mapping_a, mapping_b=mapping_b, rules=rules_config)
        st.session_state.setdefault("presets", {})[preset_name] = json.loads(preset.model_dump_json())
        st.success(f"Preset '{preset_name}' saved in session.")
    # Download presets
    if st.sidebar.button("Download Presets JSON"):
        payload = json.dumps(st.session_state.get("presets", {}), indent=2)
        st.download_button("Download Now", data=payload, file_name="presets.json", mime="application/json")
    # Upload presets
    up = st.sidebar.file_uploader("Load Presets JSON", type=["json"], key="presets_json")
    if up is not None:
        data = json.load(up)
        st.session_state["presets"] = data
        st.success("Presets loaded into session.")

    # (Old inline pipeline removed; using src.pipeline.run_pipeline)

    # --- Run Linking Button ---
    # Progress and dry-run
    dry_run = st.sidebar.checkbox("Sample-size dry run (first 500 rows)")
    sample_n = 500 if dry_run else None

    if mapping_a and mapping_b and st.button("Run Linking"):
        progress_bar = st.progress(0)
        status = st.empty()

        def cb(p: float, msg: str):
            progress_bar.progress(min(100, int(p*100)))
            status.write(msg)

        matched_pairs, suspect_pairs, unmatched_a, unmatched_b = run_pipeline(df_a, df_b, mapping_a, mapping_b, rules_config, progress=cb, sample_n=sample_n)

        # Persist results in session_state
        st.session_state["matched_pairs"] = matched_pairs
        st.session_state["suspect_pairs"] = suspect_pairs
        st.session_state["unmatched_a"] = unmatched_a
        st.session_state["unmatched_b"] = unmatched_b

        # Logging
        new_logs = [
            "Run Linking triggered at user request.",
            f"Total records in Source A: {len(df_a)}",
            f"Total records in Source B: {len(df_b)}",
            f"Matched pairs: {len(matched_pairs)}",
            f"Suspect pairs: {len(suspect_pairs)}",
            f"Unmatched Source A: {len(unmatched_a)}",
            f"Unmatched Source B: {len(unmatched_b)}",
        ]
        st.session_state.setdefault("logs", []).extend(new_logs)
        st.success(f"Linking complete! {len(matched_pairs)} matched, {len(suspect_pairs)} suspects.")

    # If results exist, display them
    if "matched_pairs" in st.session_state:
        matched_pairs = st.session_state["matched_pairs"]
        suspect_pairs = st.session_state["suspect_pairs"]
        unmatched_a = st.session_state["unmatched_a"]
        unmatched_b = st.session_state["unmatched_b"]

        # --- Metrics Dashboard ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Records in A", len(df_a))
        col2.metric("Total Records in B", len(df_b))
        col3.metric("Matched", len(matched_pairs))
        col4.metric("Suspect", len(suspect_pairs))
        col5.metric("Unmatched", len(unmatched_a) + len(unmatched_b))

        # --- Results Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["Matched", "Suspects", "Unmatched", "Needs Attention"])

        with tab1:
            st.subheader("Matched Records")
            for i, m in enumerate(matched_pairs):
                with st.expander(f"Match {i+1} - Tier {getattr(m, 'tier', '')}"):
                    st.write("Source A Record:", getattr(m, 'a_row', m))
                    st.write("Source B Record:", getattr(m, 'b_row', m))
                    st.info(f"Rationale: {getattr(m, 'rationale', '')}")
            if matched_pairs:
                matched_df = pd.DataFrame([
                    {**m.a_row, **{f"B_{k}": v for k, v in m.b_row.items()}, "rationale": m.rationale, "tier": m.tier, "score": m.score} if hasattr(m, 'a_row') else m for m in matched_pairs
                ])
                csv = matched_df.to_csv(index=False)
                st.download_button("Download Matched Results CSV", data=csv, file_name="matched_results.csv", mime="text/csv")
                if HAVE_AGGRID:
                    gob = GridOptionsBuilder.from_dataframe(matched_df)
                    gob.configure_default_column(filter=True, sortable=True, resizable=True)
                    AgGrid(matched_df, gridOptions=gob.build(), height=300)

        with tab2:
            st.subheader("Suspect Records")
            needs_attention = st.session_state.setdefault("needs_attention", [])
            for i, s in enumerate(suspect_pairs):
                with st.expander(f"Suspect {i+1} - {getattr(s, 'tier', '')}"):
                    st.write("Source A Record:", getattr(s, 'a_row', s))
                    st.write("Source B Record:", getattr(s, 'b_row', s))
                    st.info(f"Rationale: {getattr(s, 'rationale', '')}")
                    cols = st.columns(3)
                    if cols[0].button(f"Accept #{i+1}"):
                        st.session_state.setdefault("accepted", []).append(s)
                    if cols[1].button(f"Reject #{i+1}"):
                        st.session_state.setdefault("rejected", []).append(s)
                    if cols[2].button(f"Defer #{i+1}"):
                        needs_attention.append(s)

        with tab3:
            st.subheader("Unmatched Records")
            st.write("Unmatched Source A Records:", unmatched_a)
            st.write("Unmatched Source B Records:", unmatched_b)

        with tab4:
            st.subheader("Needs Attention")
            if st.session_state.get("needs_attention"):
                df_na = pd.DataFrame([{**x.a_row, **{f"B_{k}": v for k, v in x.b_row.items()}, "tier": x.tier} for x in st.session_state["needs_attention"] if hasattr(x, 'a_row')])
                if HAVE_AGGRID:
                    AgGrid(df_na, height=250)
                else:
                    st.dataframe(df_na, height=250)

        # --- Print Logs ---
        with st.expander("Run Logs"):
            cols = st.columns([1,1,6])
            if cols[0].button("Clear Logs"):
                st.session_state["logs"] = []
            for log in st.session_state.get("logs", []):
                st.write(log)
else:
    st.warning("Please upload both Source A and Source B CSV files to continue.")
