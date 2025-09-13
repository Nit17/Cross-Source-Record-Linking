import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime

from src.utils import autodetect_a, autodetect_b, MappingA, MappingB, RulesConfig, AppPreset
from src.pipeline import run_pipeline

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    HAVE_AGGRID = True
except Exception:
    HAVE_AGGRID = False

st.set_page_config(page_title="Cross-Source Record Linking", layout="wide")

# Lightweight styling
st.markdown(
        """
        <style>
            .app-subtitle { color:#4b5563; font-size:0.95rem; margin-top:-0.6rem; }
            .card { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:1rem; }
            .section-title { font-weight:600; font-size:1.05rem; margin-bottom:0.4rem; }
            .muted { color:#6b7280; }
            .spacer { height:0.5rem; }
            .tight-metric > div { padding-top:0.2rem; padding-bottom:0.2rem; }
            .small-note { font-size:0.85rem; color:#6b7280; }
        </style>
        """,
        unsafe_allow_html=True,
)

aggrid_theme = "streamlit"

# Optional dark overrides for custom elements (Streamlit global theme is controlled by Settings or config)

st.title("Cross-Source Record Linking")
st.markdown("<div class='app-subtitle'>Match and reconcile invoices across two CSV sources with exact, composite, and fuzzy rules.</div>", unsafe_allow_html=True)
st.divider()

st.sidebar.header("Control Panel")
st.sidebar.subheader("Upload Source Files")
source_a_file = st.sidebar.file_uploader("Upload Source A CSV", type=["csv"], key="source_a")
source_b_file = st.sidebar.file_uploader("Upload Source B CSV", type=["csv"], key="source_b")

if source_a_file is not None:
    df_a = pd.read_csv(source_a_file)
    with st.expander("Source A preview", expanded=False):
        st.caption("First 10 rows")
        st.dataframe(df_a.head(10), use_container_width=True)
else:
    df_a = None

if source_b_file is not None:
    df_b = pd.read_csv(source_b_file)
    with st.expander("Source B preview", expanded=False):
        st.caption("First 10 rows")
        st.dataframe(df_b.head(10), use_container_width=True)
else:
    df_b = None

if df_a is not None and df_b is not None:
    st.subheader("Column Mapping")
    st.markdown("<div class='small-note'>Map your CSV columns to semantic fields. You can try Autodetect to pre-fill suggestions.</div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    if st.session_state.get("apply_autodetect", False):
        auto = st.session_state.get("autodetect_values", {})
        for k, v in auto.items():
            st.session_state[k] = v
        st.session_state["apply_autodetect"] = False

    with col_left:
        st.markdown("<div class='section-title'>Source A mappings</div>", unsafe_allow_html=True)
        a_invoice_id = st.selectbox("A: Invoice ID", options=[None] + list(df_a.columns), key="a_invoice_id", help="Unique invoice identifier")
        a_email = st.selectbox("A: Customer Email", options=[None] + list(df_a.columns), key="a_email", help="Customer contact email")
        a_date = st.selectbox("A: Invoice Date", options=[None] + list(df_a.columns), key="a_date", help="Invoice issue date")
        a_amount = st.selectbox("A: Total Amount", options=[None] + list(df_a.columns), key="a_amount", help="Invoice total amount")
        a_po = st.selectbox("A: PO Number (optional)", options=[None] + list(df_a.columns), key="a_po", help="Purchase order number if available")

    with col_right:
        st.markdown("<div class='section-title'>Source B mappings</div>", unsafe_allow_html=True)
        b_ref = st.selectbox("B: Reference Code", options=[None] + list(df_b.columns), key="b_ref", help="External or ERP reference code")
        b_email = st.selectbox("B: Customer Email", options=[None] + list(df_b.columns), key="b_email", help="Customer contact email")
        b_date = st.selectbox("B: Document Date", options=[None] + list(df_b.columns), key="b_date", help="Document date (equiv. to invoice date)")
        b_amount = st.selectbox("B: Grand Total", options=[None] + list(df_b.columns), key="b_amount", help="Document total amount")
        b_po = st.selectbox("B: Purchase Order (optional)", options=[None] + list(df_b.columns), key="b_po", help="Purchase order number if available")

    cols_tools = st.columns([1,1,6])
    if cols_tools[0].button("Autodetect Mappings"):
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

    if cols_tools[1].button("Reset Mapping"):
        for k in [
            "a_invoice_id","a_email","a_date","a_amount","a_po",
            "b_ref","b_email","b_date","b_amount","b_po"
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_set_query_params()  # clears widget state in URL
        st.rerun()

    mapping_a = MappingA(invoice_id=a_invoice_id, customer_email=a_email, invoice_date=a_date, total_amount=a_amount, po_number=a_po) if all(v is not None for v in [a_invoice_id, a_email, a_date, a_amount]) else None
    mapping_b = MappingB(ref_code=b_ref, email=b_email, doc_date=b_date, grand_total=b_amount, purchase_order=b_po) if all(v is not None for v in [b_ref, b_email, b_date, b_amount]) else None

    required_a = [a_invoice_id, a_email, a_date, a_amount]
    required_b = [b_ref, b_email, b_date, b_amount]
    if any(v is None for v in required_a + required_b):
        st.warning("Please select all required mappings for both sources.")

    st.info("Proceed to configure matching rules in the sidebar.")
    st.sidebar.subheader("Matching Rule Configuration")
    amount_tolerance = st.sidebar.number_input("Amount Tolerance (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.01, key="amount_tolerance")
    date_tolerance = st.sidebar.number_input("Date Tolerance (days)", min_value=0, max_value=30, value=2, step=1, key="date_tolerance")
    name_sim = st.sidebar.slider("Name Similarity (fuzzy)", 0.0, 1.0, 0.85, 0.01, key="name_sim")
    domain_sim = st.sidebar.slider("Domain Similarity (fuzzy)", 0.0, 1.0, 0.9, 0.01, key="domain_sim")

    st.sidebar.caption("Weights for tie-breaking (will be normalized)")
    w_amount = st.sidebar.slider("Weight: Amount", 0.0, 1.0, 0.6, 0.01, key="w_amount")
    w_date = st.sidebar.slider("Weight: Date", 0.0, 1.0, 0.3, 0.01, key="w_date")
    w_name = st.sidebar.slider("Weight: Name", 0.0, 1.0, 0.08, 0.01, key="w_name")
    w_domain = st.sidebar.slider("Weight: Email Domain", 0.0, 1.0, 0.02, 0.01, key="w_domain")

    rules_config = RulesConfig(
        amount_tolerance=amount_tolerance/100.0,
        date_tolerance=int(date_tolerance),
        name_similarity=name_sim,
        domain_similarity=domain_sim,
        weight_amount=w_amount,
        weight_date=w_date,
        weight_name=w_name,
        weight_domain=w_domain,
    )

    st.sidebar.subheader("Rule Tier Order")
    default_order = ["exact_id", "canonical_id", "composite", "fuzzy"]
    order_selects = []
    used = set()
    for i in range(4):
        options = [o for o in default_order if o not in used]
        val = st.sidebar.selectbox(f"Tier {i+1}", options=options, index=0, key=f"tier_{i}")
        order_selects.append(val)
        used.add(val)
    st.session_state["rule_order"] = order_selects

    st.sidebar.subheader("Presets")
    preset_name = st.sidebar.text_input("Preset name", value=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    if st.sidebar.button("Save Preset") and mapping_a and mapping_b:
        preset = AppPreset(
            mapping_a=mapping_a,
            mapping_b=mapping_b,
            rules=rules_config,
            rule_order=order_selects,
            patterns=st.session_state.get("patterns"),
        )
        st.session_state.setdefault("presets", {})[preset_name] = json.loads(preset.model_dump_json())
        st.success(f"Preset '{preset_name}' saved in session.")
    payload = json.dumps(st.session_state.get("presets", {}), indent=2)
    st.sidebar.download_button("Download Presets JSON", data=payload, file_name="presets.json", mime="application/json")
    up = st.sidebar.file_uploader("Load Presets JSON", type=["json"], key="presets_json")
    if up is not None:
        data = json.load(up)
        st.session_state["presets"] = data
        st.success("Presets loaded into session.")
    with st.sidebar.expander("Pattern Manager", expanded=False):
        pats = st.session_state.setdefault("patterns", [])
        if pats:
            for idx, p in enumerate(list(pats)):
                cols = st.columns([4,2,2,2])
                cols[0].write(f"{p.get('apply_to','B')} • {p.get('kind')} • {p.get('value','')} {('/'+str(p.get('replacement','')) if p.get('replacement') else '')}")
                enabled_key = f"pat_enabled_{idx}"
                cur_en = p.get("enabled", True)
                new_en = cols[1].checkbox("Enabled", value=cur_en, key=enabled_key)
                p["enabled"] = bool(new_en)
                if cols[2].button("Test", key=f"pat_test_{idx}"):
                    st.session_state["_pat_test_idx"] = idx
                if cols[3].button("Remove", key=f"pat_rm_{idx}"):
                    pats.pop(idx)
                    st.experimental_rerun()
            if "_pat_test_idx" in st.session_state:
                ti = st.session_state.pop("_pat_test_idx")
                st.write("Test pattern against sample:")
                sample = st.text_input("Sample input", value="INV-123-XYZ", key="pat_test_sample")
                from src.utils import Pattern as _Pattern, apply_patterns as _apply
                try:
                    pm = _Pattern(**pats[ti])
                    st.write("Result:", _apply(sample, [pm]))
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("No patterns yet. Adopt from Suspects to add here.")
    # Apply preset to widgets/session
    if st.session_state.get("presets"):
        names = list(st.session_state["presets"].keys())
        sel = st.sidebar.selectbox("Apply Preset", options=["(select)"] + names, index=0, key="apply_preset_name")
        if sel != "(select)" and st.sidebar.button("Apply", key="apply_preset_btn"):
            p = st.session_state["presets"][sel]
            # set mapping selectboxes
            st.session_state["a_invoice_id"] = p["mapping_a"]["invoice_id"]
            st.session_state["a_email"] = p["mapping_a"]["customer_email"]
            st.session_state["a_date"] = p["mapping_a"]["invoice_date"]
            st.session_state["a_amount"] = p["mapping_a"]["total_amount"]
            st.session_state["a_po"] = p["mapping_a"].get("po_number")
            st.session_state["b_ref"] = p["mapping_b"]["ref_code"]
            st.session_state["b_email"] = p["mapping_b"]["email"]
            st.session_state["b_date"] = p["mapping_b"]["doc_date"]
            st.session_state["b_amount"] = p["mapping_b"]["grand_total"]
            st.session_state["b_po"] = p["mapping_b"].get("purchase_order")
            # rules
            st.session_state["amount_tolerance"] = float(p["rules"]["amount_tolerance"] * 100.0)
            st.session_state["date_tolerance"] = int(p["rules"]["date_tolerance"])
            st.session_state["name_sim"] = float(p["rules"]["name_similarity"])
            st.session_state["domain_sim"] = float(p["rules"].get("domain_similarity", 0.9))
            st.session_state["w_amount"] = float(p["rules"].get("weight_amount", 0.6))
            st.session_state["w_date"] = float(p["rules"].get("weight_date", 0.3))
            st.session_state["w_name"] = float(p["rules"].get("weight_name", 0.08))
            st.session_state["w_domain"] = float(p["rules"].get("weight_domain", 0.02))
            # order
            order = p.get("rule_order", ["exact_id","canonical_id","composite","fuzzy"])[:4]
            while len(order) < 4:
                for opt in ["exact_id","canonical_id","composite","fuzzy"]:
                    if opt not in order:
                        order.append(opt)
                        break
            for i, val in enumerate(order):
                st.session_state[f"tier_{i}"] = val
            # patterns
            st.session_state["patterns"] = p.get("patterns", [])
            st.rerun()

    # Run Linking
    dry_run = st.sidebar.checkbox("Sample-size dry run (first 500 rows)")
    sample_n = 500 if dry_run else None

    if mapping_a and mapping_b and st.button("Run Linking"):
        progress_bar = st.progress(0)
        status = st.empty()

        def cb(p: float, msg: str):
            progress_bar.progress(min(100, int(p*100)))
            status.write(msg)
        # Run pipeline
        # Prepare optional rule order and adopted patterns
        rule_order = st.session_state.get("rule_order")
        patterns_raw = st.session_state.get("patterns", [])
        try:
            from src.utils import Pattern as _Pattern
            patt_models = []
            for d in patterns_raw:
                try:
                    patt_models.append(_Pattern(**d))
                except Exception:
                    continue
            patterns_a = [p for p in patt_models if p.apply_to in ("A", "both")]
            patterns_b = [p for p in patt_models if p.apply_to in ("B", "both")]
        except Exception:
            rule_order = rule_order
            patterns_a = None
            patterns_b = None

        # capture previous metrics for deltas
        prev_metrics = st.session_state.get("last_metrics")

        t0 = time.time()
        matched_pairs, suspect_pairs, unmatched_a, unmatched_b = run_pipeline(
            df_a, df_b, mapping_a, mapping_b, rules_config, progress=cb, sample_n=sample_n,
            rule_order=rule_order, patterns_a=patterns_a, patterns_b=patterns_b
        )
        duration_s = max(0.0, time.time() - t0)

        st.session_state["matched_pairs"] = matched_pairs
        st.session_state["suspect_pairs"] = suspect_pairs
        st.session_state["unmatched_a"] = unmatched_a
        st.session_state["unmatched_b"] = unmatched_b

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
        # Persist metrics and previous for delta display
        cur = {
            "a": len(df_a),
            "b": len(df_b),
            "matched": len(matched_pairs),
            "suspects": len(suspect_pairs),
            "unmatched": len(unmatched_a) + len(unmatched_b),
        }
        st.session_state["last_metrics_prev"] = prev_metrics
        st.session_state["last_metrics"] = cur
        # record run for analytics
        st.session_state.setdefault("run_history", []).append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "matched": len(matched_pairs),
            "suspects": len(suspect_pairs),
            "unmatched": len(unmatched_a) + len(unmatched_b),
            "duration_s": round(duration_s, 3),
            "sample_n": sample_n or 0,
            "rule_order": rule_order,
            "amount_tol_pct": amount_tolerance,
            "date_tol_days": date_tolerance,
            "w_amount": w_amount,
            "w_date": w_date,
            "w_name": w_name,
            "w_domain": w_domain,
            "patterns_count": len(patterns_raw),
        })
        st.success(f"Linking complete! {len(matched_pairs)} matched, {len(suspect_pairs)} suspects.")

    if "matched_pairs" in st.session_state:
        matched_pairs = st.session_state["matched_pairs"]
        suspect_pairs = st.session_state["suspect_pairs"]
        unmatched_a = st.session_state["unmatched_a"]
        unmatched_b = st.session_state["unmatched_b"]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Records in A", len(df_a))
        col2.metric("Total Records in B", len(df_b))
        prev = st.session_state.get("last_metrics_prev")
        delta_matched = None
        delta_sus = None
        delta_unm = None
        if prev and all(k in prev for k in ("matched","suspects","unmatched")):
            delta_matched = len(matched_pairs) - prev.get("matched", 0)
            delta_sus = len(suspect_pairs) - prev.get("suspects", 0)
            delta_unm = (len(unmatched_a)+len(unmatched_b)) - prev.get("unmatched", 0)
        col3.metric("Matched", len(matched_pairs), delta=None if delta_matched is None else delta_matched)
        col4.metric("Suspect", len(suspect_pairs), delta=None if delta_sus is None else delta_sus)
        col5.metric("Unmatched", len(unmatched_a) + len(unmatched_b), delta=None if delta_unm is None else delta_unm)

        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Matched", "Suspects", "Unmatched", "Needs Attention", "Analytics"])

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
                    AgGrid(matched_df, gridOptions=gob.build(), height=320, theme=aggrid_theme)

        with tab2:
            st.subheader("Suspect Records")
            # quick view of current patterns (toggle)
            with st.expander("Current Patterns", expanded=False):
                for p in st.session_state.get("patterns", []):
                    st.write(p)
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
                    # Adopt Pattern helper
                    st.markdown("---")
                    st.caption("Adopt pattern: derive transform to align IDs next run")
                    from src.utils import suggest_pattern
                    patt = suggest_pattern(getattr(s, 'a_row', {}).get(st.session_state.get('a_invoice_id')), getattr(s, 'b_row', {}).get(st.session_state.get('b_ref')))
                    if patt:
                        st.json(patt.model_dump())
                        if st.button(f"Adopt this pattern for suspect #{i+1}"):
                            st.session_state.setdefault("patterns", []).append(patt.model_dump())
                            st.success("Pattern adopted for next run.")

        with tab3:
            st.subheader("Unmatched Records")
            col_ua, col_ub = st.columns(2)
            with col_ua:
                st.markdown("**Source A**")
                st.dataframe(pd.DataFrame(unmatched_a).head(50), use_container_width=True)
            with col_ub:
                st.markdown("**Source B**")
                st.dataframe(pd.DataFrame(unmatched_b).head(50), use_container_width=True)

        with tab4:
            st.subheader("Needs Attention")
            if st.session_state.get("needs_attention"):
                df_na = pd.DataFrame([{**x.a_row, **{f"B_{k}": v for k, v in x.b_row.items()}, "tier": x.tier} for x in st.session_state["needs_attention"] if hasattr(x, 'a_row')])
                if HAVE_AGGRID:
                    AgGrid(df_na, height=260, theme=aggrid_theme)
                else:
                    st.dataframe(df_na, height=260, use_container_width=True)

        with st.expander("Run Logs"):
            cols = st.columns([1,1,6])
            if cols[0].button("Clear Logs"):
                st.session_state["logs"] = []
            for log in st.session_state.get("logs", []):
                st.write(log)

        with st.expander("Tier Summary"):
            # compute counts by tier
            from collections import Counter
            match_tiers = Counter([getattr(m, 'tier', 'unknown') for m in matched_pairs])
            sus_tiers = Counter([getattr(s, 'tier', 'unknown') for s in suspect_pairs])
            st.write("Matched by tier:", dict(match_tiers))
            st.write("Suspects by tier:", dict(sus_tiers))
            # simple suspect reasons tally
            reasons = Counter([getattr(s, 'rationale', '') for s in suspect_pairs])
            st.write("Suspect reasons:", dict(reasons))

        # Exports: final matches, suspects with reasons, unmatched with rationale placeholder
        st.markdown("---")
        st.subheader("Export")
        # Final matches include accepted suspects
        final_matches = list(matched_pairs) + st.session_state.get("accepted", [])
        if final_matches:
            final_df = pd.DataFrame([
                {**m.a_row, **{f"B_{k}": v for k, v in m.b_row.items()}, "rationale": m.rationale, "tier": m.tier, "score": m.score} for m in final_matches if hasattr(m, 'a_row')
            ])
            st.download_button("Download Final Matches CSV", data=final_df.to_csv(index=False), file_name="final_matches.csv", mime="text/csv")
        if suspect_pairs:
            sus_df = pd.DataFrame([
                {**s.a_row, **{f"B_{k}": v for k, v in s.b_row.items()}, "rationale": s.rationale, "tier": s.tier, "score": s.score} for s in suspect_pairs if hasattr(s, 'a_row')
            ])
            st.download_button("Download Suspects CSV", data=sus_df.to_csv(index=False), file_name="suspects.csv", mime="text/csv")
        if unmatched_a is not None or unmatched_b is not None:
            ua = pd.DataFrame(unmatched_a)
            ub = pd.DataFrame(unmatched_b)
            if not ua.empty:
                # don't overwrite if pipeline provided granular rationale
                if "no_match_rationale" not in ua.columns:
                    ua["no_match_rationale"] = "No match within tolerances"
                st.download_button("Download Unmatched A CSV", data=ua.to_csv(index=False), file_name="unmatched_a.csv", mime="text/csv")
            if not ub.empty:
                if "no_match_rationale" not in ub.columns:
                    ub["no_match_rationale"] = "No match within tolerances"
                st.download_button("Download Unmatched B CSV", data=ub.to_csv(index=False), file_name="unmatched_b.csv", mime="text/csv")

        with tab5:
            st.subheader("Analytics Dashboard")
            runs = st.session_state.get("run_history", [])
            if runs:
                hist_df = pd.DataFrame(runs)
                st.caption("Runs over time")
                st.line_chart(hist_df.set_index("timestamp")[["matched","suspects","unmatched"]])
                st.caption("Run durations (s)")
                st.bar_chart(hist_df.set_index("timestamp")["duration_s"])
                st.download_button("Download Run History CSV", data=hist_df.to_csv(index=False), file_name="run_history.csv", mime="text/csv")
            else:
                st.info("No run history yet. Run the pipeline to populate analytics.")

            # Current-run insights
            from collections import Counter
            st.markdown("### Current run insights")
            mt = Counter([getattr(m, 'tier', 'unknown') for m in matched_pairs])
            st.write("Matches by tier:", dict(mt))

            # Composite diffs distribution (current run)
            comp = [m for m in matched_pairs if getattr(m, 'tier', '') == 'composite']
            if comp:
                def _safe_get(d, k):
                    return (d or {}).get(k)
                diffs = []
                for m in comp:
                    arow = getattr(m,'a_row',{})
                    brow = getattr(m,'b_row',{})
                    try:
                        # prefer normalized fields
                        da = pd.to_datetime(_safe_get(arow,'date'))
                        db = pd.to_datetime(_safe_get(brow,'date'))
                        dd = abs((da - db).days) if not pd.isna(da) and not pd.isna(db) else None
                        aa = pd.to_numeric(_safe_get(arow,'amount'))
                        ab = pd.to_numeric(_safe_get(brow,'amount'))
                        ap = float(abs((aa - ab)) / aa) if pd.notna(aa) and aa != 0 and pd.notna(ab) else None
                    except Exception:
                        dd = None
                        ap = None
                    diffs.append({"date_diff_days": dd, "amount_diff_pct": ap})
                diffs_df = pd.DataFrame(diffs).dropna()
                if not diffs_df.empty:
                    st.caption("Composite matches: date/amount differences")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.bar_chart(diffs_df["date_diff_days"].value_counts().sort_index())
                    with c2:
                        # bin percentages into 1% buckets
                        binned = (diffs_df["amount_diff_pct"]*100).round().clip(lower=0, upper=100).value_counts().sort_index()
                        st.bar_chart(binned)

            # Top email domains in matched vs suspects
            def _domain_from_row(rowdict):
                e = (rowdict or {}).get('clean_email') or (rowdict or {}).get('email')
                try:
                    return e.split('@',1)[1] if isinstance(e,str) and '@' in e else None
                except Exception:
                    return None
            md = [_domain_from_row(getattr(m,'a_row',{})) for m in matched_pairs]
            sd = [_domain_from_row(getattr(s,'a_row',{})) for s in suspect_pairs]
            md_ser = pd.Series([x for x in md if x]).value_counts().head(10)
            sd_ser = pd.Series([x for x in sd if x]).value_counts().head(10)
            c3, c4 = st.columns(2)
            with c3:
                st.caption("Top domains (matched)")
                if not md_ser.empty:
                    st.bar_chart(md_ser)
                else:
                    st.write("—")
            with c4:
                st.caption("Top domains (suspects)")
                if not sd_ser.empty:
                    st.bar_chart(sd_ser)
                else:
                    st.write("—")

    # moved tabs/analytics/export inside matched_pairs block to avoid undefined vars and fix indentation
else:
    st.warning("Please upload both Source A and Source B CSV files to continue.")
