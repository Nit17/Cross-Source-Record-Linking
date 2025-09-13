import streamlit as st
import pandas as pd
import re

# --- Data Cleaning & Normalization Functions ---
def normalize_id(text):
    """Extracts only digits from a string."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\D', '', text)

def clean_email(email):
    """Removes whitespace from an email string."""
    if not isinstance(email, str):
        return ""
    return email.replace(" ", "").lower()

def is_amount_close(amount1, amount2, tolerance=0.001): # 0.1% tolerance
    """Checks if two amounts are within a certain percentage tolerance."""
    if amount1 == 0 and amount2 == 0:
        return True
    if pd.isna(amount1) or pd.isna(amount2) or amount1 == 0:
        return False
    return abs(amount1 - amount2) / amount1 <= tolerance

def is_date_close(date1, date2, tolerance_days=2):
    """Checks if two dates are within a certain day tolerance."""
    if pd.isna(date1) or pd.isna(date2):
        return False
    return abs((date1 - date2).days) <= tolerance_days

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

# Column mapping UI (semantic fields)
if df_a is not None and df_b is not None:
    st.subheader("Column Mapping")
    col_left, col_right = st.columns(2)

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

    mapping = {
        'a': {
            'invoice_id': a_invoice_id,
            'customer_email': a_email,
            'invoice_date': a_date,
            'total_amount': a_amount,
            'po_number': a_po,
        },
        'b': {
            'ref_code': b_ref,
            'email': b_email,
            'doc_date': b_date,
            'grand_total': b_amount,
            'purchase_order': b_po,
        }
    }

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
    rules_config = {
        'amount_tolerance': amount_tolerance / 100,
        'date_tolerance': date_tolerance
    }

    # --- Main Matching Pipeline ---
    def run_linking_pipeline(df_a, df_b, mapping, rules_config):
        # Pre-processing: create normalized columns
        df_a = df_a.copy().reset_index().rename(columns={'index': '_a_idx'})
        df_b = df_b.copy().reset_index().rename(columns={'index': '_b_idx'})

        a_id = mapping['a']['invoice_id']
        a_email = mapping['a']['customer_email']
        a_date = mapping['a']['invoice_date']
        a_amt = mapping['a']['total_amount']
        b_id = mapping['b']['ref_code']
        b_email = mapping['b']['email']
        b_date = mapping['b']['doc_date']
        b_amt = mapping['b']['grand_total']

        df_a['canonical_id'] = df_a[a_id].astype(str).apply(normalize_id)
        df_a['clean_email'] = df_a[a_email].astype(str).apply(clean_email)
        df_b['canonical_id'] = df_b[b_id].astype(str).apply(normalize_id)
        df_b['clean_email'] = df_b[b_email].astype(str).apply(clean_email)
        df_a['invoice_date_norm'] = pd.to_datetime(df_a[a_date], errors='coerce')
        df_b['doc_date_norm'] = pd.to_datetime(df_b[b_date], errors='coerce')
        df_a['total_amount_norm'] = pd.to_numeric(df_a[a_amt], errors='coerce')
        df_b['grand_total_norm'] = pd.to_numeric(df_b[b_amt], errors='coerce')

        matched_pairs = []
        suspect_pairs = []
        used_a = set()
        used_b = set()

        def add_match(row_a, row_b, tier, rationale):
            matched_pairs.append({
                'source_a': row_a.to_dict(),
                'source_b': row_b.to_dict(),
                'tier': tier,
                'rationale': rationale
            })
            used_a.add(row_a['_a_idx'])
            used_b.add(row_b['_b_idx'])

        def add_suspect(row_a, row_b, tier, rationale):
            suspect_pairs.append({
                'source_a': row_a.to_dict(),
                'source_b': row_b.to_dict(),
                'tier': tier,
                'rationale': rationale
            })

        # --- Tier 1: Exact ID Match ---
        exact = df_a.merge(df_b, left_on=a_id, right_on=b_id, how='inner', suffixes=('_a', '_b'))
        if not exact.empty:
            # one-to-one resolution
            exact = exact.sort_values([a_id, b_id]).drop_duplicates(subset=['_a_idx'], keep='first')
            seen_b = set()
            for _, r in exact.iterrows():
                if r['_a_idx'] in used_a or r['_b_idx'] in used_b or r['_b_idx'] in seen_b:
                    continue
                add_match(r, r, 1, 'Exact match on primary identifier.')
                seen_b.add(r['_b_idx'])

        # --- Tier 2: Canonical ID Match ---
        rem_a = df_a[~df_a['_a_idx'].isin(used_a)]
        rem_b = df_b[~df_b['_b_idx'].isin(used_b)]
        canon = rem_a.merge(rem_b, on='canonical_id', how='inner', suffixes=('_a', '_b'))
        if not canon.empty:
            canon = canon.sort_values(['canonical_id']).drop_duplicates(subset=['_a_idx'], keep='first')
            for _, r in canon.iterrows():
                if r['_a_idx'] in used_a or r['_b_idx'] in used_b:
                    continue
                add_match(r, r, 2, 'Match on normalized numeric identifier.')

        # --- Tier 3: Composite Key Match with tie-breaker ---
        rem_a = df_a[~df_a['_a_idx'].isin(used_a)]
        rem_b = df_b[~df_b['_b_idx'].isin(used_b)]
        amount_tol = rules_config.get('amount_tolerance', 0.001)
        date_tol = rules_config.get('date_tolerance', 2)

        if not rem_a.empty and not rem_b.empty:
            cand = rem_a.merge(rem_b, left_on='clean_email', right_on='clean_email', how='inner', suffixes=('_a', '_b'))
            if not cand.empty:
                cand['amount_diff_pct'] = (cand['total_amount_norm'] - cand['grand_total_norm']).abs() / cand['total_amount_norm'].replace(0, pd.NA)
                cand['date_diff_days'] = (cand['invoice_date_norm'] - cand['doc_date_norm']).abs().dt.days
                eligible = cand[(cand['amount_diff_pct'] <= amount_tol) & (cand['date_diff_days'] <= date_tol)]

                # Greedy assignment by score
                if not eligible.empty:
                    eligible['score'] = eligible['amount_diff_pct'].fillna(1e9) * 100 + eligible['date_diff_days'].fillna(1e9)
                    eligible = eligible.sort_values(['score'])
                    taken_b = set()
                    taken_a = set()
                    for _, r in eligible.iterrows():
                        if r['_a_idx'] in used_a or r['_b_idx'] in used_b:
                            continue
                        if r['_a_idx'] in taken_a or r['_b_idx'] in taken_b:
                            # competing candidate becomes suspect
                            ra = rem_a.loc[rem_a['_a_idx'] == r['_a_idx']].iloc[0]
                            rb = rem_b.loc[rem_b['_b_idx'] == r['_b_idx']].iloc[0]
                            add_suspect(ra, rb, 3, 'Competing candidate (tie) in composite match.')
                            continue
                        ra = rem_a.loc[rem_a['_a_idx'] == r['_a_idx']].iloc[0]
                        rb = rem_b.loc[rem_b['_b_idx'] == r['_b_idx']].iloc[0]
                        add_match(ra, rb, 3, f"Match on email with date ≤ {date_tol} days and amount ≤ {amount_tol*100:.2f}%.")
                        taken_a.add(r['_a_idx'])
                        taken_b.add(r['_b_idx'])

                # Near-miss suspects
                near = cand[((cand['amount_diff_pct'] <= amount_tol) & (cand['date_diff_days'] > date_tol)) |
                            ((cand['amount_diff_pct'] > amount_tol) & (cand['date_diff_days'] <= date_tol))]
                if not near.empty:
                    for _, r in near.iterrows():
                        ra = rem_a.loc[rem_a['_a_idx'] == r['_a_idx']].iloc[0]
                        rb = rem_b.loc[rem_b['_b_idx'] == r['_b_idx']].iloc[0]
                        reason = 'Email matches; one of date/amount just outside tolerance.'
                        add_suspect(ra, rb, 3, reason)

        # Final unmatched
        unmatched_a = df_a[~df_a['_a_idx'].isin(used_a)]
        unmatched_b = df_b[~df_b['_b_idx'].isin(used_b)]

        return matched_pairs, suspect_pairs, unmatched_a, unmatched_b

    # --- Run Linking Button ---
    if mapping_ready and st.button("Run Linking"):
        matched_pairs, suspect_pairs, unmatched_a, unmatched_b = run_linking_pipeline(df_a, df_b, mapping, rules_config)

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
    if mapping_ready and "matched_pairs" in st.session_state:
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
        tab1, tab2, tab3 = st.tabs(["Matched", "Suspects", "Unmatched"])

        with tab1:
            st.subheader("Matched Records")
            for i, match in enumerate(matched_pairs):
                with st.expander(f"Match {i+1} - Tier {match['tier']}"):
                    st.write("Source A Record:", match['source_a'])
                    st.write("Source B Record:", match['source_b'])
                    st.info(f"Rationale: {match['rationale']}")
            if matched_pairs:
                matched_df = pd.DataFrame([
                    {**m['source_a'], **{f"B_{k}": v for k, v in m['source_b'].items()}, "rationale": m['rationale'], "tier": m['tier']} for m in matched_pairs
                ])
                csv = matched_df.to_csv(index=False)
                st.download_button("Download Matched Results CSV", data=csv, file_name="matched_results.csv", mime="text/csv")

        with tab2:
            st.subheader("Suspect Records")
            for i, suspect in enumerate(suspect_pairs):
                with st.expander(f"Suspect {i+1} - Tier {suspect['tier']}"):
                    st.write("Source A Record:", suspect['source_a'])
                    st.write("Source B Record:", suspect['source_b'])
                    st.info(f"Rationale: {suspect['rationale']}")
                    st.button(f"Confirm Match {i+1}", key=f"confirm_{i}")

        with tab3:
            st.subheader("Unmatched Records")
            st.write("Unmatched Source A Records:", unmatched_a)
            st.write("Unmatched Source B Records:", unmatched_b)

        # --- Print Logs ---
        with st.expander("Run Logs"):
            cols = st.columns([1,1,6])
            if cols[0].button("Clear Logs"):
                st.session_state["logs"] = []
            for log in st.session_state.get("logs", []):
                st.write(log)
else:
    st.warning("Please upload both Source A and Source B CSV files to continue.")
