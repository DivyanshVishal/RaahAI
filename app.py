#!/usr/bin/env python3
"""
Train delay predictor - Streamlit app
-------------------------------------
Enter every model input by hand and get the predicted arrival delay at a
destination station, with a likely range, from the trained models.

The live model needs ~40 inputs and the combined (BEST) prediction also uses the
pre-departure model's ~50 inputs for the destination, so the sidebar can fill
every box from a real journey in the data. Edit any value afterwards to explore
"what if" scenarios (e.g. raise the current delay and watch the forecast move).

Run (from the project folder, next to predict.py and train_multistop.py):
    pip install streamlit
    streamlit run app.py

Optional: streamlit run app.py -- --features processed/features/year=2026 --models models/multistop
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_models, raw_q, train_index          # noqa: E402
from train_multistop import make_pairs, prepare, apply_blend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="processed/features/year=2026")
ap.add_argument("--models", default="models/multistop")
ARGS, _ = ap.parse_known_args()

# Plain-English descriptions of the live (pair) features built by train_multistop.make_pairs
LIVE_HELP = {
    "known_delay_A": "Current delay at station A (min, + = late). The single most important input.",
    "known_delay_A_src": "Where the current delay comes from: dep = departure recorded, arr = only arrival recorded",
    "dep_delay_A": "Departure delay at A (min); leave empty if the train has not left A yet",
    "arr_delay_A": "Arrival delay at A (min); empty at the journey's first station",
    "dwell_overrun_A": "Actual minus scheduled stop time at A (min)",
    "gain_prev1": "Delay gained on the section that ended at A (min)",
    "gain_prev2": "Delay gained on the section before that (min)",
    "trend_2stops": "Departure delay at A minus departure delay two stops earlier (min)",
    "origin_dep_delay": "Departure delay at the journey's first station (min)",
    "max_dep_delay_so_far": "Largest departure delay so far on this journey (min)",
    "arr_imputed_A": "1 if the arrival time at A was back-calculated, not observed",
    "train_type": "Train category",
    "dep_hour": "Scheduled departure time from A, as decimal hour (13.5 = 13:30)",
    "dow": "Day of week (0 = Monday ... 6 = Sunday)",
    "seg_idx": "Section number of A within the journey (0 = first station)",
    "n_segments": "Number of sections in the whole journey",
    "frac_A": "How far along the journey A is (0 = start, 1 = end)",
    "from_lat": "Latitude of A", "from_lng": "Longitude of A",
    "stn_A_dep_delay_mean_30d": "All trains at A: average departure delay over the last 30 days (min)",
    "stn_A_daily_deps": "Trains scheduled to depart A that day",
    "stops_ahead": "How many stops ahead the destination is",
    "path_sch_min": "Scheduled time from leaving A to arriving at the destination (min)",
    "path_km": "Distance from A to the destination (km)",
    "path_dwell_sch_min": "Scheduled stopping time at stations in between (min)",
    "path_stations_passed": "Stations passed without stopping, in between",
    "path_hist_median_sum": "This train's usual delay gain, summed over the sections to the destination (min)",
    "path_hist_mean90_sum": "Same, using 90-day averages (min)",
    "path_hist_p90_sum": "Same, using bad-day (90th percentile) values (min)",
    "path_sec_gain_sum": "All trains' average gain, summed over the sections (min)",
    "path_hist_missing": "Sections on the way with no history for this train",
    "path_same_dir_sum": "Other trains on the same sections at the same time, summed",
    "path_opposing_sum": "Opposing trains on the sections, summed",
    "to_lat": "Latitude of the destination", "to_lng": "Longitude of the destination",
    "frac_B": "How far along the journey the destination is (0-1)",
    "km_to_end_B": "Distance from the destination to the journey's end (km)",
    "is_last_B": "1 if the destination is the journey's last station",
    "stn_B_arr_delay_mean_30d": "All trains at the destination: average arrival delay, last 30 days (min)",
    "hist_arr_delay_mean_30d": "This train's average arrival delay at the destination, last 30 days (min)",
    "month": "Month (1-12)", "day_of_year": "Day of year (1-366)",
}
DERIVED = {"hour_sin", "hour_cos"}          # computed from dep_hour, not typed in
BINARY_HINT = ("is_", "arr_imputed")


@st.cache_resource
def get_models(path):
    meta, models = load_models(path)
    return meta, models


@st.cache_data
def get_dictionary(feat_dir):
    p = os.path.join(feat_dir, "_data_dictionary.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        return dict(zip(d["column"], d["description"]))
    return {}


@st.cache_data
def get_index(feat_dir):
    return train_index(feat_dir)


@st.cache_data
def load_train(feat_dir, file):
    return pd.read_parquet(os.path.join(feat_dir, file))


def to_val(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def fill_from_journey(J, i, target_row, live_feats, pre_feats):
    """Put a real journey's values into the input boxes. Returns the actual delay at the destination."""
    n = len(J)
    pairs = make_pairs(J, [i], [target_row], [0], [n])
    for c in live_feats:
        v = pairs[c].iloc[0]
        st.session_state["L_" + c] = str(v) if c in ("train_type", "known_delay_A_src") else to_val(v)
    row = J.iloc[target_row]
    for c in pre_feats:
        v = row[c]
        st.session_state["P_" + c] = str(v) if c == "train_type" else to_val(v)
    return to_val(J["y_arr_delay_B"].iloc[target_row]), str(J["from_stn"].iloc[i]), str(J["to_stn"].iloc[target_row])


def input_box(prefix, c, desc, cats):
    key = prefix + c
    if c in ("train_type", "known_delay_A_src"):
        opts = [o for o in cats.get(c, []) if o != "<NA>"] or ["<NA>"]
        cur = st.session_state.get(key)
        idx = opts.index(cur) if cur in opts else 0
        return st.selectbox(c, opts, index=idx, key=key + "_sel", help=desc)
    if key not in st.session_state:
        st.session_state[key] = None
    return st.number_input(c, value=st.session_state[key], key=key + "_num", help=desc,
                           placeholder="empty = unknown", format="%.3f")


def sync(prefix, feats):
    """Copy widget values back into the stored values (so refills and edits both work)."""
    for c in feats:
        for suf in ("_num", "_sel"):
            k = prefix + c + suf
            if k in st.session_state:
                st.session_state[prefix + c] = st.session_state[k]


def row_frame(prefix, feats):
    d = {}
    for c in feats:
        v = st.session_state.get(prefix + c)
        d[c] = [v if v is not None else (np.nan if c not in ("train_type", "known_delay_A_src") else "<NA>")]
    df = pd.DataFrame(d)
    if "dep_hour" in df and pd.notna(df["dep_hour"].iloc[0]):
        h = float(df["dep_hour"].iloc[0])
        if "hour_sin" in df:
            df["hour_sin"] = math.sin(2 * math.pi * h / 24)
        if "hour_cos" in df:
            df["hour_cos"] = math.cos(2 * math.pi * h / 24)
    return df


# ======================================================================= UI
st.set_page_config(page_title="Train Delay Predictor", page_icon="🚆", layout="wide")
st.title("🚆 Train delay predictor")
st.caption("Enter the situation of a train and a destination station; the trained models predict the arrival "
           "delay there, with a likely range (about 80% of real outcomes fall inside it).")

try:
    meta, models = get_models(ARGS.models)
except SystemExit as e:
    st.error(f"Could not load the models from {ARGS.models}: {e}")
    st.stop()
live_feats, pre_feats, cats = meta["live"], meta["pre_departure"], meta["categories"]
descr = get_dictionary(ARGS.features)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Fill from a real journey")
    st.caption("Loads every input from the data. You can then edit any value.")
    idx_df = None
    if os.path.isdir(ARGS.features):
        try:
            idx_df = get_index(ARGS.features)
        except Exception as e:
            st.warning(f"Could not read the features folder: {e}")
    if idx_df is not None and len(idx_df):
        train_no = st.selectbox("Train number", sorted(idx_df["train_no"].unique()))
        df = load_train(ARGS.features, idx_df.loc[idx_df["train_no"] == train_no, "file"].iloc[0])
        df = df[df["train_no"].astype(str) == str(train_no)]
        dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
        day = st.selectbox("Journey date", dates, index=len(dates) - 1)
        J = df[df["date"] == pd.Timestamp(day)].sort_values("seg_idx").reset_index(drop=True)
        if len(J) >= 1:
            starts = [f"{k}: {s}" for k, s in enumerate(J["from_stn"].astype(str))]
            a = st.selectbox("Train is now at (station A)", starts)
            i = int(a.split(":")[0])
            targets = [f"{r}: {J['to_stn'].iloc[r]}" for r in range(i, len(J))]
            b = st.selectbox("Destination station (B)", targets)
            tr = int(b.split(":")[0])
            if st.button("Load these values", type="primary", use_container_width=True):
                actual, sa, sb = fill_from_journey(J, i, tr, live_feats, pre_feats)
                for k in list(st.session_state.keys()):
                    if k.endswith("_num") or k.endswith("_sel"):
                        del st.session_state[k]
                st.session_state["actual"] = actual
                st.session_state["route"] = f"Train {train_no}, {day}: {sa} → {sb}"
                st.rerun()
    else:
        st.info(f"Features folder not found ({ARGS.features}); fill the inputs by hand.")
    st.divider()
    st.caption(f"Models: {ARGS.models}")

if st.session_state.get("route"):
    st.info(f"Loaded: **{st.session_state['route']}**. Edit any value below, then press Predict.")

# ---------------------------------------------------------------- inputs
groups = {
    "Where the train is now (station A) - live information": [c for c in live_feats if c in LIVE_HELP and not
                                                              c.startswith(("path_", "to_", "frac_B", "km_to_end_B",
                                                                            "is_last_B", "stn_B", "hist_arr", "stops"))],
    "Route from A to the destination": [c for c in live_feats if c.startswith("path_") or c == "stops_ahead"],
    "Destination station (B)": [c for c in live_feats if c.startswith(("to_", "frac_B", "km_to_end_B", "is_last_B",
                                                                     "stn_B", "hist_arr"))],
}
grouped = {c for v in groups.values() for c in v}
other = [c for c in live_feats if c not in grouped and c not in DERIVED]
if other:
    groups["Other live inputs"] = other

with st.form("inputs"):
    st.subheader("Live model inputs")
    for title, feats in groups.items():
        feats = [c for c in feats if c not in DERIVED]
        with st.expander(f"{title}  ({len(feats)})", expanded=title.startswith("Where")):
            cols = st.columns(3)
            for k, c in enumerate(feats):
                with cols[k % 3]:
                    input_box("L_", c, LIVE_HELP.get(c, descr.get(c, "")), cats)
    st.subheader("Pre-departure model inputs (for the section ending at the destination)")
    st.caption("Used for the combined BEST prediction and the pre-departure prediction. hour_sin / hour_cos are "
               "calculated from dep_hour automatically.")
    pf = [c for c in pre_feats if c not in DERIVED]
    with st.expander(f"Timetable, history and congestion inputs  ({len(pf)})"):
        cols = st.columns(3)
        for k, c in enumerate(pf):
            with cols[k % 3]:
                input_box("P_", c, descr.get(c, ""), cats)
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------------- predict
if submitted:
    sync("L_", live_feats)
    sync("P_", pre_feats)
    L = row_frame("L_", live_feats)
    P = row_frame("P_", pre_feats)
    k = L["known_delay_A"].iloc[0] if "known_delay_A" in L else np.nan
    if pd.isna(k):
        st.error("Please enter the current delay at A (known_delay_A).")
        st.stop()
    k = float(k)
    P_live = k + raw_q(models, "live", prepare(L, live_feats, cats))
    P_pre = raw_q(models, "pre", prepare(P, pre_feats, cats))
    sc = meta.get("range_scale", {})
    sl, sp = sc.get("live", 1.0), sc.get("pre_departure", 1.0)
    stops = float(L["stops_ahead"].iloc[0]) if pd.notna(L["stops_ahead"].iloc[0]) else 1.0
    if meta.get("blend"):
        b50, blo, bhi = apply_blend(P_live, P_pre, [stops], [k], meta["blend"]["weights"], meta["blend"]["scale"])
        b50, blo, bhi = float(b50[0]), float(blo[0]), float(bhi[0])
    else:
        b50 = float(P_live[0, 1])
        blo, bhi = b50 - sl * (b50 - P_live[0, 0]), b50 + sl * (P_live[0, 2] - b50)
    live50 = float(P_live[0, 1])
    pre50 = float(P_pre[0, 1])
    pre_lo, pre_hi = pre50 - sp * (pre50 - P_pre[0, 0]), pre50 + sp * (P_pre[0, 2] - pre50)

    st.subheader("Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted delay at destination", f"{b50:+.0f} min", help="BEST: live and pre-departure combined")
    c2.metric("Likely range", f"{blo:+.0f} to {bhi:+.0f} min")
    c3.metric("Change from now", f"{b50 - k:+.0f} min", help="Predicted delay minus current delay")
    actual = st.session_state.get("actual")
    if actual is not None and st.session_state.get("route"):
        c4.metric("Actual (from the data)", f"{actual:+.0f} min", delta=f"error {b50 - actual:+.1f} min",
                  delta_color="off")
    st.write(f"**Live model alone:** {live50:+.1f} min  ·  **Pre-departure model alone:** {pre50:+.1f} min "
             f"(range {pre_lo:+.0f} to {pre_hi:+.0f})  ·  current delay at A: {k:+.0f} min")
    st.caption("+ = late, − = early. If you edited loaded values, the 'actual' figure refers to the original "
               "journey, not to your edited scenario. Inputs depend on each other (e.g. path sums grow with "
               "stops ahead), so keep edits realistic.")
