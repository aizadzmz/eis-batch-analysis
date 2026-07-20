import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must happen before core.plotting imports pyplot

import streamlit as st

from core import EISParseError, parse_eis_file
from core.filtering import clear_mask, mask_inductive_points
from core.mb_parser import parse_modulobat_file
from core.plotting import plot_overlay, plot_residuals, plot_single
from core.validation import mask_residual_outliers, run_kk_test, run_zhit

st.set_page_config(page_title="EIS Batch Analysis", layout="wide")
st.title("EIS Batch Analysis — Parser & Plotter")

def _parse_file(path: str):
    """Try the Modulo Bat cycling-sequence parser first, then fall back to
    the generic EIS parser. Returns (datasets, parser_name)."""
    try:
        return parse_modulobat_file(path), "Modulo Bat (cycling sequence)"
    except Exception as mb_exc:
        try:
            return parse_eis_file(path), "Standard EIS export"
        except Exception as std_exc:
            raise EISParseError(
                f"Could not parse '{Path(path).name}'.\n"
                f"- Modulo Bat parser: {mb_exc}\n"
                f"- Standard EIS parser: {std_exc}"
            ) from std_exc


with st.sidebar:
    st.header("1. Load file")
    uploaded = st.file_uploader("EIS export", type=["mpt", "txt"])

if uploaded is None:
    st.info("Upload a .mpt or .txt file to begin.")
    st.stop()

cache_key = (uploaded.name, uploaded.size)
if st.session_state.get("_cache_key") != cache_key:
    suffix = Path(uploaded.name).suffix or ".mpt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        datasets, parser_used = _parse_file(tmp_path)
    except EISParseError as exc:
        st.error(str(exc))
        st.stop()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    st.session_state["_cache_key"] = cache_key
    st.session_state["_datasets"] = datasets
    st.session_state["_parser_used"] = parser_used
    st.session_state["_validation_results"] = {}

datasets = st.session_state["_datasets"]
parser_used = st.session_state["_parser_used"]
st.success(f"Parsed {len(datasets)} EIS sweep(s) from '{uploaded.name}' using the {parser_used} parser.")

with st.sidebar:
    st.header("2. Plot selection")
    labels = [ds.label for ds in datasets]
    mode = st.radio("Mode", ["Single", "Overlay"])
    if mode == "Single":
        idx = st.selectbox("Sweep", options=range(len(datasets)), format_func=lambda i: labels[i])
        selected = [datasets[idx]]
    else:
        chosen = st.multiselect("Sweeps", options=labels, default=labels[: min(5, len(labels))])
        selected = [ds for ds in datasets if ds.label in chosen]

    st.header("3. Style")
    style = st.radio(
        "Marker style",
        options=["scatter", "line"],
        format_func=lambda s: "Markers" if s == "scatter" else "Line",
    )

    st.header("4. Filtering")
    filter_inductive = st.checkbox("Remove inductive tail (Im(Z) > 0)", value=False)

    st.header("5. Validation")
    validation_method = st.radio(
        "Method",
        options=["Kramers-Kronig", "Z-HIT"],
        help="Kramers-Kronig checks linearity/causality via a lin-KK fit. "
             "Z-HIT reconstructs the modulus from the phase data and is "
             "good at catching non-steady-state artifacts such as "
             "low-frequency drift.",
    )
    validation_threshold = st.number_input(
        "Outlier threshold (%)",
        min_value=0.0,
        value=2.0,
        step=0.5,
        help="Points whose relative residual (real or imaginary) exceeds "
             "this percentage are removed.",
    )
    run_validation = st.button(f"Run {validation_method} validation")
    show_removed = st.checkbox("Show removed points on plot", value=True)

if not selected:
    st.warning("Select at least one sweep to plot.")
    st.stop()

for ds in selected:
    if filter_inductive:
        mask_inductive_points(ds)
    else:
        clear_mask(ds)

validation_results = st.session_state.setdefault("_validation_results", {})

if run_validation:
    runner = run_kk_test if validation_method == "Kramers-Kronig" else run_zhit
    with st.spinner(f"Running {validation_method} analysis..."):
        for ds in selected:
            validation_results[(validation_method, ds.label)] = runner(ds)

stale_labels = []
for ds in selected:
    result = validation_results.get((validation_method, ds.label))
    if result is not None:
        try:
            mask_residual_outliers(ds, result, validation_threshold)
        except ValueError:
            stale_labels.append(ds.label)

if stale_labels:
    st.warning(
        f"{validation_method} results for {', '.join(stale_labels)} no longer match "
        f"the current mask (e.g. the inductive-tail filter changed) — click "
        f"'Run {validation_method} validation' again."
    )

if mode == "Single":
    fig, _ = plot_single(selected[0], show=False, style=style, show_removed=show_removed)
else:
    fig, _ = plot_overlay(selected, show=False, style=style, show_removed=show_removed)

st.pyplot(fig)

residual_sweeps = [
    (ds, validation_results[(validation_method, ds.label)])
    for ds in selected
    if (validation_method, ds.label) in validation_results
]
if residual_sweeps:
    with st.expander(f"{validation_method} relative residuals", expanded=True):
        for ds, result in residual_sweeps:
            fig_r, _ = plot_residuals(
                result,
                title=f"{validation_method} residuals — {ds.label}",
                threshold=validation_threshold,
                show=False,
            )
            st.pyplot(fig_r)

with st.expander("Sweep details"):
    for ds in selected:
        validated_with = [
            method
            for method in ("Kramers-Kronig", "Z-HIT")
            if (method, ds.label) in validation_results
        ]
        note = f" (validated: {', '.join(validated_with)})" if validated_with else ""
        st.write(f"**{ds.label}** — {ds.num_points} points{note}")
