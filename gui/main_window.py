"""Main window: sidebar controls on the left, plot tabs on the right.

Mirrors the Streamlit app's workflow (load -> select -> style -> filter ->
validate) but with explicit event handling instead of rerun-everything.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import EISParseError, parse_eis_file
from core.drt import (
    CROSS_VALIDATION_METHODS,
    DATA_MODES,
    RBF_SHAPE_CONTROLS,
    RBF_TYPES,
    analyze_drt_peaks,
    run_drt,
    run_drt_bht,
)
from core.filtering import clear_mask, mask_inductive_points
from core.mb_parser import parse_modulobat_file
from core.plotting import plot_drt, plot_overlay, plot_residuals, plot_single
from core.validation import mask_residual_outliers, run_kk_test, run_zhit
from gui.figure_panes import FigureListPane, FigurePane
from gui.theme import THEMES, apply_theme
from gui.workers import DRTWorker, ValidationWorker

SIDEBAR_WIDTH = 320
VALIDATION_METHODS = ("Kramers-Kronig", "Z-HIT")
ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


def _titleize(rbf_type: str) -> str:
    """'c2-matern' -> 'C2 Matern', 'piecewise-linear' -> 'Piecewise Linear'."""
    return " ".join(word.capitalize() for word in rbf_type.split("-"))


def _add_combo_items(combo: QComboBox, pairs) -> None:
    """Populate a QComboBox from (display_text, value) pairs, retrievable
    via combo.currentData()."""
    for display, value in pairs:
        combo.addItem(display, value)


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIS Batch Analysis — Parser & Plotter")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1200, 800)

        self._datasets: List = []
        self._parser_used = ""
        self._source_name = ""
        # {(method, dataset label): KramersKronigResult | ZHITResult}
        self._validation_results = {}
        self._worker: Optional[ValidationWorker] = None
        self._worker_errors: List[Tuple[str, str]] = []
        # {dataset label: TRRBFResult | BHTResult} — last DRT run wins
        self._drt_results = {}
        # {dataset label: DRTPeaks}
        self._drt_peaks = {}
        self._drt_worker: Optional[DRTWorker] = None
        self._drt_worker_errors: List[Tuple[str, str]] = []
        # Rebuilding every tab's figures on every checkbox click is O(tabs *
        # selected sweeps) and dominates when overlaying many curves, so tab
        # content is rebuilt lazily: _refresh() does the cheap masking/
        # validity bookkeeping and marks every tab dirty, then only the
        # currently visible tab is actually (re)plotted. Switching tabs
        # renders whichever tab was still dirty.
        self._pending: Optional[dict] = None
        self._tab_dirty: set = set()

        self._settings = QSettings()
        saved = self._settings.value("theme", "light")
        self._theme_mode = saved if saved in THEMES else "light"

        self._build_menu()
        self._build_ui()

        # Apply the theme once the widgets exist, then reflect the current
        # mode in the menu without re-triggering the toggle handler.
        apply_theme(self._theme_mode)
        self.dark_action.blockSignals(True)
        self.dark_action.setChecked(self._theme_mode == "dark")
        self.dark_action.blockSignals(False)

    # ------------------------------------------------------------------ UI

    def _build_menu(self) -> None:
        # One QAction backs the menu item, the status-bar button, and the
        # shortcut, so their checked states stay in sync automatically.
        self.dark_action = QAction("🌙", self, checkable=True)
        self.dark_action.setShortcut("Ctrl+D")
        self.dark_action.setStatusTip("Toggle between light and dark themes (Ctrl+D)")
        self.dark_action.toggled.connect(self._on_theme_toggled)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.dark_action)

        # A flat button in the status bar shares the same action, so it stays
        # in sync with the menu item and Ctrl+D. addPermanentWidget docks it
        # in the bottom-right corner; swap to addWidget for the bottom-left.
        theme_button = QToolButton()
        theme_button.setDefaultAction(self.dark_action)
        theme_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        theme_button.setAutoRaise(True)
        self.statusBar().addPermanentWidget(theme_button)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_main_area(), stretch=1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Open a .mpt or .txt file to begin.")

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. Load file
        load_box = QGroupBox("1. Load file")
        load_layout = QVBoxLayout(load_box)
        self.open_button = QPushButton("Open EIS export…")
        self.open_button.clicked.connect(self._open_file)
        self.file_label = QLabel("No file loaded.")
        self.file_label.setWordWrap(True)
        load_layout.addWidget(self.open_button)
        load_layout.addWidget(self.file_label)
        layout.addWidget(load_box)

        # 2. Plot selection
        select_box = QGroupBox("2. Plot selection")
        select_layout = QVBoxLayout(select_box)
        self.single_radio = QRadioButton("Single")
        self.overlay_radio = QRadioButton("Overlay")
        self.single_radio.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.single_radio)
        mode_group.addButton(self.overlay_radio)
        self.single_radio.toggled.connect(self._on_mode_changed)
        select_layout.addWidget(self.single_radio)
        select_layout.addWidget(self.overlay_radio)

        self.sweep_combo = QComboBox()
        self.sweep_combo.currentIndexChanged.connect(self._refresh)
        select_layout.addWidget(self.sweep_combo)

        self.sweep_list = QListWidget()
        self.sweep_list.itemChanged.connect(self._refresh)
        self.sweep_list.setVisible(False)
        self.sweep_list.setMaximumHeight(160)
        select_layout.addWidget(self.sweep_list)
        layout.addWidget(select_box)

        # 3. Style
        style_box = QGroupBox("3. Style")
        style_layout = QVBoxLayout(style_box)
        self.markers_radio = QRadioButton("Markers")
        self.line_radio = QRadioButton("Line")
        self.markers_radio.setChecked(True)
        style_group = QButtonGroup(self)
        style_group.addButton(self.markers_radio)
        style_group.addButton(self.line_radio)
        self.markers_radio.toggled.connect(self._refresh)
        style_layout.addWidget(self.markers_radio)
        style_layout.addWidget(self.line_radio)
        layout.addWidget(style_box)

        # 4. Filtering
        filter_box = QGroupBox("4. Filtering")
        filter_layout = QVBoxLayout(filter_box)
        self.inductive_check = QCheckBox("Remove inductive tail (Im(Z) > 0)")
        self.inductive_check.toggled.connect(self._refresh)
        filter_layout.addWidget(self.inductive_check)
        layout.addWidget(filter_box)

        # 5. Validation
        valid_box = QGroupBox("5. Validation")
        valid_layout = QVBoxLayout(valid_box)
        self.kk_radio = QRadioButton("Kramers-Kronig")
        self.zhit_radio = QRadioButton("Z-HIT")
        self.kk_radio.setChecked(True)
        method_group = QButtonGroup(self)
        method_group.addButton(self.kk_radio)
        method_group.addButton(self.zhit_radio)
        valid_box.setToolTip(
            "Kramers-Kronig checks linearity/causality via a lin-KK fit. "
            "Z-HIT reconstructs the modulus from the phase data and is good "
            "at catching non-steady-state artifacts such as low-frequency drift."
        )
        self.kk_radio.toggled.connect(self._on_method_changed)
        valid_layout.addWidget(self.kk_radio)
        valid_layout.addWidget(self.zhit_radio)

        valid_layout.addWidget(QLabel("Outlier threshold (%)"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(2.0)
        self.threshold_spin.setToolTip(
            "Points whose relative residual (real or imaginary) exceeds "
            "this percentage are removed."
        )
        self.threshold_spin.valueChanged.connect(self._refresh)
        valid_layout.addWidget(self.threshold_spin)

        self.run_validation_button = QPushButton()
        self.run_validation_button.clicked.connect(self._run_validation)
        valid_layout.addWidget(self.run_validation_button)

        self.show_removed_check = QCheckBox("Show removed points on plot")
        self.show_removed_check.setChecked(True)
        self.show_removed_check.toggled.connect(self._refresh)
        valid_layout.addWidget(self.show_removed_check)
        layout.addWidget(valid_box)

        # 6. DRT settings
        drt_settings_box = QGroupBox("6. DRT settings")
        drt_settings_layout = QVBoxLayout(drt_settings_box)
        drt_settings_box.setToolTip(
            "Settings for Tikhonov regularization + radial basis function "
            "discretization (TR-RBF) and the Bayesian Hilbert Transform "
            "(BHT), applied to each selected sweep's currently unmasked "
            "points."
        )

        drt_settings_layout.addWidget(QLabel("Method of Discretization"))
        self.drt_rbf_combo = QComboBox()
        _add_combo_items(self.drt_rbf_combo, [(_titleize(v), v) for v in RBF_TYPES])
        drt_settings_layout.addWidget(self.drt_rbf_combo)

        drt_settings_layout.addWidget(QLabel("Data Used"))
        self.drt_mode_combo = QComboBox()
        _add_combo_items(
            self.drt_mode_combo,
            [
                ("Combined Re-Im Data", "complex"),
                ("Re Data", "real"),
                ("Im Data", "imaginary"),
            ],
        )
        drt_settings_layout.addWidget(self.drt_mode_combo)

        self.drt_inductance_check = QCheckBox("Fit with inductance")
        self.drt_inductance_check.setToolTip(
            "Include an inductive element in the fit. To discard inductive "
            "points entirely, use the 'Remove inductive tail' filter above "
            "instead."
        )
        drt_settings_layout.addWidget(self.drt_inductance_check)

        drt_settings_layout.addWidget(QLabel("Regularization Derivative"))
        self.drt_derivative_combo = QComboBox()
        _add_combo_items(
            self.drt_derivative_combo,
            [("1st order", 1), ("2nd order", 2)],
        )
        self.drt_derivative_combo.setToolTip(
            "pyimpspec's TR-RBF/BHT only implement 1st- and 2nd-order "
            "Tikhonov regularization (0th order is not available)."
        )
        self.drt_derivative_combo.setCurrentIndex(0)
        drt_settings_layout.addWidget(self.drt_derivative_combo)

        drt_settings_layout.addWidget(QLabel("Parameter Selection Method"))
        self.drt_cv_combo = QComboBox()
        _add_combo_items(
            self.drt_cv_combo,
            [
                ("custom", ""),
                ("GCV", "gcv"),
                ("mGCV", "mgcv"),
                ("rGCV", "rgcv"),
                ("re-im", "re-im"),
                ("L-curve", "lc"),
            ],
        )
        drt_settings_layout.addWidget(self.drt_cv_combo)

        drt_settings_layout.addWidget(QLabel("Regularization parameter"))
        self.drt_lambda_spin = QDoubleSpinBox()
        self.drt_lambda_spin.setDecimals(6)
        self.drt_lambda_spin.setMinimum(1e-10)
        self.drt_lambda_spin.setMaximum(10.0)
        self.drt_lambda_spin.setSingleStep(0.001)
        self.drt_lambda_spin.setValue(0.001)
        self.drt_lambda_spin.setToolTip(
            "Used directly when Parameter Selection Method is 'custom'; "
            "otherwise used as the initial value for the chosen "
            "cross-validation method."
        )
        drt_settings_layout.addWidget(self.drt_lambda_spin)

        drt_settings_layout.addWidget(QLabel("Optimal Regularization parameter"))
        self.drt_optimal_lambda_label = QLabel("—")
        self.drt_optimal_lambda_label.setToolTip(
            "The regularization parameter actually used by the most recent "
            "run, shown when exactly one sweep is selected."
        )
        drt_settings_layout.addWidget(self.drt_optimal_lambda_label)

        drt_settings_layout.addWidget(QLabel("RBF Shape Control"))
        self.drt_shape_control_combo = QComboBox()
        _add_combo_items(
            self.drt_shape_control_combo,
            [("FWHM Coefficient", "fwhm"), ("Shape Factor", "factor")],
        )
        drt_settings_layout.addWidget(self.drt_shape_control_combo)

        drt_settings_layout.addWidget(QLabel("FWHM / Shape Factor Control"))
        self.drt_shape_coeff_spin = QDoubleSpinBox()
        self.drt_shape_coeff_spin.setDecimals(4)
        self.drt_shape_coeff_spin.setMinimum(0.0001)
        self.drt_shape_coeff_spin.setMaximum(10.0)
        self.drt_shape_coeff_spin.setSingleStep(0.05)
        self.drt_shape_coeff_spin.setValue(0.5)
        drt_settings_layout.addWidget(self.drt_shape_coeff_spin)

        drt_settings_layout.addWidget(QLabel("Number of Samples"))
        self.drt_num_samples_spin = QSpinBox()
        self.drt_num_samples_spin.setMinimum(1000)
        self.drt_num_samples_spin.setMaximum(100000)
        self.drt_num_samples_spin.setSingleStep(500)
        self.drt_num_samples_spin.setValue(1000)
        self.drt_num_samples_spin.setToolTip(
            "Only used by Bayesian Run and Hilbert Transform. Must be >= "
            "1000; larger values are more accurate but slower."
        )
        drt_settings_layout.addWidget(self.drt_num_samples_spin)

        drt_settings_layout.addWidget(QLabel("Bayesian Run timeout (s)"))
        self.drt_timeout_spin = QSpinBox()
        self.drt_timeout_spin.setMinimum(0)
        self.drt_timeout_spin.setMaximum(36000)
        self.drt_timeout_spin.setSingleStep(60)
        self.drt_timeout_spin.setValue(300)
        self.drt_timeout_spin.setToolTip(
            "Bayesian Run's credible-interval sampler can be extremely slow "
            "(tens of minutes for even modest sweeps). It aborts once this "
            "many seconds pass; 0 disables the limit entirely."
        )
        drt_settings_layout.addWidget(self.drt_timeout_spin)

        layout.addWidget(drt_settings_box)

        # 7. Run DRT
        run_drt_box = QGroupBox("7. Run DRT")
        run_drt_layout = QVBoxLayout(run_drt_box)

        self.run_drt_simple_button = QPushButton("Simple Run")
        self.run_drt_simple_button.setToolTip(
            "Fast, deterministic TR-RBF point estimate (no credible intervals)."
        )
        self.run_drt_simple_button.clicked.connect(self._run_drt_simple)
        run_drt_layout.addWidget(self.run_drt_simple_button)

        self.run_drt_bayesian_button = QPushButton("Bayesian Run")
        self.run_drt_bayesian_button.setToolTip(
            "TR-RBF with Bayesian credible intervals via HMC sampling. Can "
            "be very slow — runs in the background so the UI stays "
            "responsive; see the timeout setting above."
        )
        self.run_drt_bayesian_button.clicked.connect(self._run_drt_bayesian)
        run_drt_layout.addWidget(self.run_drt_bayesian_button)

        self.run_drt_bht_button = QPushButton("Hilbert Transform")
        self.run_drt_bht_button.setToolTip(
            "Bayesian Hilbert Transform (BHT): estimates the DRT separately "
            "from the real and imaginary parts, and scores how well they "
            "agree (a Kramers-Kronig-style consistency check)."
        )
        self.run_drt_bht_button.clicked.connect(self._run_drt_bht)
        run_drt_layout.addWidget(self.run_drt_bht_button)

        layout.addWidget(run_drt_box)

        # 8. DRT peak analysis
        peak_box = QGroupBox("8. DRT peak analysis")
        peak_layout = QVBoxLayout(peak_box)
        peak_box.setToolTip(
            "Fits individual (skew) normal peaks to the most recently "
            "computed DRT result for each selected sweep."
        )

        peak_layout.addWidget(QLabel("Number of peaks (0 = all detected)"))
        self.drt_num_peaks_spin = QSpinBox()
        self.drt_num_peaks_spin.setMinimum(0)
        self.drt_num_peaks_spin.setMaximum(50)
        self.drt_num_peaks_spin.setValue(0)
        peak_layout.addWidget(self.drt_num_peaks_spin)

        self.run_peak_analysis_button = QPushButton("Peak deconvolution")
        self.run_peak_analysis_button.clicked.connect(self._run_peak_analysis)
        peak_layout.addWidget(self.run_peak_analysis_button)

        layout.addWidget(peak_box)

        layout.addStretch()
        self._update_validation_button_text()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sidebar = scroll
        return scroll

    def _build_main_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "color: #7a5900; background: #fff3cd; border: 1px solid #ffe08a;"
            "border-radius: 4px; padding: 6px;"
        )
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        self.stack = QStackedWidget()

        placeholder = QLabel("Open a .mpt or .txt file to begin.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(placeholder)

        self.tabs = QTabWidget()
        self.nyquist_pane = FigurePane(with_toolbar=True)
        self.residuals_pane = FigureListPane()
        self.drt_pane = FigurePane(with_toolbar=True)
        self.drt_peaks_text = QPlainTextEdit()
        self.drt_peaks_text.setReadOnly(True)
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.tabs.addTab(self.nyquist_pane, "Nyquist")
        self.tabs.addTab(self.residuals_pane, "Residuals")
        self.tabs.addTab(self.drt_pane, "DRT")
        self.tabs.addTab(self.drt_peaks_text, "DRT Peaks")
        self.tabs.addTab(self.details_text, "Sweep details")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.stack.addWidget(self.tabs)

        layout.addWidget(self.stack, stretch=1)
        return container

    # ------------------------------------------------------------- helpers

    @property
    def _mode(self) -> str:
        return "Single" if self.single_radio.isChecked() else "Overlay"

    @property
    def _style(self) -> str:
        return "scatter" if self.markers_radio.isChecked() else "line"

    @property
    def _validation_method(self) -> str:
        return VALIDATION_METHODS[0] if self.kk_radio.isChecked() else VALIDATION_METHODS[1]

    def _selected_datasets(self) -> List:
        if not self._datasets:
            return []
        if self._mode == "Single":
            idx = self.sweep_combo.currentIndex()
            return [self._datasets[idx]] if 0 <= idx < len(self._datasets) else []
        return [
            ds
            for i, ds in enumerate(self._datasets)
            if self.sweep_list.item(i).checkState() == Qt.Checked
        ]

    def _update_validation_button_text(self) -> None:
        # Keep it short — the selected method is shown by the radios above.
        if self._worker is not None:
            self.run_validation_button.setText("Running…")
        else:
            self.run_validation_button.setText("Run validation")

    # ------------------------------------------------------------ handlers

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EIS export", "", "EIS exports (*.mpt *.txt)"
        )
        if not path:
            return

        try:
            datasets, parser_used = _parse_file(path)
        except EISParseError as exc:
            QMessageBox.critical(self, "Parse error", str(exc))
            return

        self._datasets = datasets
        self._parser_used = parser_used
        self._source_name = Path(path).name
        self._validation_results = {}
        self._drt_results = {}
        self._drt_peaks = {}

        # Elide instead of wrapping: filenames are one unbreakable token, and
        # a word-wrapped QLabel's minimum width would force the sidebar to
        # scroll sideways. Full name stays available as a tooltip.
        metrics = self.file_label.fontMetrics()
        self.file_label.setText(
            metrics.elidedText(self._source_name, Qt.ElideMiddle, SIDEBAR_WIDTH - 60)
        )
        self.file_label.setToolTip(self._source_name)
        self.statusBar().showMessage(
            f"Parsed {len(datasets)} EIS sweep(s) from '{self._source_name}' "
            f"using the {parser_used} parser."
        )

        # Repopulate the sweep selectors without triggering refreshes.
        self.sweep_combo.blockSignals(True)
        self.sweep_combo.clear()
        self.sweep_combo.addItems([ds.label for ds in datasets])
        self.sweep_combo.setCurrentIndex(0)
        self.sweep_combo.blockSignals(False)

        self.sweep_list.blockSignals(True)
        self.sweep_list.clear()
        for i, ds in enumerate(datasets):
            item = QListWidgetItem(ds.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < 5 else Qt.Unchecked)
            self.sweep_list.addItem(item)
        self.sweep_list.blockSignals(False)

        self.stack.setCurrentWidget(self.tabs)
        self._refresh()

    def _on_mode_changed(self) -> None:
        single = self._mode == "Single"
        self.sweep_combo.setVisible(single)
        self.sweep_list.setVisible(not single)
        self._refresh()

    def _on_method_changed(self, checked: bool) -> None:
        if not checked:
            return  # fires once per radio; only act on the newly-checked one
        self._update_validation_button_text()
        self._refresh()

    def _on_theme_toggled(self, checked: bool) -> None:
        self._theme_mode = "dark" if checked else "light"
        apply_theme(self._theme_mode)
        self._settings.setValue("theme", self._theme_mode)
        # Existing figures were drawn with the old rcParams; regenerate them
        # so plot colors follow the new theme.
        if self._datasets:
            self._refresh()

    def _run_validation(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return
        method = self._validation_method
        runner = run_kk_test if method == VALIDATION_METHODS[0] else run_zhit

        # Masks must be stable while the worker reads them, so lock the
        # sidebar for the duration of the run.
        self._worker_errors = []
        self._worker = ValidationWorker(method, runner, selected, parent=self)
        self._worker.result_ready.connect(self._on_validation_result)
        self._worker.error.connect(self._on_validation_error)
        self._worker.finished.connect(self._on_validation_finished)
        self._sidebar.setEnabled(False)
        self._update_validation_button_text()
        self.statusBar().showMessage(f"Running {method} analysis…")
        self._worker.start()

    def _on_validation_result(self, method: str, label: str, result) -> None:
        self._validation_results[(method, label)] = result

    def _on_validation_error(self, label: str, message: str) -> None:
        self._worker_errors.append((label, message))

    def _on_validation_finished(self) -> None:
        self._worker = None
        self._sidebar.setEnabled(True)
        self._update_validation_button_text()
        self.statusBar().showMessage("Validation finished.")
        if self._worker_errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in self._worker_errors)
            QMessageBox.warning(
                self, "Validation errors", f"Some sweeps failed:\n{details}"
            )
        self._refresh()

    def _drt_settings(self) -> dict:
        """Shared TR-RBF/BHT settings read from the '6. DRT settings' panel."""
        return dict(
            rbf_type=self.drt_rbf_combo.currentData(),
            derivative_order=self.drt_derivative_combo.currentData(),
            rbf_shape=self.drt_shape_control_combo.currentData(),
            shape_coeff=self.drt_shape_coeff_spin.value(),
        )

    def _update_optimal_lambda_label(self, selected: List) -> None:
        lambda_value = None
        if len(selected) == 1:
            result = self._drt_results.get(selected[0].label)
            lambda_value = getattr(result, "lambda_value", None)
        self.drt_optimal_lambda_label.setText(
            f"{lambda_value:.4g}" if lambda_value is not None else "—"
        )

    def _run_drt_simple(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        settings = self._drt_settings()
        errors = []
        for ds in selected:
            try:
                self._drt_results[ds.label] = run_drt(
                    ds,
                    mode=self.drt_mode_combo.currentData(),
                    inductance=self.drt_inductance_check.isChecked(),
                    cross_validation=self.drt_cv_combo.currentData(),
                    lambda_value=self.drt_lambda_spin.value(),
                    credible_intervals=False,
                    **settings,
                )
            except Exception as exc:
                errors.append((ds.label, str(exc)))

        if errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in errors)
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")

        self._update_optimal_lambda_label(selected)
        self.statusBar().showMessage(f"Simple Run DRT computed for {len(selected) - len(errors)} sweep(s).")
        self._refresh()

    def _run_drt_bayesian(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        confirm = QMessageBox.question(
            self,
            "Bayesian DRT",
            "Computing Bayesian credible intervals can take a very long "
            "time (tens of minutes per sweep is common). It will run in "
            "the background, but may still be slow to finish. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        settings = self._drt_settings()
        mode = self.drt_mode_combo.currentData()
        inductance = self.drt_inductance_check.isChecked()
        cross_validation = self.drt_cv_combo.currentData()
        lambda_value = self.drt_lambda_spin.value()
        num_samples = self.drt_num_samples_spin.value()
        timeout = self.drt_timeout_spin.value()

        def runner(ds):
            return run_drt(
                ds,
                mode=mode,
                inductance=inductance,
                cross_validation=cross_validation,
                lambda_value=lambda_value,
                credible_intervals=True,
                num_samples=num_samples,
                timeout=timeout,
                **settings,
            )

        self._drt_worker_errors = []
        self._drt_worker = DRTWorker(runner, selected, parent=self)
        self._drt_worker.result_ready.connect(self._on_drt_worker_result)
        self._drt_worker.error.connect(self._on_drt_worker_error)
        self._drt_worker.finished.connect(self._on_drt_worker_finished)
        self._sidebar.setEnabled(False)
        self.statusBar().showMessage("Running Bayesian DRT… this may take a while.")
        self._drt_worker.start()

    def _on_drt_worker_result(self, label: str, result) -> None:
        self._drt_results[label] = result

    def _on_drt_worker_error(self, label: str, message: str) -> None:
        self._drt_worker_errors.append((label, message))

    def _on_drt_worker_finished(self) -> None:
        self._drt_worker = None
        self._sidebar.setEnabled(True)
        self.statusBar().showMessage("Bayesian DRT finished.")
        if self._drt_worker_errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in self._drt_worker_errors)
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")
        self._update_optimal_lambda_label(self._selected_datasets())
        self._refresh()

    def _run_drt_bht(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        settings = self._drt_settings()
        errors = []
        for ds in selected:
            try:
                self._drt_results[ds.label] = run_drt_bht(
                    ds,
                    num_samples=self.drt_num_samples_spin.value(),
                    **settings,
                )
            except Exception as exc:
                errors.append((ds.label, str(exc)))

        if errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in errors)
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")

        self._update_optimal_lambda_label(selected)
        self.statusBar().showMessage(
            f"Hilbert Transform DRT computed for {len(selected) - len(errors)} sweep(s)."
        )
        self._refresh()

    def _run_peak_analysis(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        num_peaks = self.drt_num_peaks_spin.value()
        errors = []
        for ds in selected:
            result = self._drt_results.get(ds.label)
            if result is None:
                errors.append((ds.label, "Run a DRT calculation first."))
                continue
            try:
                self._drt_peaks[ds.label] = analyze_drt_peaks(result, num_peaks=num_peaks)
            except Exception as exc:
                errors.append((ds.label, str(exc)))

        if errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in errors)
            QMessageBox.warning(self, "Peak analysis errors", f"Some sweeps failed:\n{details}")

        self.statusBar().showMessage(
            f"Peak analysis computed for {len(selected) - len(errors)} sweep(s)."
        )
        self._refresh()

    # ------------------------------------------------------------- refresh

    def _refresh(self) -> None:
        """Cheap, always-run bookkeeping: apply masks, validate against the
        current selection, and update tab labels/counts. The actual figures
        are (re)built lazily by _render_active_tab(), since replotting every
        tab (in particular one residuals figure per selected sweep) on every
        single checkbox click is wasted work for tabs the user isn't even
        looking at — this is what made overlaying many curves feel slow.
        """
        if not self._datasets:
            return

        selected = self._selected_datasets()
        if not selected:
            self.warning_label.setText("Select at least one sweep to plot.")
            self.warning_label.show()
            self.nyquist_pane.clear()
            self.residuals_pane.clear()
            self.drt_pane.clear()
            self.drt_peaks_text.clear()
            self.details_text.clear()
            self._pending = None
            self._tab_dirty.clear()
            return

        method = self._validation_method
        threshold = self.threshold_spin.value()

        for ds in selected:
            if self.inductive_check.isChecked():
                mask_inductive_points(ds)
            else:
                clear_mask(ds)

        stale_labels = []
        for ds in selected:
            result = self._validation_results.get((method, ds.label))
            if result is not None:
                try:
                    mask_residual_outliers(ds, result, threshold)
                except ValueError:
                    stale_labels.append(ds.label)

        if stale_labels:
            self.warning_label.setText(
                f"{method} results for {', '.join(stale_labels)} no longer match "
                f"the current mask (e.g. the inductive-tail filter changed) — "
                f"click 'Run {method} validation' again."
            )
            self.warning_label.show()
        else:
            self.warning_label.hide()

        validated_selected = [
            ds
            for ds in selected
            if (method, ds.label) in self._validation_results and ds.label not in stale_labels
        ]
        drt_selected = [
            (ds.label, self._drt_results[ds.label])
            for ds in selected
            if ds.label in self._drt_results
        ]
        self.tabs.setTabText(1, f"Residuals ({len(validated_selected)})")
        self.tabs.setTabText(2, f"DRT ({len(drt_selected)})")
        self.tabs.setTabText(
            3, f"DRT Peaks ({sum(1 for ds in selected if ds.label in self._drt_peaks)})"
        )

        self._pending = dict(
            selected=selected,
            show_removed=self.show_removed_check.isChecked(),
            method=method,
            threshold=threshold,
            validated_selected=validated_selected,
            drt_selected=drt_selected,
        )
        self._tab_dirty = {0, 1, 2, 3, 4}
        self._render_active_tab()

    def _on_tab_changed(self, _index: int) -> None:
        self._render_active_tab()

    def _render_active_tab(self) -> None:
        """Build the figures/text for whichever tab is currently visible, if
        it's still dirty. Other tabs stay dirty and get built the moment the
        user actually clicks over to them."""
        if self._pending is None:
            return
        index = self.tabs.currentIndex()
        if index not in self._tab_dirty:
            return
        p = self._pending

        if index == 0:
            if self._mode == "Single":
                fig, _ = plot_single(
                    p["selected"][0],
                    show=False,
                    style=self._style,
                    show_removed=p["show_removed"],
                )
            else:
                fig, _ = plot_overlay(
                    p["selected"], show=False, style=self._style, show_removed=p["show_removed"]
                )
            self.nyquist_pane.set_figure(fig)

        elif index == 1:
            residual_figs = []
            for ds in p["validated_selected"]:
                result = self._validation_results[(p["method"], ds.label)]
                fig_r, _ = plot_residuals(
                    result,
                    title=f"{p['method']} residuals — {ds.label}",
                    threshold=p["threshold"],
                    show=False,
                )
                residual_figs.append(fig_r)
            self.residuals_pane.set_figures(residual_figs)

        elif index == 2:
            if p["drt_selected"]:
                fig_drt, _ = plot_drt(p["drt_selected"], show=False)
                self.drt_pane.set_figure(fig_drt)
            else:
                self.drt_pane.clear()

        elif index == 3:
            peak_lines = []
            for ds in p["selected"]:
                peaks = self._drt_peaks.get(ds.label)
                if peaks is None:
                    continue
                peak_lines.append(f"=== {ds.label} ({peaks.get_num_peaks()} peak(s)) ===")
                peak_lines.append(peaks.to_peaks_dataframe().to_string(index=False))
                peak_lines.append("")
            self.drt_peaks_text.setPlainText("\n".join(peak_lines))

        elif index == 4:
            lines = []
            for ds in p["selected"]:
                validated_with = [
                    m for m in VALIDATION_METHODS if (m, ds.label) in self._validation_results
                ]
                note = f" (validated: {', '.join(validated_with)})" if validated_with else ""
                lines.append(f"{ds.label} — {ds.num_points} points{note}")
            self.details_text.setPlainText("\n".join(lines))

        self._tab_dirty.discard(index)
