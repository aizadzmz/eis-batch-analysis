"""Background workers so long-running analysis never freezes the UI."""

from typing import Callable, List

from PySide6.QtCore import QThread, Signal


class ValidationWorker(QThread):
    """Runs a validation method (KK or Z-HIT) over several datasets.

    Emits one result_ready per dataset so partial results land as they
    finish, plus error for any dataset that fails. QThread.finished fires
    when the whole batch is done.
    """

    result_ready = Signal(str, str, object)  # method name, dataset label, result
    error = Signal(str, str)                 # dataset label, message

    def __init__(self, method_name: str, runner: Callable, datasets: List, parent=None):
        super().__init__(parent)
        self._method_name = method_name
        self._runner = runner
        self._datasets = datasets

    def run(self) -> None:
        for ds in self._datasets:
            try:
                result = self._runner(ds)
            except Exception as exc:
                self.error.emit(ds.label, str(exc))
            else:
                self.result_ready.emit(self._method_name, ds.label, result)


class DRTWorker(QThread):
    """Runs a (potentially very slow) DRT calculation over several datasets
    off the UI thread — namely the Bayesian TR-RBF credible-interval run,
    whose HMC sampler can take tens of minutes per sweep.

    Emits one result_ready per dataset, plus error for any dataset that
    fails or times out. QThread.finished fires when the whole batch is done.
    """

    result_ready = Signal(str, object)  # dataset label, result
    error = Signal(str, str)            # dataset label, message

    def __init__(self, runner: Callable, datasets: List, parent=None):
        super().__init__(parent)
        self._runner = runner
        self._datasets = datasets

    def run(self) -> None:
        for ds in self._datasets:
            try:
                result = self._runner(ds)
            except Exception as exc:
                self.error.emit(ds.label, str(exc))
            else:
                self.result_ready.emit(ds.label, result)
