"""Entry point for the desktop GUI: python -m gui.app"""

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow

ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


def main() -> None:
    app = QApplication(sys.argv)
    # Organization/application names give QSettings a stable place to persist
    # the user's theme choice across launches.
    app.setOrganizationName("EIS Batch Analysis")
    app.setApplicationName("EIS Batch Analysis")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()  # applies the saved (or default) theme itself
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
