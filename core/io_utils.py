# This file will read files, implement PyProBE's file parser, return the relevant plots.
# Currently, it only reads .mpt file but I will return and add the ability to read .txt and .csv

from pathlib import Path
import pyprobe


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension has no registered reader."""
    pass


def _read_mpt(filepath: Path, cell_info: dict | None = None):
    """Read a Bio-Logic .mpt file using PyProBE.

    Args:
        filepath: Path to the .mpt file.
        cell_info: Metadata dict for the PyProBE Cell object (e.g. Name,
            Chemistry). Defaults to a minimal placeholder if not provided.

    Returns:
        The PyProBE Procedure object for this file.
    """
    if cell_info is None:
        cell_info = {"Name": filepath.stem}

    cell = pyprobe.Cell(info=cell_info)
    cell.import_from_cycler(
        procedure_name=filepath.stem,
        cycler="biologic",
        input_data_path=str(filepath),
    )
    return cell.procedure[filepath.stem]


def _read_txt(filepath: Path):
    """Read a plaintext EIS export file.

    Args:
        filepath: Path to the .txt file.

    Returns:
        Parsed data ready for downstream analysis.
    """
    raise NotImplementedError("TXT reading not yet implemented.")


def _read_csv(filepath: Path):
    """Read a CSV EIS export file.

    Args:
        filepath: Path to the .csv file.

    Returns:
        Parsed data ready for downstream analysis.
    """
    raise NotImplementedError("CSV reading not yet implemented.")


# Maps file extensions to their reader function.
# Add new formats here as they're implemented - no other code needs to change.
_READERS = {
    ".mpt": _read_mpt,
    ".txt": _read_txt,
    ".csv": _read_csv,
}


def load_eis_file(filepath: Path):
    """Load a single EIS data file, dispatching based on file extension.

    Args:
        filepath: Path to the raw data file.

    Returns:
        Parsed data ready for downstream analysis.

    Raises:
        UnsupportedFileTypeError: If the file extension has no registered reader.
    """
    extension = filepath.suffix.lower()
    reader = _READERS.get(extension)

    if reader is None:
        raise UnsupportedFileTypeError(
            f"No reader available for '{extension}' files ({filepath.name})."
        )

    return reader(filepath)


def load_batch(directory: Path) -> list:
    """Load all supported EIS files found in a directory.

    Args:
        directory: Folder containing raw data files.

    Returns:
        List of parsed EIS datasets, one per successfully loaded file.
    """
    results = []
    for filepath in sorted(directory.iterdir()):
        if filepath.suffix.lower() not in _READERS:
            continue  # silently skip unsupported file types
        results.append(load_eis_file(filepath))
    return results