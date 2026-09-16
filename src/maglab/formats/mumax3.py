from pathlib import Path

import numpy as np
import pandas as pd


def read_table(filepath: Path | str) -> pd.DataFrame:
    """Reads a mumax3 output table and returns a pandas DataFrame."""
    filepath = Path(filepath)
    with filepath.open(mode="r") as file:
        names = file.readline().lstrip("# ").strip().split("\t")
        return pd.read_table(file, sep="\t", names=names)


def read_ovf_header(filepath: Path | str) -> dict:
    """
    Parse through an OVF2 file's header and return a dictionary containing
    all of the parameters and an additional value of the header's length.
    """
    header = {}
    filepath = Path(filepath)

    with filepath.open(mode="rb") as file:
        # Header should begin with "# OOMMF OVF 2.0"
        is_ovf2 = b"2.0" in next(file)
        if not is_ovf2:
            raise NotImplementedError("Only OVF2 is supported! Got OVF1")
        # Each line until the final reads in a format of # key: value
        for line in file:
            line = line.decode("utf-8")
            # Catch when header ends with "# Begin: Data Binary 4"
            if line.startswith("# Begin: Data"):
                break
            # Add values to header dictionary without any spaces or newlines.
            key, value = line.removeprefix("#").split(":", 1)
            header[key.strip()] = value.strip()
        # Add the header length into the header dictionary
        header["header_length"] = file.tell()

    return header


def read_ovf(filepath: Path | str) -> tuple[np.ndarray, dict]:
    """
    Read an OVF file and return a numpy array of its data in
    shape (nz, ny, nx, valuedim) and the header of metadata.
    """
    filepath = Path(filepath)
    # Extract relevant values from header
    header = read_ovf_header(filepath)
    nx, ny, nz = (int(header[f"{ax}nodes"]) for ax in "xyz")
    valuedim = int(header["valuedim"])
    nnodes = nx * ny * nz * valuedim

    with filepath.open(mode="rb") as file:
        file.seek(header["header_length"])  # Move to the end of the header.
        fmt = "<f4"
        # OVF 2.0 files have a value of 1234567.0 at the end of the header.
        # Check against it to make sure everything works.
        check_value = 1234567.0
        check = np.frombuffer(file.read(4), dtype=fmt)[0]
        if not np.isclose(check, check_value):
            raise ValueError(f"OVF binary check value mismatch: got {check}")

        arr = np.fromfile(file, count=nnodes, dtype=fmt).reshape(nz, ny, nx, valuedim)

    return arr, header


def read_ovf_array(filepath: Path | str) -> np.ndarray:
    """Read an OVF file and return a numpy array of its data in shape (nz, ny, nx, valuedim)."""
    return read_ovf(filepath)[0]
