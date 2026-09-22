from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .header import Header


def read_table(filepath: Path | str) -> pd.DataFrame:
    """Reads a mumax3 output table and returns a pandas DataFrame."""
    filepath = Path(filepath)
    with filepath.open(mode="r") as file:
        names = file.readline().lstrip("# ").strip().split("\t")
        return pd.read_table(file, sep="\t", names=names)


def read_ovf_header(filepath: Path | str) -> Header:
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

        # OVF 2.0 files have a value of 1234567.0 at the end of the header.
        # Check against it to make sure everything works.
        check = np.frombuffer(file.read(4), dtype="<f4")[0]
        check_value = 1234567.0
        if not np.isclose(check, check_value):
            raise ValueError(f"OVF binary check value mismatch: got {check}")

        header_length = file.tell()

    # Convert into Header class
    nx, ny, nz = (int(header[f"{ax}nodes"]) for ax in "xyz")
    xmin, ymin, zmin = (float(header[f"{ax}min"]) for ax in "xyz")
    xmax, ymax, zmax = (float(header[f"{ax}max"]) for ax in "xyz")
    dx, dy, dz = (float(header[f"{ax}stepsize"]) for ax in "xyz")
    xbase, ybase, zbase = (float(header[f"{ax}base"]) for ax in "xyz")
    valuedim = int(header["valuedim"])
    value = str(header["Title"])
    value_unit = str(header["valueunits"]).split(" ")[0]
    description = str(header["Desc"])

    return Header(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        valuedim=valuedim,
        value=value,
        value_unit=value_unit,
        title=value,
        description=description,
        header_length=header_length,
        xmin=xmin,
        ymin=ymin,
        zmin=zmin,
        xmax=xmax,
        ymax=ymax,
        zmax=zmax,
        xbase=xbase,
        ybase=ybase,
        zbase=zbase,
    )


def read_ovf(filepath: Path | str) -> tuple[np.ndarray, Header]:
    """
    Read an OVF file and return a numpy array of its data in
    shape (nz, ny, nx, valuedim) and the header of metadata.
    """
    filepath = Path(filepath)
    # Extract relevant values from header
    header = read_ovf_header(filepath)
    nx, ny, nz, valuedim = header.get_dims()
    nnodes = nx * ny * nz * valuedim

    dtype = "<f4"
    arr = np.fromfile(filepath, dtype=dtype, count=nnodes, offset=header.header_length)
    arr = arr.reshape(nz, ny, nx, valuedim)
    return arr, header


def read_ovf_array(filepath: Path | str) -> np.ndarray:
    """Read an OVF file and return a numpy array of its data in shape (nz, ny, nx, valuedim)."""
    return read_ovf(filepath)[0]
