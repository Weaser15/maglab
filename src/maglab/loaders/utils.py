from collections.abc import Sequence

import numpy as np
import pandas as pd


def compute_dot_vectors(
    df: pd.DataFrame,
    names: list[str],
    direction: str | tuple[float, float, float] | np.ndarray,
    comps: tuple[str, ...] = ("x", "y", "z"),
) -> list[np.ndarray]:

    # Get normalised direction
    if isinstance(direction, str):
        mask = df.columns.str.contains("|".join(direction + c for c in comps))
        direction = df.loc[0, mask].to_numpy()  # type:ignore
    norm_direction = np.array(direction) / np.linalg.norm(direction)
    dotted = []
    for name in names:
        print(name)
        # Find all columns that contain name + comp
        mask = df.columns.str.contains("|".join(name + c for c in comps))
        dotted.append(np.dot(df.loc[:, mask].to_numpy(), norm_direction))
    return dotted


def apply_template(
    df: pd.DataFrame, column: str, template: str, types: Sequence[type] | None = None
) -> pd.DataFrame:
    df = df.copy()
    if "?P" not in template:
        raise ValueError("?P not found in template! Use ?P<name>.+ for simple matching.")

    extracted = df[column].str.extract(template)
    if types is not None:
        extracted = extracted.astype(dict(zip(extracted.columns, types, strict=True)))
    loc: int = df.columns.get_loc(column)  # type:ignore
    df.pop(column)
    for i, (name, ser) in enumerate(extracted.items()):
        df.insert(loc + i, name, ser)

    return df
