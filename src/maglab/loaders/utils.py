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
