from dataclasses import dataclass


@dataclass
class Header:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    valuedim: int
    value: str
    value_unit: str

    title: str = ""
    description: str = ""
    header_length: int = 0

    xmin: float | None = None
    ymin: float | None = None
    zmin: float | None = None
    xmax: float | None = None
    ymax: float | None = None
    zmax: float | None = None
    xbase: float | None = None
    ybase: float | None = None
    zbase: float | None = None

    def __post_init__(self):
        self.xmin = 0.0 if self.xmin is None else self.xmin
        self.ymin = 0.0 if self.ymin is None else self.ymin
        self.zmin = 0.0 if self.zmin is None else self.zmin

        self.xmax = self.nx * self.dx if self.xmax is None else self.xmax
        self.ymax = self.ny * self.dy if self.ymax is None else self.ymax
        self.zmax = self.nz * self.dz if self.zmax is None else self.zmax

        self.xbase = self.dx / 2 if self.xbase is None else self.xbase
        self.ybase = self.dy / 2 if self.ybase is None else self.ybase
        self.zbase = self.dz / 2 if self.zbase is None else self.zbase

    def get_dims(self):
        return self.nx, self.ny, self.nz, self.valuedim

    def get_nnodes(self):
        return self.nx * self.ny * self.nz * self.valuedim
