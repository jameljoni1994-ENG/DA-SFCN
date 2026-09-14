"""Dynamically-Adaptive Saddle-Free Cubic Newton family."""

from .algorithms import (
    DA_SFCN,
    KrylovCRN,
    SaddleFreeNewton,
    CubicNewton,
    GradientDescent,
    LBFGS,
    VR_DA_SFCN,
    LK_Newton,
    WQK_Newton,
    Fed_DA_SFCN,
    SC_Newton,
    RandomSubspaceCubic,
)
from .problems import (
    Rosenbrock,
    QuadraticSaddle,
    MonkeySaddle,
    NonconvexLogistic,
    MatrixFactorization,
    FiniteSumQuadratics,
)

__all__ = [
    "DA_SFCN",
    "KrylovCRN",
    "SaddleFreeNewton",
    "CubicNewton",
    "GradientDescent",
    "LBFGS",
    "VR_DA_SFCN",
    "LK_Newton",
    "WQK_Newton",
    "Fed_DA_SFCN",
    "SC_Newton",
    "RandomSubspaceCubic",
    "Rosenbrock",
    "QuadraticSaddle",
    "MonkeySaddle",
    "NonconvexLogistic",
    "MatrixFactorization",
    "FiniteSumQuadratics",
]
