r"""Test for checking how regular time series is."""

__all__ = [
    # Functions
    "approx_float_gcd",
    "float_gcd",
    "is_quasiregular",
    "is_regular",
    "regularity_coefficient",
    "time_gcd",
]

import warnings
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, Series


def approx_float_gcd(x: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08) -> float:
    r"""Compute approximate GCD of multiple floats.

    .. math:: 𝗀𝖼𝖽_ϵ(x) = 𝗆𝖺𝗑\{y∣ ∀i : 𝖽𝗂𝗌𝗍(x_i, yℤ)≤ϵ\}

    .. warning:: This implementation does not work 100% correctly yet!

    Parameters
    ----------
    x: ArrayLike
    rtol: float, default: 1e-05
    atol: float, default: 1e-08

    Returns
    -------
    float

    References
    ----------
    - https://stackoverflow.com/q/45323619/9318372
    """
    warnings.warn(
        "The implementation of approx_float_gcd does not work 100% correctly yet!"
    )
    x = np.abs(x).flatten()

    def _float_gcd(z: np.ndarray) -> float:
        n = len(z)
        t = np.min(z)
        if n == 1:
            return float(z[0])
        if n == 2:
            while np.abs(z[1]) > rtol * t + atol:
                z[0], z[1] = z[1], z[0] % z[1]
            return float(z[0])
        # n >= 3:
        out = np.empty(2)
        out[0] = _float_gcd(z[: (n // 2)])
        out[1] = _float_gcd(z[(n // 2) :])
        return _float_gcd(out)

    return _float_gcd(x)


def float_gcd(x: ArrayLike) -> float:
    r"""Compute the greatest common divisor (GCD) of a list of floats.

    Note that since floats are rational numbers, this is well-defined.

    Parameters
    ----------
    x: ArrayLike

    Returns
    -------
    float:
        The GCD of the list
    """
    x = np.asanyarray(x)

    assert np.issubdtype(x.dtype, np.floating), "input is not float!"

    mantissa_bits = {
        np.dtype("float16"): 11,
        np.dtype("float32"): 24,
        np.dtype("float64"): 53,
    }
    try:
        mantissa_bits[np.dtype("float128")] = 113
    except TypeError:
        pass
    mantissa_bits = mantissa_bits[x.dtype]

    _, e = np.frexp(x)
    min_exponent = int(np.min(e))
    fac = mantissa_bits - min_exponent
    z = x * np.float_power(2, fac)  # <- use float_power to avoid overflow!
    assert np.allclose(z, np.rint(z)), "something went wrong"

    gcd = np.gcd.reduce(np.rint(z).astype(int))
    gcd = gcd * 2 ** (-fac)

    z = x / gcd
    z_int = np.rint(z).astype(int)
    assert np.allclose(z, z_int), "Not a GCD!"
    assert np.gcd.reduce(z_int) == 1, "something went wrong"
    return cast(float, gcd)


def is_quasiregular(s: Series | DataFrame) -> bool:
    r"""Test if time series is quasi-regular.

    By definition, this is the case if all timedeltas are
    integer multiples of the minimal, non-zero timedelta of the series.

    Parameters
    ----------
    s: DataFrame

    Returns
    -------
    bool
    """
    if isinstance(s, DataFrame):
        return is_quasiregular(Series(s.index))

    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt_min = np.min(Δt[Δt > zero])
    z = Δt / Δt_min
    return np.allclose(z, np.rint(z))


def is_regular(s: Series | DataFrame) -> bool:
    r"""Test if time series is regular, i.e. iff $Δt_i$ is constant.

    Parameters
    ----------
    s: Series
        The timestamps

    Returns
    -------
    bool
    """
    if isinstance(s, DataFrame):
        return is_regular(Series(s.index))

    Δt = np.diff(s)
    return bool(np.all(Δt == np.min(Δt)))


def regularity_coefficient(
    s: Series | DataFrame, ignore_duplicates: bool = True
) -> float:
    r"""Compute the regularity coefficient of a time series.

    The regularity coefficient is equal to the ratio of length of the smallest regular time-series
    that contains s and the length of s.

    .. math:: κ(𝐭) = \frac{(t_\max-t_\min)/𝗀𝖼𝖽(𝐭)}{|𝐭|}

    In particular, if the time-series is regular, $κ=1$, and if it is irregular, $κ=∞$.
    To make the time-series regular, one would have to insert an additional $(κ(𝐭)-1)|𝐭|$ data-points.

    Parameters
    ----------
    s: Series
    ignore_duplicates: bool
        If `True`, data points with the same time-stamp will be treated as a single data point.

    Returns
    -------
    k:
        The regularity coefficient
    """
    if isinstance(s, DataFrame):
        return regularity_coefficient(Series(s.index))

    gcd = time_gcd(s)
    Δt = np.diff(s)
    if ignore_duplicates:
        zero = np.array(0, dtype=Δt.dtype)
        Δt = Δt[Δt > zero]
    # Δt_min = np.min(Δt)
    # return Δt_min / gcd
    coef: float = ((np.max(s) - np.min(s)) / gcd) / len(Δt)
    return coef


def time_gcd(s: Series) -> float:
    r"""Compute the greatest common divisor of datetime64/int/float data.

    Parameters
    ----------
    s: Series

    Returns
    -------
    gcd
    """
    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt = Δt[Δt > zero]

    if np.issubdtype(Δt.dtype, np.timedelta64):
        Δt = Δt.astype("timedelta64[ns]").astype(int)
        gcd = np.gcd.reduce(Δt)
        return gcd.astype("timedelta64[ns]")
    if np.issubdtype(Δt.dtype, np.integer):
        return np.gcd.reduce(Δt)
    if np.issubdtype(Δt.dtype, np.floating):
        return float_gcd(Δt)

    raise NotImplementedError(f"Data type {Δt.dtype=} not understood")


def distributiveness(s: Series) -> float:
    r"""Compute the distributiveness of a time series.

    For a given irregular timeseries, we define this as the minimum:

    .. math:: σ(TS) = \min\{ d(TS, TS') ∣ 𝐄[∆t(TS')] = 𝐄[∆t(TS)], TS' regular \}

    I.e. the minimum distance (for example Dynamic Time Warping) between the time series,
    and a regular time series with the same average frequency.
    """
    raise NotImplementedError
