"""Minimum-energy path (MEP) on a reconstructed 2D PMF.

Given the GP-reconstructed free-energy surface produced by
:func:`gpr_umbrella_1d.gpr2d.gpr_umbrella_integration_2d`, this module

  1. locates the local minima of the surface,
  2. connects a chosen pair with a minimum-energy path over the grid, and
  3. reports the energy and uncertainty profile along that path — the barrier,
     the reaction free energy, and the transition-state location.

The path is the *minimum-bottleneck* (minimax) path: of all routes between the
two endpoints it is the one whose highest point is as low as possible, i.e. the
one crossing the lowest saddle.  Its peak is therefore the true barrier.  The
path rides on the existing GP grid, so PMF and calibrated sigma are read
straight off the surface.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import minimum_filter


def find_minima(pmf: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                max_minima: int = 8) -> list[dict]:
    """Local minima of *pmf* (shape ``(nx, ny)`` on axes ``gx``, ``gy``).

    Returns ``{"ij", "xy", "energy"}`` dicts sorted by energy (deepest first),
    de-duplicated so clustered/plateau cells collapse to one per basin.
    """
    mn = minimum_filter(pmf, size=3, mode="nearest")
    ii, jj = np.where(pmf == mn)
    order = np.argsort(pmf[ii, jj])
    ii, jj = ii[order], jj[order]

    kept: list[dict] = []
    for i, j in zip(ii, jj):
        if any(abs(i - m["ij"][0]) <= 2 and abs(j - m["ij"][1]) <= 2
               for m in kept):
            continue
        kept.append({"ij": (int(i), int(j)),
                     "xy": (float(gx[i]), float(gy[j])),
                     "energy": float(pmf[i, j])})
        if len(kept) >= max_minima:
            break
    return kept


def _snap(xy, gx, gy) -> tuple[int, int]:
    """Nearest grid cell to a physical (x, y) point."""
    return int(np.argmin(np.abs(gx - xy[0]))), int(np.argmin(np.abs(gy - xy[1])))


def _minimax_path(pmf, start_ij, end_ij, dx, dy):
    """Minimum-bottleneck path: minimise the maximum PMF along the route, with
    total length as a tie-break so the path stays clean on flat ground."""
    nx, ny = pmf.shape
    diag = float(np.hypot(dx, dy))
    steps = [(-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
             (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]

    best = np.full((nx, ny), np.inf)          # best (bottleneck) reaching a cell
    best_len = np.full((nx, ny), np.inf)
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    si, sj = start_ij
    ti, tj = end_ij
    best[si, sj] = pmf[si, sj]
    best_len[si, sj] = 0.0
    pq = [(pmf[si, sj], 0.0, si, sj)]
    while pq:
        b, length, i, j = heapq.heappop(pq)
        if (b, length) > (best[i, j], best_len[i, j]):
            continue
        if (i, j) == (ti, tj):
            break
        for di, dj, dist in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                nb = max(b, pmf[ni, nj])
                nl = length + dist
                if (nb, nl) < (best[ni, nj], best_len[ni, nj]):
                    best[ni, nj] = nb
                    best_len[ni, nj] = nl
                    prev[(ni, nj)] = (i, j)
                    heapq.heappush(pq, (nb, nl, ni, nj))

    if not np.isfinite(best[ti, tj]):
        raise RuntimeError("No path found between the requested endpoints.")
    path = [(ti, tj)]
    while path[-1] != (si, sj):
        path.append(prev[path[-1]])
    return path[::-1]


def find_mep(results: dict, endpoints=None, max_minima: int = 8) -> dict:
    """Locate minima and the minimum-energy path between a pair of them.

    Parameters
    ----------
    results
        Output dict of :func:`gpr_umbrella_integration_2d` (needs ``gx``, ``gy``,
        ``pmf``, ``pmf_std``).
    endpoints
        Optional ``((x0, y0), (x1, y1))`` physical coordinates to connect (each
        snapped to the nearest grid cell).  Default: the two deepest minima.

    Returns a dict with the path (``s``, ``x``, ``y``, ``pmf``, ``pmf_rel``,
    ``sigma``), the ``minima`` list, ``barrier``/``barrier_err``,
    ``delta_f``/``delta_f_err`` and the transition state (``ts_xy``, ``ts_s``).
    The barrier error is the conservative ``sqrt(sigma_TS^2 + sigma_start^2)``.
    """
    gx, gy = results["gx"], results["gy"]
    pmf, pmf_std = results["pmf"], results["pmf_std"]
    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0

    minima = find_minima(pmf, gx, gy, max_minima=max_minima)
    if endpoints is not None:
        start_ij = _snap(endpoints[0], gx, gy)
        end_ij = _snap(endpoints[1], gx, gy)
    else:
        if len(minima) < 2:
            raise ValueError(
                f"Need >=2 minima to build an MEP, found {len(minima)}. "
                "Pass endpoints=((x0,y0),(x1,y1)) explicitly.")
        start_ij, end_ij = minima[0]["ij"], minima[1]["ij"]

    # The path rides on the grid nodes; PMF and its (GP posterior, LOO-
    # calibrated) sigma are read straight off the surface.
    path = _minimax_path(pmf, start_ij, end_ij, dx, dy)
    pi = np.array([p[0] for p in path])
    pj = np.array([p[1] for p in path])
    px, py = gx[pi], gy[pj]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    E = pmf[pi, pj]
    sig = pmf_std[pi, pj]
    E_rel = E - E[0]
    ts = int(np.argmax(E_rel))

    return {
        "s": s, "x": px, "y": py, "pmf": E, "pmf_rel": E_rel, "sigma": sig,
        "minima": minima,
        "start_xy": (float(px[0]), float(py[0])),
        "end_xy": (float(px[-1]), float(py[-1])),
        "ts_xy": (float(px[ts]), float(py[ts])), "ts_s": float(s[ts]),
        "barrier": float(E_rel[ts]),
        "barrier_err": float(np.hypot(sig[ts], sig[0])),
        "delta_f": float(E_rel[-1]),
        "delta_f_err": float(np.hypot(sig[-1], sig[0])),
        "cv_names": results.get("cv_names", ("cv0", "cv1")),
        "cv_units": results.get("cv_units", ("", "")),
        "energy_unit": results.get("energy_unit", "eV"),
    }


def save_mep(mep: dict, path: str) -> None:
    """Write the MEP energy/error profile to a headed text file."""
    cvn, cvu = mep["cv_names"], mep["cv_units"]
    eu = mep["energy_unit"]
    header = (
        f"Minimum-energy path\n"
        f"start = ({mep['start_xy'][0]:.4f}, {mep['start_xy'][1]:.4f}) {cvu[0]}\n"
        f"end   = ({mep['end_xy'][0]:.4f}, {mep['end_xy'][1]:.4f}) {cvu[0]}\n"
        f"transition state = ({mep['ts_xy'][0]:.4f}, {mep['ts_xy'][1]:.4f}) "
        f"at s = {mep['ts_s']:.4f} {cvu[0]}\n"
        f"barrier     = {mep['barrier']:.4f} +/- {mep['barrier_err']:.4f} {eu}\n"
        f"reaction dF = {mep['delta_f']:.4f} +/- {mep['delta_f_err']:.4f} {eu}\n"
        f"s({cvu[0]})  {cvn[0]}({cvu[0]})  {cvn[1]}({cvu[1]})  "
        f"PMF({eu})  PMF_rel_start({eu})  sigma({eu})"
    )
    data = np.column_stack([mep["s"], mep["x"], mep["y"],
                            mep["pmf"], mep["pmf_rel"], mep["sigma"]])
    np.savetxt(path, data, header=header, fmt="%.6f")
