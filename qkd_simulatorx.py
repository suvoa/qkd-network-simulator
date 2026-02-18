
import time
import math
import random
import traceback
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# SeQUeNCe IMPORTS
# ============================================================================
try:
    from sequence.kernel.timeline import Timeline
    from sequence.topology.node import QKDNode
    from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
    from sequence.qkd.BB84 import pair_bb84_protocols
    SEQUENCE_AVAILABLE = True
    print("[OK] SeQUeNCe library loaded successfully")
except ImportError as e:
    SEQUENCE_AVAILABLE = False
    print(f"[X] SeQUeNCe not available: {e}")
    print("  Install with: pip install sequence")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
import networkx as nx

SECOND = 1e12  # SeQUeNCe uses picoseconds

# ============================================================================
# ENUMS & CONFIGURATION
# ============================================================================

class ChannelType(Enum):
    FIBER = "fiber"
    FSO_GROUND = "fso_ground"
    FSO_UAV = "fso_uav"
    FSO_SATELLITE = "fso_satellite"


class NodeTrust(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


# --------------- physics dataclasses ---------------

@dataclass
class FiberPhysics:
    attenuation: float = 0.2        # dB/km (standard SMF-28)
    efficiency: float = 0.9
    darkcount: float = 100.0        # Hz
    baseerror: float = 0.01         # 1% intrinsic QBER
    pmderror: float = 0.0003        # polarisation-mode-dispersion error /km
    jitter: float = 0.002


@dataclass
class FSOPhysics:
    wavelength: float = 1550e-9     # m
    txaperture: float = 0.1         # m
    rxaperture: float = 0.3         # m
    beamdivergence: float = 10e-6   # rad
    pointingerror: float = 1e-6     # rad
    efficiency: float = 0.85
    darkcount: float = 50.0         # Hz
    baseerror: float = 0.005
    jitter: float = 0.003


@dataclass
class AtmosphericConditions:
    visibility: float = 23.0        # km
    cn2: float = 1e-15              # refractive-index structure constant
    weathertype: str = "clear"
    zenithangle: float = 0.0
    altitude: float = 0.0


@dataclass
class UAVConfig:
    altitude: float = 1.0           # km
    speed: float = 50.0             # m/s
    hovertime: float = 30.0         # s
    maxrange: float = 50.0          # km


@dataclass
class SatelliteConfig:
    """Table I parameters from arXiv:2507.23466"""
    orbittype: str = "GEO"
    altitude: float = 35786.0       # km  (GEO default)
    passtime: float = 86400.0       # s   (GEO -> continuous)
    elevation: float = 30.0         # deg (paper default)
    Dsat: float = 0.50              # m   satellite telescope aperture
    DOGS: float = 1.0               # m   OGS aperture (20 cm - 1 m)
    tau_syst_dB: float = 2.8        # dB  internal optical system loss
    theta_jitter: float = 0.07e-6   # rad residual tracking error
    paa: float = 18.5e-6            # rad point-ahead angle


# --------------- main simulation config ---------------

@dataclass
class SimConfig:
    nodecount: int = 4
    topology: str = "linear"
    protocol: str = "BB84"
    keylength: int = 256
    keycount: int = 50
    duration: float = 10.0          # seconds
    epsilon: float = 1e-9
    ecefficiency: float = 1.16
    usefinitesize: bool = False
    securitymodel: str = "shannon"   # "shannon" or "gllp" (BB84 only)
    gllp_mu: float = 0.5             # signal intensity (mean photons/pulse)
    gllp_nu: float = 0.1             # weak decoy intensity (nu < mu)
    gllp_optimize: bool = True       # auto-optimize mu/nu for GLLP
    gllp_clock_hz: float = 1e9       # pulse clock for dark-yield conversion
    channeltype: ChannelType = ChannelType.FIBER
    fiberphysics: FiberPhysics = field(default_factory=FiberPhysics)
    fsophysics: FSOPhysics = field(default_factory=FSOPhysics)
    atmosphere: AtmosphericConditions = field(default_factory=AtmosphericConditions)
    uavconfig: UAVConfig = field(default_factory=UAVConfig)
    satelliteconfig: SatelliteConfig = field(default_factory=SatelliteConfig)
    linkdistances: Dict[Tuple[int, int], float] = field(default_factory=dict)
    nodetrust: Dict[int, NodeTrust] = field(default_factory=dict)

    # TF-QKD parameters (Table I of arXiv:2507.23466)
    tfqkd_darkcount: float = 1e-8       # p_d
    tfqkd_detector_eff: float = 0.70    # eta__D
    tfqkd_repetition_rate: float = 2.5e9  # F  (Hz)
    tfqkd_misalignment: float = 0.001   # e_d  (~0.1%)
    tfqkd_fec: float = 1.1              # f_EC (paper value)
    tfqkd_detector_deadtime_ns: float = 20.0   # detector dead time (ns)
    tfqkd_detector_count: int = 2              # detectors at untrusted relay
    tfqkd_saturation_model: str = "nonparalyzable"  # none|nonparalyzable|paralyzable

    def getlinkdistance(self, a: int, b: int) -> float:
        if (a, b) in self.linkdistances:
            return self.linkdistances[(a, b)]
        if (b, a) in self.linkdistances:
            return self.linkdistances[(b, a)]
        return 10.0

    def getnodtrust(self, nid: int) -> NodeTrust:
        return self.nodetrust.get(nid, NodeTrust.TRUSTED)


# ============================================================================
# UTILITY HELPERS
# ============================================================================

def getlinklist(topology: str, n: int) -> List[Tuple[int, int]]:
    t = topology.lower()
    if t == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    elif t == "star":
        return [(0, i) for i in range(1, n)]
    elif t == "mesh":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return []


def gettopologyart(topology: str, n: int, trust: Dict[int, NodeTrust] = None) -> str:
    def ns(i):
        if trust and trust.get(i) == NodeTrust.UNTRUSTED:
            return f"[N{i}*]"
        return f"[N{i}]"

    t = topology.lower()
    if t == "linear":
        return "  " + " --- ".join(ns(i) for i in range(n))
    elif t == "star":
        hub = f"       {ns(0)} (hub)\n"
        hub += "      " + "/ " * (n - 1) + "\n"
        hub += "    " + "  ".join(ns(i) for i in range(1, n))
        return hub
    else:
        lines = ["  Mesh connections:"]
        for i in range(n):
            for j in range(i + 1, n):
                lines.append(f"    {ns(i)} <---> {ns(j)}")
        return "\n".join(lines)


# ============================================================================
# CHANNEL LOSS MODELS
# ============================================================================

class ChannelModels:
    """Compute (attenuation_per_m, polarisation_fidelity) for SeQUeNCe."""

    @staticmethod
    def get_sequence_params(distkm: float, cfg: SimConfig) -> Tuple[float, float]:
        ct = cfg.channeltype
        if ct == ChannelType.FIBER:
            return ChannelModels.fiber_params(distkm, cfg.fiberphysics)
        elif ct == ChannelType.FSO_GROUND:
            return ChannelModels.fso_ground_params(distkm, cfg.fsophysics, cfg.atmosphere)
        elif ct == ChannelType.FSO_UAV:
            return ChannelModels.fso_uav_params(distkm, cfg.fsophysics, cfg.atmosphere, cfg.uavconfig)
        elif ct == ChannelType.FSO_SATELLITE:
            return ChannelModels.fso_satellite_params(distkm, cfg.fsophysics, cfg.atmosphere, cfg.satelliteconfig)
        return ChannelModels.fiber_params(distkm, cfg.fiberphysics)

    # ---------- Fiber ----------
    @staticmethod
    def fiber_params(distkm: float, ph: FiberPhysics) -> Tuple[float, float]:
        att_per_m = ph.attenuation / 1000.0                       # dB/m
        qber = min(0.5, ph.baseerror + ph.pmderror * distkm
                   + random.uniform(0, ph.jitter * 0.5))
        fid = max(0.5, 1.0 - 2.0 * qber)
        return att_per_m, fid

    # ---------- FSO Ground ----------
    @staticmethod
    def fso_ground_params(distkm: float, ph: FSOPhysics,
                          atm: AtmosphericConditions) -> Tuple[float, float]:
        vis = atm.visibility
        wl_nm = ph.wavelength * 1e9
        q = 1.6 if vis > 50 else (1.3 if vis > 6 else 0.585 * vis ** (1 / 3))
        alpha = (3.91 / vis) * (wl_nm / 550) ** (-q)
        wp = {"clear": 0.0, "hazy": 2.0, "foggy": 10.0, "rainy": 5.0}.get(atm.weathertype, 0.0)
        dm = distkm * 1000.0
        br = dm * ph.beamdivergence
        geom = min(1.0, (ph.rxaperture / (2 * br)) ** 2)
        geom_db = -10 * math.log10(max(geom, 1e-10))
        point = math.exp(-(ph.pointingerror ** 2) / (ph.beamdivergence ** 2))
        point_db = -10 * math.log10(max(point, 1e-10))
        total_db_km = alpha + (wp + geom_db + point_db) / max(distkm, 0.1)
        att_per_m = total_db_km / 1000.0
        turb_err = 0.001 * (atm.cn2 * 1e15) * distkm
        qber = min(0.5, ph.baseerror + turb_err + random.uniform(0, ph.jitter * 0.5))
        fid = max(0.5, 1.0 - 2.0 * qber)
        return att_per_m, fid

    # ---------- FSO UAV ----------
    @staticmethod
    def fso_uav_params(distkm: float, ph: FSOPhysics,
                       atm: AtmosphericConditions, uav: UAVConfig) -> Tuple[float, float]:
        slant = math.sqrt(distkm ** 2 + uav.altitude ** 2)
        sf = math.exp(-uav.altitude / 8.5)
        scaled = AtmosphericConditions(
            visibility=atm.visibility / max(sf, 0.1),
            cn2=atm.cn2 * sf,
            weathertype=atm.weathertype,
        )
        att, fid = ChannelModels.fso_ground_params(slant, ph, scaled)
        fid = max(0.5, fid - 0.002)   # vibration penalty
        return att, fid

    # ---------- FSO Satellite (paper secII) ----------
    @staticmethod
    def fso_satellite_params(distkm: float, ph: FSOPhysics,
                             atm: AtmosphericConditions,
                             sat: SatelliteConfig) -> Tuple[float, float]:
        elev_rad = math.radians(max(sat.elevation, 5.0))
        atmos_thick = 20.0 / math.sin(elev_rad)                  # effective path km
        vis = atm.visibility
        wl_nm = ph.wavelength * 1e9
        q = 1.6 if vis > 50 else (1.3 if vis > 6 else 0.585 * vis ** (1 / 3))
        alpha = (3.91 / vis) * (wl_nm / 550) ** (-q)
        atmos_db = alpha * min(atmos_thick, 50)

        # Geometric loss  tau__geom = (pi D_OGS D_sat / 4lambda_ L)^2   (Eq. 2)
        L_m = distkm * 1000.0
        lam = ph.wavelength
        D_ogs = sat.DOGS
        D_sat = sat.Dsat
        geom_eff = (math.pi * D_ogs * D_sat / (4.0 * lam * L_m)) ** 2
        geom_db = -10 * math.log10(max(geom_eff, 1e-30))

        # System loss (Table I)
        syst_db = sat.tau_syst_dB

        # Scintillation / turbulence
        cn2 = atm.cn2
        k0 = 2 * math.pi / lam
        zenith = 90.0 - sat.elevation
        rytov = 1.23 * cn2 * (k0 ** 1.17) * \
                (atmos_thick * 1000) ** (11 / 6) * \
                (math.cos(math.radians(zenith))) ** (-11 / 6)
        turb_db = 4.34 * rytov / 2.0

        total_db = atmos_db + geom_db + syst_db + turb_db
        eff_att_km = total_db / max(distkm, 1.0)
        att_per_m = eff_att_km / 1000.0

        turbloss = math.exp(-rytov / 2.0)
        qber = min(0.5, ph.baseerror + 0.001 + 0.002 * (1 - turbloss))
        fid = max(0.5, 1.0 - 2.0 * qber)
        return att_per_m, fid

    # ---------- Transmission helper ----------
    @staticmethod
    def get_transmission(distkm: float, cfg: SimConfig) -> float:
        att_per_m, _ = ChannelModels.get_sequence_params(distkm, cfg)
        loss_db = att_per_m * distkm * 1000.0   # att(dB/m) x distance(m)
        return 10 ** (-loss_db / 10.0)

    # ---------- Satellite link budget (paper Eq. 1-2, Table I) ----------
    @staticmethod
    def satellite_link_transmission(distkm: float, sat: SatelliteConfig,
                                    atm: AtmosphericConditions) -> float:
        """
        Full link budget from Eq.(1): tau_ = eta__turb * eta__jitter * tau__abs * tau__syst * tau__geom
        Calibrated to match paper Fig. 6 results (~50-65 dB for D_OGS 20-100 cm).
        Returns total one-way channel transmittance.
        """
        lam = 1550e-9
        L = distkm * 1000.0
        elev_rad = math.radians(max(sat.elevation, 5.0))

        # tau__geom  (Eq. 2)
        tau_geom = min(1.0, (math.pi * sat.DOGS * sat.Dsat / (4.0 * lam * L)) ** 2)

        # tau__syst  (Table I: 2.8 dB)
        tau_syst = 10 ** (-sat.tau_syst_dB / 10.0)

        # tau__abs  (atmospheric molecular absorption through slant path)
        atmos_km = 20.0 / math.sin(elev_rad)
        vis = atm.visibility
        wl_nm = lam * 1e9
        q = 1.6 if vis > 50 else (1.3 if vis > 6 else 0.585 * vis ** (1 / 3))
        alpha_db = (3.91 / vis) * (wl_nm / 550) ** (-q)
        tau_abs = 10 ** (-alpha_db * min(atmos_km, 50) / 10.0)

        # eta__turb  (mean values from Table II, interpolated by D_OGS)
        # Paper values (MMSE correction): D=20cm->0.73, 40->0.66, 60->0.61, 80->0.58, 100->0.56
        D_cm = sat.DOGS * 100.0
        if D_cm <= 20:
            eta_turb = 0.73
        elif D_cm <= 40:
            eta_turb = 0.73 - (D_cm - 20) / 20 * 0.07
        elif D_cm <= 60:
            eta_turb = 0.66 - (D_cm - 40) / 20 * 0.05
        elif D_cm <= 80:
            eta_turb = 0.61 - (D_cm - 60) / 20 * 0.03
        elif D_cm <= 100:
            eta_turb = 0.58 - (D_cm - 80) / 20 * 0.02
        else:
            eta_turb = max(0.35, 0.56 - (D_cm - 100) / 200 * 0.1)

        # eta__jitter  (satellite pointing jitter, Eq. 4 with Weibull parameters)
        # For theta__jitter = 0.07 urad, the mean coupling efficiency is ~0.85-0.95
        # (beam wander is small compared to beam width at satellite FOV)
        # Using simplified model: eta__jitter ~ exp(-2 * (theta__jitter/theta__div)^2)
        # where theta__div = lambda_ / D_OGS (diffraction divergence of OGS)
        theta_div = lam / sat.DOGS  # uplink beam divergence from OGS
        ratio = sat.theta_jitter / theta_div if theta_div > 0 else 0
        eta_jitter = math.exp(-2.0 * ratio ** 2)

        tau = eta_turb * eta_jitter * tau_abs * tau_syst * tau_geom

        # Add random scintillation fluctuation (+/-10%)
        fluct = random.uniform(0.9, 1.1)
        tau *= fluct

        return max(tau, 1e-30)


# ============================================================================
# CRYPTOGRAPHIC MATHS
# ============================================================================

class CryptoMath:
    @staticmethod
    def h(p: float) -> float:
        """Binary entropy H(p)."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    @staticmethod
    def finite_correction(n: int, eps: float = 1e-9) -> float:
        if n <= 0:
            return 1.0
        return 7.0 * math.sqrt(math.log(2.0 / eps) / n)

    @staticmethod
    def secure_fraction(qber: float, fec: float = 1.16,
                        n_sifted: Optional[int] = None,
                        eps: float = 1e-9, finite: bool = True) -> float:
        if qber >= 0.11:
            return 0.0
        he = CryptoMath.h(qber)
        frac = 1.0 - (1.0 + fec) * he
        if finite and n_sifted is not None and n_sifted > 0:
            frac -= CryptoMath.finite_correction(n_sifted, eps)
        return max(0.0, frac)

    @staticmethod
    def secure_bits(sifted: int, qber: float, fec: float = 1.16,
                    eps: float = 1e-9, finite: bool = True) -> int:
        f = CryptoMath.secure_fraction(qber, fec, sifted, eps, finite)
        return max(0, int(sifted * f))

    @staticmethod
    def dark_yield_per_pulse(dark_rate_hz: float, clock_hz: float,
                             detectors: int = 2) -> float:
        """Approximate vacuum yield Y0 from detector dark count rate."""
        if dark_rate_hz <= 0 or clock_hz <= 0 or detectors <= 0:
            return 0.0
        y0 = detectors * dark_rate_hz / clock_hz
        return max(0.0, min(1.0, y0))

    @staticmethod
    def decoy_gains_from_channel(mu: float, nu: float, eta: float, y0: float) -> Tuple[float, float]:
        """
        Gain model for weak coherent states:
          Q_x = Y0 + 1 - exp(-eta * x)
        """
        mu = max(1e-9, float(mu))
        nu = max(1e-9, min(float(nu), 0.999 * mu))
        eta = max(0.0, min(1.0, float(eta)))
        y0 = max(0.0, min(1.0, float(y0)))
        q_mu = y0 + 1.0 - math.exp(-eta * mu)
        q_nu = y0 + 1.0 - math.exp(-eta * nu)
        return max(1e-15, q_mu), max(1e-15, q_nu)

    @staticmethod
    def decoy_bounds_two_intensity(mu: float, nu: float, q_mu: float, q_nu: float,
                                   e_nu: float, y0: float) -> Tuple[float, float, float]:
        """
        Vacuum+weak-decoy bounds (Lo-Ma-Chen, PRL 94, 230504):
          Y1_lower, e1_upper, Q1_lower
        """
        mu = max(1e-9, float(mu))
        nu = max(1e-9, min(float(nu), 0.999 * mu))
        q_mu = max(1e-15, float(q_mu))
        q_nu = max(1e-15, float(q_nu))
        e_nu = max(0.0, min(0.5, float(e_nu)))
        y0 = max(0.0, min(1.0, float(y0)))

        denom = mu * nu - nu * nu
        if denom <= 0:
            return 0.0, 0.5, 0.0

        term = (
            q_nu * math.exp(nu)
            - (nu * nu / (mu * mu)) * q_mu * math.exp(mu)
            - ((mu * mu - nu * nu) / (mu * mu)) * y0
        )
        y1_lower = (mu / denom) * term
        y1_lower = max(0.0, min(1.0, y1_lower))

        q1_lower = mu * math.exp(-mu) * y1_lower
        q1_lower = max(0.0, min(q_mu, q1_lower))

        if y1_lower <= 0:
            e1_upper = 0.5
        else:
            e1_upper = (e_nu * q_nu * math.exp(nu) - 0.5 * y0) / max(y1_lower * nu, 1e-15)
            e1_upper = max(0.0, min(0.5, e1_upper))

        return y1_lower, e1_upper, q1_lower

    @staticmethod
    def secure_fraction_gllp(qber: float, fec: float = 1.16, mu: float = 0.5,
                             nu: float = 0.1, eta: float = 0.1, y0: float = 1e-7,
                             n_sifted: Optional[int] = None,
                             eps: float = 1e-9, finite: bool = True) -> float:
        """
        GLLP with vacuum+weak decoy estimation:
          R >= Q1_lower * (1 - H2(e1_upper)) - fEC * Q_mu * H2(E_mu)
        Returned value is normalized per sifted signal bit:
          frac = (Q1_lower / Q_mu) * (1 - H2(e1_upper)) - fEC * H2(E_mu)
        """
        e_mu = max(0.0, min(0.5, float(qber)))
        if e_mu >= 0.5:
            return 0.0

        q_mu, q_nu = CryptoMath.decoy_gains_from_channel(mu, nu, eta, y0)
        e0 = 0.5
        ed = (e_mu * q_mu - e0 * y0) / max(q_mu - y0, 1e-15)
        ed = max(0.0, min(0.5, ed))
        e_nu = (e0 * y0 + ed * max(0.0, q_nu - y0)) / max(q_nu, 1e-15)
        e_nu = max(0.0, min(0.5, e_nu))

        _, e1_upper, q1_lower = CryptoMath.decoy_bounds_two_intensity(
            mu, nu, q_mu, q_nu, e_nu, y0
        )

        frac = (q1_lower / max(q_mu, 1e-15)) * (1.0 - CryptoMath.h(e1_upper)) - fec * CryptoMath.h(e_mu)
        if finite and n_sifted is not None and n_sifted > 0:
            frac -= CryptoMath.finite_correction(n_sifted, eps)
        return max(0.0, frac)

    @staticmethod
    def secure_bits_gllp(sifted: int, qber: float, fec: float = 1.16, mu: float = 0.5,
                         nu: float = 0.1, eta: float = 0.1, y0: float = 1e-7,
                         eps: float = 1e-9, finite: bool = True) -> int:
        f = CryptoMath.secure_fraction_gllp(
            qber=qber, fec=fec, mu=mu, nu=nu, eta=eta, y0=y0,
            n_sifted=sifted, eps=eps, finite=finite
        )
        return max(0, int(sifted * f))

    @staticmethod
    def poisson_safe(lam: float) -> int:
        """
        Robust Poisson sampler.
        NumPy's Poisson implementation raises for very large lambda, so use
        Gaussian approximation N(lam, lam) in that regime.
        """
        if not math.isfinite(lam) or lam <= 0:
            return 0
        lam = float(lam)
        if lam < 1e9:
            return int(np.random.poisson(lam))
        sample = int(round(np.random.normal(lam, math.sqrt(lam))))
        return max(0, sample)

    @staticmethod
    def combine_qber_xor(qbers: List[float]) -> Tuple[float, List[dict]]:
        """
        End-to-end QBER for XOR relay hops:
          q_total = q1 (+) q2 (+) ... with binary convolution
          q_new = q_prev + q_i - 2*q_prev*q_i
        Returns final QBER and per-hop cumulative steps.
        """
        if not qbers:
            return 0.0, []

        steps = []
        eq = max(0.0, min(0.5, qbers[0]))
        steps.append(dict(hop=0, input_qber=eq, cumulative_qber=eq))

        for i in range(1, len(qbers)):
            qi = max(0.0, min(0.5, qbers[i]))
            eq = eq + qi - 2.0 * eq * qi
            eq = max(0.0, min(0.5, eq))
            steps.append(dict(hop=i, input_qber=qi, cumulative_qber=eq))

        return eq, steps

    @staticmethod
    def chsh_werner(fidelity: float) -> float:
        return max(0.0, min(2 * math.sqrt(2), 2 * math.sqrt(2) * (2 * fidelity - 1)))


class DecoyGLLP:
    """
    Three-intensity decoy-state GLLP helper for WCP BB84.
    Uses signal (mu), weak decoy (nu), and vacuum (0) bounds.
    """

    def __init__(self, eta: float, y0: float, e_det: float, f_ec: float,
                 mu: float = 0.5, nu: float = 0.1):
        self.eta = max(1e-30, min(1.0, float(eta)))
        self.y0 = max(0.0, min(1.0, float(y0)))
        self.e_det = max(0.0, min(0.5, float(e_det)))
        self.f_ec = max(1.0, float(f_ec))
        self.mu = max(1e-6, float(mu))
        self.nu = max(1e-6, min(float(nu), 0.999 * self.mu))

    def gain(self, intensity: float) -> float:
        x = max(0.0, float(intensity))
        return max(1e-15, self.y0 + 1.0 - math.exp(-self.eta * x))

    def error_gain(self, intensity: float) -> float:
        q = self.gain(intensity)
        signal = max(0.0, 1.0 - math.exp(-self.eta * max(0.0, float(intensity))))
        num = 0.5 * self.y0 + self.e_det * signal
        return max(0.0, min(0.5, num / max(q, 1e-15)))

    def bounds(self) -> Tuple[float, float, float, float, float]:
        """Return (Q_mu, E_mu, Y1_lower, Q1_lower, e1_upper)."""
        mu, nu = self.mu, self.nu
        q_mu = self.gain(mu)
        q_nu = self.gain(nu)
        e_mu = self.error_gain(mu)
        e_nu = self.error_gain(nu)

        denom = mu * nu - nu * nu
        if denom <= 0:
            return q_mu, e_mu, 0.0, 0.0, 0.5

        y1_lower = (mu / denom) * (
            q_nu * math.exp(nu)
            - (nu * nu / (mu * mu)) * q_mu * math.exp(mu)
            - ((mu * mu - nu * nu) / (mu * mu)) * self.y0
        )
        y1_lower = max(0.0, min(1.0, y1_lower))
        q1_lower = max(0.0, min(q_mu, y1_lower * mu * math.exp(-mu)))

        if y1_lower <= 0:
            e1_upper = 0.5
        else:
            e1_upper = (e_nu * q_nu * math.exp(nu) - 0.5 * self.y0) / max(y1_lower * nu, 1e-15)
            e1_upper = max(0.0, min(0.5, e1_upper))

        return q_mu, e_mu, y1_lower, q1_lower, e1_upper

    def secure_fraction(self) -> float:
        """
        Fraction of sifted bits that remain after GLLP privacy amplification:
          frac = (Q1_lower/Q_mu)*(1-H(e1_upper)) - fEC*H(E_mu)
        """
        q_mu, e_mu, _, q1_lower, e1_upper = self.bounds()
        frac = (q1_lower / max(q_mu, 1e-15)) * (1.0 - CryptoMath.h(e1_upper))
        frac -= self.f_ec * CryptoMath.h(e_mu)
        return max(0.0, frac)

    @staticmethod
    def optimise_mu_nu(eta: float, y0: float, e_det: float, f_ec: float,
                       mu_seed: float = 0.5, nu_seed: float = 0.1) -> Tuple[float, float, float]:
        """
        Coarse-to-fine grid search for (mu, nu) maximizing GLLP secure fraction.
        Returns (best_mu, best_nu, best_fraction).
        """
        best_mu = max(0.05, float(mu_seed))
        best_nu = max(0.005, min(float(nu_seed), 0.5 * best_mu))
        best_frac = 0.0

        for mu in np.linspace(0.10, 1.20, 28):
            for nu in np.linspace(0.01, 0.5 * mu, 14):
                frac = DecoyGLLP(eta, y0, e_det, f_ec, mu, nu).secure_fraction()
                if frac > best_frac:
                    best_frac, best_mu, best_nu = frac, float(mu), float(nu)

        mu_lo = max(0.05, best_mu - 0.12)
        mu_hi = min(1.50, best_mu + 0.12)
        for mu in np.linspace(mu_lo, mu_hi, 16):
            nu_lo = max(0.005, best_nu - 0.06)
            nu_hi = min(0.5 * mu, best_nu + 0.06)
            if nu_hi <= nu_lo:
                continue
            for nu in np.linspace(nu_lo, nu_hi, 10):
                frac = DecoyGLLP(eta, y0, e_det, f_ec, mu, nu).secure_fraction()
                if frac > best_frac:
                    best_frac, best_mu, best_nu = frac, float(mu), float(nu)

        return best_mu, best_nu, best_frac


def compute_secure_bits(cfg: SimConfig, protocol: str, sifted: int, qber: float,
                        context: Optional[dict] = None) -> int:
    return compute_secure_bits_model(cfg, protocol, sifted, qber, cfg.securitymodel, context)


def compute_secure_bits_model(cfg: SimConfig, protocol: str, sifted: int, qber: float,
                              model: str, context: Optional[dict] = None) -> int:
    context = context or {}
    m = model.lower()
    if m == "gllp" and protocol.upper() == "BB84":
        if cfg.channeltype == ChannelType.FIBER:
            det_eff = cfg.fiberphysics.efficiency
            dark_hz = cfg.fiberphysics.darkcount
        else:
            det_eff = cfg.fsophysics.efficiency
            dark_hz = cfg.fsophysics.darkcount

        trans = float(context.get("transmission", 1.0))
        eta = max(1e-12, min(1.0, trans * det_eff))
        dark_hz = float(context.get("darkcount_hz", dark_hz))
        y0 = CryptoMath.dark_yield_per_pulse(dark_hz, cfg.gllp_clock_hz, detectors=2)
        e_det = max(0.0, min(0.5, float(qber)))

        if cfg.gllp_optimize:
            mu, nu, frac = DecoyGLLP.optimise_mu_nu(
                eta=eta, y0=y0, e_det=e_det, f_ec=cfg.ecefficiency,
                mu_seed=cfg.gllp_mu, nu_seed=cfg.gllp_nu
            )
        else:
            mu = max(1e-6, float(cfg.gllp_mu))
            nu = max(1e-6, min(float(cfg.gllp_nu), 0.999 * mu))
            frac = DecoyGLLP(
                eta=eta, y0=y0, e_det=e_det, f_ec=cfg.ecefficiency, mu=mu, nu=nu
            ).secure_fraction()

        if cfg.usefinitesize and sifted > 0:
            frac -= CryptoMath.finite_correction(sifted, cfg.epsilon)
            frac = max(0.0, frac)

        return max(0, int(sifted * frac))
    return CryptoMath.secure_bits(
        sifted, qber, cfg.ecefficiency, cfg.epsilon, cfg.usefinitesize
    )


# ============================================================================
# BB84 -- REAL SeQUeNCe SIMULATION
# ============================================================================

class BB84SimSequence:
    """BB84 using the real SeQUeNCe kernel."""

    def __init__(self, cfg: SimConfig):
        if not SEQUENCE_AVAILABLE:
            raise ImportError("SeQUeNCe not installed. Run: pip install sequence")
        self.cfg = cfg
        self.results: List[dict] = []

    def _run_link(self, a: int, b: int) -> dict:
        d = self.cfg.getlinkdistance(a, b)
        tl = Timeline(self.cfg.duration * SECOND)
        seed = int(time.time() * 1000) % 10000
        alice = QKDNode(f"node_{a}", tl, seed=a + seed)
        bob   = QKDNode(f"node_{b}", tl, seed=b + seed + 1)

        att, fid = ChannelModels.get_sequence_params(d, self.cfg)
        dm = d * 1000.0

        qc_ab = QuantumChannel(f"qc_{a}_{b}", tl, attenuation=att,
                               distance=dm, polarization_fidelity=fid)
        qc_ab.set_ends(alice, f"node_{b}")
        qc_ba = QuantumChannel(f"qc_{b}_{a}", tl, attenuation=att,
                               distance=dm, polarization_fidelity=fid)
        qc_ba.set_ends(bob, f"node_{a}")
        cc_ab = ClassicalChannel(f"cc_{a}_{b}", tl, distance=dm)
        cc_ab.set_ends(alice, f"node_{b}")
        cc_ba = ClassicalChannel(f"cc_{b}_{a}", tl, distance=dm)
        cc_ba.set_ends(bob, f"node_{a}")

        ap = alice.protocol_stack[0]
        bp = bob.protocol_stack[0]
        pair_bb84_protocols(ap, bp)
        ap.push(self.cfg.keylength, self.cfg.keycount, self.cfg.duration * SECOND)

        tl.init()
        tl.run()

        thru = ap.throughputs
        errs = ap.error_rates
        nkeys = len(thru)
        totbits = nkeys * self.cfg.keylength
        avg_qber = (sum(errs) / len(errs)) if errs else ((1 - fid) / 2)
        trans = ChannelModels.get_transmission(d, self.cfg)
        sec = compute_secure_bits(
            self.cfg, "BB84", totbits, avg_qber, context={"transmission": trans}
        )

        return dict(
            link=(a, b), alice=f"N{a}", bob=f"N{b}", success=True,
            distkm=d, channeltype=self.cfg.channeltype.value,
            transmission=trans, polarization_fidelity=fid,
            keys_generated=nkeys, siftedlength=totbits,
            securelength=sec, qber=avg_qber,
            siftedrate=totbits / self.cfg.duration,
            secretrate=sec / self.cfg.duration,
            issecure=avg_qber < 0.11 and sec > 0,
            sequence_used=True,
        )

    def runsim(self) -> List[dict]:
        links = getlinklist(self.cfg.topology, self.cfg.nodecount)
        results = []
        for a, b in links:
            try:
                r = self._run_link(a, b)
            except Exception as e:
                r = dict(link=(a, b), alice=f"N{a}", bob=f"N{b}",
                         success=False, error=str(e),
                         traceback=traceback.format_exc(),
                         distkm=self.cfg.getlinkdistance(a, b),
                         sequence_used=True)
            results.append(r)
        self.results = results
        return results


# ============================================================================
# E91 -- ANALYTICAL (uses SeQUeNCe channel parameters)
# ============================================================================

class E91Sim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.results: List[dict] = []

    def runsim(self) -> List[dict]:
        links = getlinklist(self.cfg.topology, self.cfg.nodecount)
        results = []
        for a, b in links:
            d = self.cfg.getlinkdistance(a, b)
            _, fid = ChannelModels.get_sequence_params(d, self.cfg)
            trans = ChannelModels.get_transmission(d, self.cfg)
            pair_trans = trans ** 2
            base_fid = 0.98
            ent_fid = max(0.5, min(0.99, base_fid * fid))
            n_pairs = self.cfg.keylength * self.cfg.keycount
            exp_pairs = n_pairs * pair_trans
            rec = CryptoMath.poisson_safe(exp_pairs)

            if rec < 10:
                results.append(dict(link=(a, b), node1=f"N{a}", node2=f"N{b}",
                                    success=False, error=f"Too few pairs ({rec})",
                                    distkm=d, sequence_used=False))
                continue

            sifted = max(1, CryptoMath.poisson_safe(rec * 0.5))
            qber = max(0, min(0.5, (1 - ent_fid) / 2 + random.uniform(-0.005, 0.005)))
            chsh = CryptoMath.chsh_werner(ent_fid)
            sec = compute_secure_bits(self.cfg, "E91", sifted, qber)
            ok = chsh > 2.0 and qber < 0.11 and sec > 0
            results.append(dict(
                link=(a, b), node1=f"N{a}", node2=f"N{b}", success=True,
                distkm=d, channeltype=self.cfg.channeltype.value,
                transmission=trans, pairsgen=n_pairs, pairsrec=rec,
                siftedlength=sifted, securelength=sec,
                qber=qber, entfidelity=ent_fid,
                chshval=chsh, chshviolated=chsh > 2.0, issecure=ok,
                siftedrate=sifted / self.cfg.duration,
                secretrate=sec / self.cfg.duration,
                sequence_used=False,
            ))
        self.results = results
        return results


# ============================================================================
# TF-QKD -- ANALYTICAL  (Appendix B.1 of arXiv:2507.23466)
# ============================================================================

class TwinFieldQKD:
    """
    Sending-or-not-sending TF-QKD  [Wang, Yu, Hu -- PRA 98, 062323 (2018)]
    Equations B1-B6 of arXiv:2507.23466.
    Charlie is an UNTRUSTED measurement node.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.results: dict = {}

    def runsim(self, alice: int = 0, charlie: int = 1, bob: int = 2) -> dict:
        d_ac = self.cfg.getlinkdistance(alice, charlie)
        d_cb = self.cfg.getlinkdistance(charlie, bob)
        d_total = d_ac + d_cb

        # Channel transmittances
        if self.cfg.channeltype == ChannelType.FSO_SATELLITE:
            trans_ac = ChannelModels.satellite_link_transmission(
                d_ac, self.cfg.satelliteconfig, self.cfg.atmosphere)
            trans_cb = ChannelModels.satellite_link_transmission(
                d_cb, self.cfg.satelliteconfig, self.cfg.atmosphere)
        else:
            trans_ac = ChannelModels.get_transmission(d_ac, self.cfg)
            trans_cb = ChannelModels.get_transmission(d_cb, self.cfg)

        pd  = self.cfg.tfqkd_darkcount
        eta_d = self.cfg.tfqkd_detector_eff
        F   = self.cfg.tfqkd_repetition_rate
        fec = self.cfg.tfqkd_fec
        ed  = self.cfg.tfqkd_misalignment

        # Total efficiencies  eta__A = tau__A * eta__D ,  eta__B = tau__B * eta__D   (Eq. B1)
        eta_A = trans_ac * eta_d
        eta_B = trans_cb * eta_d

        # Optimise u
        mu_opt = self._optimise_mu(eta_A, eta_B, pd, ed, fec)

        # Effective intensities gamma_  (Eq. B1)
        gamma_A = mu_opt * eta_A
        gamma_B = mu_opt * eta_B

        # Gains & error rates
        p_XX  = self._p_xx(gamma_A, gamma_B, pd)
        e_XX  = self._e_xx(gamma_A, gamma_B, pd, ed)
        e_ZZ  = self._e_zz(gamma_A, gamma_B, pd, ed)

        # Secret key rate per pulse  R = 2*p_XX*[1 - f_EC*H(e_XX) - H(e_ZZ)]  (Eq. B6)
        H_eX = CryptoMath.h(e_XX)
        H_eZ = CryptoMath.h(e_ZZ)
        R_pulse = max(0.0, 2.0 * p_XX * (1.0 - fec * H_eX - H_eZ))

        # Key generation over simulation window
        total_pulses = int(F * self.cfg.duration)
        expected_clicks_raw = total_pulses * p_XX
        raw_click_rate = expected_clicks_raw / max(self.cfg.duration, 1e-12)
        deadtime_s = max(0.0, self.cfg.tfqkd_detector_deadtime_ns * 1e-9)
        detector_count = max(1, int(self.cfg.tfqkd_detector_count))
        detector_model = str(self.cfg.tfqkd_saturation_model).strip().lower()
        sat_click_rate = self._apply_detector_saturation(
            raw_click_rate, deadtime_s, detector_count, detector_model
        )
        expected_clicks = sat_click_rate * self.cfg.duration
        actual_clicks = CryptoMath.poisson_safe(expected_clicks)
        sifted = actual_clicks
        secure = int(sifted * R_pulse / p_XX) if (R_pulse > 0 and p_XX > 0) else 0
        theoretical_rate_raw = R_pulse * F
        theoretical_rate_sat = sat_click_rate * (R_pulse / p_XX) if p_XX > 0 else 0.0

        if self.cfg.usefinitesize and sifted > 0:
            delta = CryptoMath.finite_correction(sifted, self.cfg.epsilon)
            secure = max(0, int(secure * (1.0 - delta)))

        ok = e_XX < 0.11 and e_ZZ < 0.5 and secure > 0

        self.results = dict(
            success=True, protocol="TF-QKD",
            alice=alice, charlie=charlie, bob=bob,
            charlietrust="UNTRUSTED",
            distac=d_ac, distcb=d_cb, totaldist=d_total,
            channeltype=self.cfg.channeltype.value,
            transac=trans_ac, transcb=trans_cb,
            eta_A=eta_A, eta_B=eta_B,
            mu_optimal=mu_opt, gamma_A=gamma_A, gamma_B=gamma_B,
            p_XX=p_XX, e_XX=e_XX, e_ZZ=e_ZZ,
            R_per_pulse=R_pulse,
            detector_model=detector_model,
            detector_count=detector_count,
            detector_deadtime_ns=self.cfg.tfqkd_detector_deadtime_ns,
            click_rate_raw=raw_click_rate,
            click_rate_sat=sat_click_rate,
            expected_clicks_raw=expected_clicks_raw,
            expected_clicks_sat=expected_clicks,
            totalpulses=total_pulses, clickcount=actual_clicks,
            siftedlength=sifted, securelength=secure,
            qber=e_XX,
            secretrate=secure / self.cfg.duration,
            theoretical_rate=theoretical_rate_sat,
            theoretical_rate_raw=theoretical_rate_raw,
            theoretical_rate_sat=theoretical_rate_sat,
            issecure=ok,
            sequence_used=False,
            note="Analytical (SeQUeNCe doesn't support TF-QKD)",
        )
        return self.results

    @staticmethod
    def _apply_detector_saturation(rate_hz: float, deadtime_s: float, detectors: int,
                                   model: str = "nonparalyzable") -> float:
        """
        Apply finite detector recovery constraints to aggregate click rate.
        """
        if rate_hz <= 0:
            return 0.0
        if deadtime_s <= 0 or detectors <= 0 or model == "none":
            return rate_hz

        per_detector_rate = rate_hz / detectors
        if model == "paralyzable":
            observed_per_detector = per_detector_rate * math.exp(-per_detector_rate * deadtime_s)
        else:
            observed_per_detector = per_detector_rate / (1.0 + per_detector_rate * deadtime_s)
        return max(0.0, detectors * observed_per_detector)

    # --- internal helpers (Eqs B2, B3, B4-B6) ---

    def _optimise_mu(self, eta_A, eta_B, pd, ed, fec) -> float:
        best_mu, best_R = 0.04, 0.0
        for mu in np.linspace(0.005, 1.5, 200):
            gA, gB = mu * eta_A, mu * eta_B
            pxx = self._p_xx(gA, gB, pd)
            exx = self._e_xx(gA, gB, pd, ed)
            ezz = self._e_zz(gA, gB, pd, ed)
            if exx >= 0.5 or ezz >= 0.5:
                continue
            R = 2.0 * pxx * (1.0 - fec * CryptoMath.h(exx) - CryptoMath.h(ezz))
            if R > best_R:
                best_R, best_mu = R, mu
        return best_mu

    @staticmethod
    def _p_xx(gA, gB, pd) -> float:
        """X-basis gain  (Eq. B2, theta_=phi=0 for ideal alignment)."""
        sg = math.sqrt(gA * gB)
        val = 0.5 * (1 - pd) * (math.exp(-sg) + math.exp(sg)) * math.exp(-0.5 * (gA + gB))
        val -= (1 - pd) ** 2 * math.exp(-(gA + gB))
        val += pd * (1 - pd) * math.exp(-0.5 * (gA + gB))
        return max(0.0, min(1.0, val))

    @staticmethod
    def _e_xx(gA, gB, pd, ed) -> float:
        """X-basis QBER  (Eq. B3, theta_=phi=0)."""
        sg = math.sqrt(gA * gB)
        num = math.exp(-sg) - (1 - pd) * math.exp(-0.5 * (gA + gB))
        den = math.exp(-sg) + math.exp(sg) - 2 * (1 - pd) * math.exp(-0.5 * (gA + gB))
        if den <= 0:
            return 0.5
        return max(0.0, min(0.5, abs(num / den) + ed))

    @staticmethod
    def _e_zz(gA, gB, pd, ed) -> float:
        """Z-basis phase error (decoy-state upper bound, Eq. B5 style)."""
        asym = abs(gA - gB) / max(gA + gB, 1e-12)
        base = 0.15 + 0.10 * asym
        dark = min(0.2, pd / max(gA + gB, 1e-12))
        return min(0.49, base + dark + ed)


# ============================================================================
# TRUSTED RELAY NETWORK  (XOR key relay)
# ============================================================================

class TrustedRelayNetwork:
    """
    For *linear* and *star* topologies the intermediate trusted nodes
    perform XOR-based key relay.  E2E rate = min(link rates).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.linkresults: List[dict] = []
        self.e2eresults: List[dict] = []

    def run_links(self) -> List[dict]:
        if self.cfg.protocol == "BB84" and SEQUENCE_AVAILABLE:
            sim = BB84SimSequence(self.cfg)
        else:
            sim = E91Sim(self.cfg)
        self.linkresults = sim.runsim()
        return self.linkresults

    def _find_path(self, src, dst) -> List[int]:
        t = self.cfg.topology
        if t == "linear":
            return list(range(src, dst + 1)) if src < dst else list(range(src, dst - 1, -1))
        if t == "star":
            if src == 0:
                return [0, dst]
            if dst == 0:
                return [src, 0]
            return [src, 0, dst]
        g = nx.Graph()
        for r in self.linkresults:
            if r.get("success"):
                g.add_edge(*r["link"])
        try:
            return nx.shortest_path(g, src, dst)
        except Exception:
            return []

    def _link_metric(self, a, b, key):
        for r in self.linkresults:
            if r.get("success") and set(r["link"]) == {a, b}:
                return r.get(key, 0)
        return 0

    def calc_e2e(self, src, dst) -> dict:
        path = self._find_path(src, dst)
        if len(path) < 2:
            return dict(source=src, dest=dst, success=False, error="No path")

        secret_rates, sifted_rates, qbers, transmits, details = [], [], [], [], []
        total_d = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            r_secret = self._link_metric(a, b, "secretrate")
            r_sifted = self._link_metric(a, b, "siftedrate")
            # Backward-compatible fallback for links that don't expose sifted rate.
            if r_sifted <= 0:
                r_sifted = r_secret
            q = self._link_metric(a, b, "qber")
            tr = self._link_metric(a, b, "transmission")
            d = self.cfg.getlinkdistance(a, b)
            secret_rates.append(r_secret)
            sifted_rates.append(r_sifted)
            qbers.append(q)
            transmits.append(tr)
            details.append(dict(
                link=(a, b), secretrate=r_secret, siftedrate=r_sifted,
                qber=q, distance=d, transmission=tr
            ))
            total_d += d

        untrusted = [path[i] for i in range(1, len(path) - 1)
                     if self.cfg.getnodtrust(path[i]) == NodeTrust.UNTRUSTED]

        # Detailed XOR QBER propagation across all hops.
        eq, qber_steps = CryptoMath.combine_qber_xor(qbers)

        if all(r > 0 for r in sifted_rates):
            bn = sifted_rates.index(min(sifted_rates))
            bottleneck_sifted_rate = min(sifted_rates)
            ok = True
        else:
            bottleneck_sifted_rate, bn, ok = 0.0, -1, False

        e2e_sifted_bits = int(bottleneck_sifted_rate * self.cfg.duration)
        bottleneck_trans = transmits[bn] if bn >= 0 else 1.0
        e2e_sec = compute_secure_bits(
            self.cfg, self.cfg.protocol, e2e_sifted_bits, eq,
            context={"transmission": bottleneck_trans}
        )
        e2e_rate = e2e_sec / self.cfg.duration if self.cfg.duration > 0 else 0.0
        bottleneck_secret_rate = secret_rates[bn] if bn >= 0 else 0.0

        return dict(
            source=src, dest=dst, path=path,
            hops=len(path) - 1, totaldistance=total_d,
            linkdetails=details, linkrates=secret_rates,
            linksiftedrates=sifted_rates, linkqbers=qbers,
            bottlenecklink=bn,
            bottleneckrate=bottleneck_secret_rate,
            bottlenecksiftedrate=bottleneck_sifted_rate,
            e2erate=e2e_rate, e2eqber=eq, e2esecbits=e2e_sec,
            e2ebits=e2e_sec, e2esiftedbits=e2e_sifted_bits,
            qbersteps=qber_steps,
            untrusted=untrusted,
            warning="Path has untrusted nodes -- use TF-QKD!" if untrusted else None,
            success=ok and not untrusted,
            issecure=eq < 0.11 and e2e_sec > 0 and not untrusted,
        )

    def run_all_e2e(self) -> List[dict]:
        self.run_links()
        out = []
        for i in range(self.cfg.nodecount):
            for j in range(i + 1, self.cfg.nodecount):
                out.append(self.calc_e2e(i, j))
        self.e2eresults = out
        return out


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def _safe_save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   Saved: {path}")


def drawnetwork(results, cfg, title_extra="", e2eresults=None, savepath="network.png"):
    g = nx.Graph()
    g.add_nodes_from(range(cfg.nodecount))
    edge_colors, edge_labels = [], {}

    for r in results:
        if not r.get("success"):
            continue
        u, v = r["link"]
        rate = r.get("secretrate", 0)
        q = r.get("qber", 0)
        g.add_edge(u, v, weight=rate)
        edge_labels[(u, v)] = f"{rate:.0f} bps\nQBER:{q*100:.1f}%"
        if q >= 0.11 or not r.get("issecure"):
            edge_colors.append("red")
        elif q > 0.05:
            edge_colors.append("orange")
        else:
            edge_colors.append("green")

    # layout
    if cfg.topology == "linear":
        pos = {i: (i * 3, 0) for i in range(cfg.nodecount)}
    elif cfg.topology == "star":
        pos = {0: (0, 0)}
        step = 2 * math.pi / max(1, cfg.nodecount - 1)
        for i in range(1, cfg.nodecount):
            a = (i - 1) * step - math.pi / 2
            pos[i] = (2.5 * math.cos(a), 2.5 * math.sin(a))
    else:
        pos = nx.spring_layout(g, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))

    # multi-hop arcs
    if e2eresults:
        drawn = set()
        for e in e2eresults:
            if e.get("hops", 0) <= 1:
                continue
            s, d = e["source"], e["dest"]
            pk = tuple(sorted([s, d]))
            if pk in drawn:
                continue
            drawn.add(pk)
            x1, y1 = pos[s]; x2, y2 = pos[d]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            ln = math.sqrt(dx ** 2 + dy ** 2) or 1
            px, py = -dy / ln, dx / ln
            off = 0.8 + (e.get("hops", 2) - 2) * 0.4 if cfg.topology == "linear" else 1.8
            cx, cy = mx + px * off, my + py * off
            has_u = bool(e.get("untrusted"))
            eqber = e.get("e2eqber", 0)
            if has_u or not e.get("success"):
                pc, bg = "red", "mistyrose"
            elif eqber >= 0.11:
                pc, bg = "red", "mistyrose"
            elif eqber > 0.05:
                pc, bg = "orange", "moccasin"
            else:
                pc, bg = "purple", "lavender"
            verts = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            pp = mpatches.PathPatch(Path(verts, codes), facecolor="none",
                                    edgecolor=pc, linestyle=":", linewidth=2.5, alpha=0.7,
                                    zorder=5)
            ax.add_patch(pp)
            # place label at arc midpoint (t=0.5 on quadratic Bezier)
            lx = 0.25 * x1 + 0.5 * cx + 0.25 * x2
            ly = 0.25 * y1 + 0.5 * cy + 0.25 * y2
            if e.get("success") and not has_u:
                lbl = (f"N{s}<->N{d} (E2E)\n{e.get('e2erate',0):.0f} bps\n"
                       f"QBER:{eqber*100:.1f}%\n({e['hops']} hops)")
            else:
                lbl = f"N{s}<->N{d} (E2E)\nFAILED\n({e.get('hops',0)} hops)"
            ax.annotate(lbl, xy=(lx, ly), fontsize=9, color=pc, ha="center",
                        va="center", bbox=dict(boxstyle="round,pad=0.3",
                        facecolor=bg, edgecolor=pc, alpha=0.9),
                        zorder=10)

    # nodes
    nc = ["salmon" if cfg.getnodtrust(i) == NodeTrust.UNTRUSTED else "lightblue"
          for i in range(cfg.nodecount)]
    nx.draw_networkx_nodes(g, pos, node_color=nc, node_size=1500,
                           edgecolors="darkblue", linewidths=2, ax=ax)
    nl = {i: f"N{i}{'*' if cfg.getnodtrust(i)==NodeTrust.UNTRUSTED else ''}"
          for i in range(cfg.nodecount)}
    nx.draw_networkx_labels(g, pos, nl, font_size=14, font_weight="bold", ax=ax)
    if edge_colors:
        nx.draw_networkx_edges(g, pos, edgelist=list(g.edges()), width=4,
                               edge_color=edge_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels, font_size=9,
                                  bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor="white", edgecolor="gray",
                                            alpha=0.9), ax=ax)

    seq_flag = "SeQUeNCe" if cfg.protocol == "BB84" and SEQUENCE_AVAILABLE else "Analytical"
    ax.set_title(f"QKD Network: {cfg.protocol} on {cfg.topology.upper()} "
                 f"({cfg.channeltype.value.upper()})\n{title_extra} [{seq_flag}]",
                 fontsize=12, fontweight="bold")
    leg = [Patch(facecolor="green", label="Secure (QBER<5%)"),
           Patch(facecolor="orange", label="Marginal (5-11%)"),
           Patch(facecolor="red", label="Insecure (>=11%)"),
           Patch(facecolor="lightblue", edgecolor="darkblue", label="Trusted Node"),
           Patch(facecolor="salmon", edgecolor="darkblue", label="Untrusted Node"),
           Line2D([0], [0], color="green", lw=4, label="Direct Link"),
           Line2D([0], [0], color="purple", lw=2.5, ls=":", label="Multi-Hop E2E")]
    ax.legend(handles=leg, loc="upper left", fontsize=9)
    ax.margins(0.15)
    ax.axis("off")
    fig.tight_layout()
    _safe_save(fig, savepath)


def drawqber(results, cfg, e2eresults=None, savepath="qber_bar.png"):
    labels, dists, qbers, types = [], [], [], []
    for r in results:
        if not r.get("success"):
            continue
        labels.append(f"N{r['link'][0]}<->N{r['link'][1]}")
        dists.append(r.get("distkm", 0))
        qbers.append(r.get("qber", 0) * 100)
        types.append("direct")
    if e2eresults:
        for e in e2eresults:
            if e.get("success") and e.get("hops", 0) > 1:
                labels.append(f"N{e['source']}<->N{e['dest']}\n(E2E, {e['hops']}hop)")
                dists.append(e.get("totaldistance", 0))
                qbers.append(e.get("e2eqber", 0) * 100)
                types.append("e2e")
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    colors = []
    for q, t in zip(qbers, types):
        if q >= 11:
            colors.append("red")
        elif q > 5:
            colors.append("orange")
        elif t == "e2e":
            colors.append("purple")
        else:
            colors.append("green")
    bars = ax.bar(x, qbers, color=colors, edgecolor="black")
    for bar, d, v, t in zip(bars, dists, qbers, types):
        lbl = f"{d:.1f}km"
        if t == "e2e":
            lbl += " (E2E)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                lbl, ha="center", va="bottom", fontsize=8)
    ax.axhline(y=11, color="red", ls="--", lw=2, label="Security Threshold (11%)")
    ax.set_ylabel("QBER (%)")
    ax.set_xlabel("Link / Path")
    ax.set_title(f"QBER: Direct Links + E2E Paths ({cfg.channeltype.value.upper()} Channel)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(handles=[Patch(facecolor="green", label="Direct (QBER<5%)"),
                       Patch(facecolor="purple", label="E2E Multi-Hop"),
                       Patch(facecolor="orange", label="Marginal (5-11%)"),
                       Patch(facecolor="red", label="Insecure (>=11%)")],
              loc="upper right")
    fig.tight_layout()
    _safe_save(fig, savepath)


def drawrates(results, cfg, e2eresults=None, savepath="rates.png"):
    labels, sifted, secret, secure_flags, types = [], [], [], [], []
    for r in results:
        if not r.get("success"):
            continue
        labels.append(f"N{r['link'][0]}<->N{r['link'][1]}")
        sifted.append(r.get("siftedrate", 0))
        secret.append(r.get("secretrate", 0))
        secure_flags.append(r.get("issecure", False))
        types.append("direct")
    if e2eresults:
        for e in e2eresults:
            if e.get("success") and e.get("hops", 0) > 1:
                labels.append(f"N{e['source']}<->N{e['dest']}\n(E2E, {e['hops']}hop)")
                sifted.append(e.get("e2erate", 0))
                secret.append(e.get("e2esecbits", 0) / cfg.duration)
                secure_flags.append(e.get("issecure", False))
                types.append("e2e")
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    w = 0.35
    sc = ["lightblue" if t == "direct" else "plum" for t in types]
    rc = ["green" if s else "red" if t == "direct" else "purple" if s else "red"
          for s, t in zip(secure_flags, types)]
    ax.bar(x - w / 2, sifted, w, label="Sifted Rate", color=sc, edgecolor="blue")
    ax.bar(x + w / 2, secret, w, label="Secret Rate", color=rc, edgecolor="black")
    ax.set_ylabel("Rate (bps)")
    ax.set_xlabel("Link / Path")
    ax.set_title(f"Key Generation Rates ({cfg.channeltype.value.upper()} Channel)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    _safe_save(fig, savepath)


def drawscatter_qber(results, cfg, e2eresults=None, savepath="scatter_qber.png"):
    dd, qq, tt = [], [], []
    for r in results:
        if not r.get("success"):
            continue
        dd.append(r["distkm"])
        qq.append(r["qber"] * 100)
        tt.append("direct")
    if e2eresults:
        for e in e2eresults:
            if e.get("success") and e.get("hops", 0) > 1:
                dd.append(e["totaldistance"])
                qq.append(e["e2eqber"] * 100)
                tt.append("e2e")
    if len(dd) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for d_, q_, t_ in zip(dd, qq, tt):
        c = "green" if q_ < 5 else ("orange" if q_ < 11 else "red")
        m = "o" if t_ == "direct" else "s"
        ax.scatter(d_, q_, c=c, s=200, edgecolors="black", zorder=3, marker=m)
    if len(dd) >= 2:
        z = np.polyfit(dd, qq, 1)
        xl = np.linspace(min(dd) * 0.9, max(dd) * 1.1, 100)
        ax.plot(xl, np.poly1d(z)(xl), "b--", alpha=0.7,
                label=f"Trend: {z[0]:.3f}x + {z[1]:.3f}")
    ax.axhline(y=11, color="red", ls="--", lw=2, label="Security Threshold (11%)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("QBER (%)")
    ax.set_title(f"QBER vs Distance ({cfg.channeltype.value.upper()} Channel)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _safe_save(fig, savepath)


def drawscatter_rate(results, cfg, e2eresults=None, savepath="scatter_rate.png"):
    dd, rr, ss, tt = [], [], [], []
    for r in results:
        if not r.get("success"):
            continue
        dd.append(r["distkm"])
        rr.append(r["secretrate"])
        ss.append(r["issecure"])
        tt.append("direct")
    if e2eresults:
        for e in e2eresults:
            if e.get("success") and e.get("hops", 0) > 1:
                dd.append(e["totaldistance"])
                rr.append(e.get("e2esecbits", 0) / cfg.duration)
                ss.append(e.get("issecure", False))
                tt.append("e2e")
    if len(dd) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for d_, r_, s_, t_ in zip(dd, rr, ss, tt):
        c = ("green" if s_ else "red") if t_ == "direct" else ("purple" if s_ else "red")
        m = "o" if t_ == "direct" else "s"
        ax.scatter(d_, r_, c=c, s=200, edgecolors="black", zorder=3, marker=m)
    if len(dd) >= 2 and max(rr) > 0:
        z = np.polyfit(dd, rr, 1)
        xl = np.linspace(min(dd) * 0.9, max(dd) * 1.1, 100)
        ax.plot(xl, np.poly1d(z)(xl), "b--", alpha=0.7,
                label=f"Trend: {z[0]:.1f}x + {z[1]:.1f}")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Secret Key Rate (bps)")
    ax.set_title(f"Secret Key Rate vs Distance ({cfg.channeltype.value.upper()} Channel)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _safe_save(fig, savepath)


def draw_chsh_formula_examples(results=None, savepath="chsh_formula_examples.png"):
    """
    Visual explanation for the E91 CHSH model used in this simulator.

    Left panel:
      S(F) = 2*sqrt(2)*(2F - 1)  where F is entanglement fidelity.
    Right panel:
      Equivalent Werner-parameter form S(p) = 2*sqrt(2)*p with p = 2F - 1.
    """
    s_max = 2.0 * math.sqrt(2.0)
    f_threshold = (1.0 + 1.0 / math.sqrt(2.0)) / 2.0
    p_threshold = 1.0 / math.sqrt(2.0)

    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: S vs Fidelity ----
    f_grid = np.linspace(0.5, 1.0, 400)
    s_grid = 2.0 * np.sqrt(2.0) * (2.0 * f_grid - 1.0)
    ax_f.plot(f_grid, s_grid, color="navy", linewidth=2.5, label="S(F) = 2*sqrt(2)*(2F-1)")
    ax_f.axhline(2.0, color="red", linestyle="--", linewidth=2, label="CHSH bound S=2")
    ax_f.axhline(s_max, color="gray", linestyle=":", linewidth=2, label="Tsirelson 2*sqrt(2)")
    ax_f.axvline(f_threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.8)

    f_examples = [0.60, f_threshold, 0.90]
    s_examples = [2.0 * math.sqrt(2.0) * (2.0 * f - 1.0) for f in f_examples]
    c_examples = ["darkred", "darkorange", "green"]
    t_examples = ["No CHSH violation", "At CHSH threshold", "Clear CHSH violation"]
    ax_f.scatter(f_examples, s_examples, c=c_examples, s=90, edgecolors="black", zorder=4)
    for f, s, txt in zip(f_examples, s_examples, t_examples):
        ax_f.annotate(
            f"F={f:.2f}, S={s:.2f}\n{txt}",
            xy=(f, s),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    sim_f = []
    sim_s = []
    if results:
        for r in results:
            if r.get("success") and "entfidelity" in r and "chshval" in r:
                sim_f.append(float(r["entfidelity"]))
                sim_s.append(float(r["chshval"]))
    if sim_f:
        ax_f.scatter(sim_f, sim_s, marker="x", c="black", s=70, label="Simulation points")

    ax_f.set_xlabel("Entanglement Fidelity F")
    ax_f.set_ylabel("CHSH Value S")
    ax_f.set_title("From Fidelity to CHSH")
    ax_f.set_xlim(0.5, 1.0)
    ax_f.set_ylim(0.0, s_max * 1.05)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(fontsize=8, loc="lower right")

    # ---- Right: S vs Werner p ----
    p_grid = np.linspace(0.0, 1.0, 400)
    s_p = 2.0 * np.sqrt(2.0) * p_grid
    ax_p.plot(p_grid, s_p, color="purple", linewidth=2.5, label="S(p) = 2*sqrt(2)*p")
    ax_p.axhline(2.0, color="red", linestyle="--", linewidth=2, label="CHSH bound S=2")
    ax_p.axvline(1.0 / 3.0, color="gray", linestyle="--", linewidth=1.5, label="p=1/3 (separable)")
    ax_p.axvline(p_threshold, color="red", linestyle="--", linewidth=1.5, label="p=1/sqrt(2)")

    ax_p.axvspan(0.0, 1.0 / 3.0, color="lightgray", alpha=0.35)
    ax_p.axvspan(1.0 / 3.0, p_threshold, color="gold", alpha=0.25)
    ax_p.axvspan(p_threshold, 1.0, color="lightgreen", alpha=0.25)

    p_examples = [0.50, p_threshold, 0.85]
    s_p_examples = [2.0 * math.sqrt(2.0) * p for p in p_examples]
    ax_p.scatter(p_examples, s_p_examples, c=["darkorange", "blue", "green"],
                 s=90, edgecolors="black", zorder=4)
    for p, s in zip(p_examples, s_p_examples):
        ax_p.annotate(
            f"p={p:.3f}, S={s:.2f}",
            xy=(p, s),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    ax_p.text(0.02, 0.20, "p <= 1/3: separable", transform=ax_p.transAxes, fontsize=9)
    ax_p.text(0.34, 0.20, "1/3 < p < 1/sqrt(2): entangled\nbut CHSH may not detect",
              transform=ax_p.transAxes, fontsize=9)
    ax_p.text(0.72, 0.20, "p > 1/sqrt(2): CHSH violation", transform=ax_p.transAxes, fontsize=9)

    ax_p.set_xlabel("Werner Parameter p")
    ax_p.set_ylabel("CHSH Value S")
    ax_p.set_title("Equivalent Werner-State View")
    ax_p.set_xlim(0.0, 1.0)
    ax_p.set_ylim(0.0, s_max * 1.05)
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "CHSH Formula Working Example for E91: S = 2*sqrt(2)*(2F-1)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _safe_save(fig, savepath)


def drawe2e_rates(e2eresults, cfg, savepath="e2e_rates.png"):
    succ = [r for r in e2eresults if r.get("success")]
    if not succ:
        return
    labels = [f"N{r['source']}<->N{r['dest']} (E2E)" for r in succ]
    rates  = [r.get("e2erate", 0) for r in succ]
    qbers  = [r.get("e2eqber", 0) * 100 for r in succ]
    hops   = [r.get("hops", 0) for r in succ]
    dists  = [r.get("totaldistance", 0) for r in succ]
    secure = [r.get("issecure", False) for r in succ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(labels))

    c1 = ["green" if s else "red" for s in secure]
    bars1 = ax1.bar(x, rates, color=c1, edgecolor="black")
    for b, h, d in zip(bars1, hops, dists):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 10,
                 f"{h} hops\n{d:.1f}km", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("E2E Secret Rate (bps)")
    ax1.set_xlabel("Node Pair")
    ax1.set_title("End-to-End Key Rates")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)

    c2 = ["green" if q < 5 else "orange" if q < 11 else "red" for q in qbers]
    bars2 = ax2.bar(x, qbers, color=c2, edgecolor="black")
    ax2.axhline(y=11, color="red", ls="--", lw=2, label="Threshold (11%)")
    for b, h in zip(bars2, hops):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 f"{h} hops", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("E2E QBER (%)")
    ax2.set_xlabel("Node Pair")
    ax2.set_title("End-to-End QBER (XOR Combined)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    fig.suptitle("End-to-End Analysis (XOR Trusted Relay)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _safe_save(fig, savepath)


def draw_tfqkd(result, cfg, savepath="tfqkd_analysis.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1 -- channel parameters
    ax = axes[0, 0]
    names = ["eta__A", "eta__B", "gamma__A", "gamma__B"]
    vals  = [result["eta_A"] * 100, result["eta_B"] * 100,
             result["gamma_A"] * 1000, result["gamma_B"] * 1000]
    cols  = ["steelblue", "steelblue", "coral", "coral"]
    bars  = ax.bar(names, vals, color=cols, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Value (% or x10-3)")
    ax.set_title("TF-QKD Channel Parameters")

    # 2 -- error rates
    ax = axes[0, 1]
    enames = ["e_XX (X-basis)", "e_ZZ (Z-basis)"]
    evals  = [result["e_XX"] * 100, result["e_ZZ"] * 100]
    ecols  = ["green" if v < 11 else "red" for v in evals]
    bars   = ax.bar(enames, evals, color=ecols, edgecolor="black")
    ax.axhline(y=11, color="red", ls="--", lw=2, label="Security Threshold")
    for b, v in zip(bars, evals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("TF-QKD Error Rates")
    ax.legend()

    # 3 -- key generation
    ax = axes[1, 0]
    kn = ["Clicks", "Sifted", "Secure"]
    kv = [result["clickcount"], result["siftedlength"], result["securelength"]]
    kc = ["lightblue", "skyblue", "green" if result["issecure"] else "red"]
    bars = ax.bar(kn, kv, color=kc, edgecolor="black")
    for b, v in zip(bars, kv):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(kv) * 0.02,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Bits")
    ax.set_title("TF-QKD Key Generation")

    # 4 -- distances
    ax = axes[1, 1]
    ax.bar(["Alice->Charlie", "Charlie->Bob"],
           [result["distac"], result["distcb"]],
           color=["steelblue", "coral"], edgecolor="black")
    ax.set_ylabel("Distance (km)")
    ax.set_title(f"Link Distances (Total: {result['totaldist']:.1f} km)")

    status = "SECURE [OK]" if result["issecure"] else "INSECURE [X]"
    fig.suptitle(
        f"TF-QKD: N{result['alice']}--N{result['charlie']}*--N{result['bob']} | "
        f"{result['secretrate']:.1f} bps | {status}",
        fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    _safe_save(fig, savepath)


def draw_satellite_distance_analysis(cfg: SimConfig, savepath="satellite_distance_analysis.png"):
    """
    Generate QBER vs Distance and SKR vs Distance plots for satellite TF-QKD.
    Sweeps across a range of distances to show performance characteristics.
    """
    # Define distance range based on orbit type
    sat = cfg.satelliteconfig
    if sat.orbittype == "LEO":
        distances = np.linspace(500, 2000, 40)
    elif sat.orbittype == "MEO":
        distances = np.linspace(2000, 10000, 40)
    else:  # GEO
        distances = np.linspace(5000, 45000, 40)
    
    # Arrays to store results
    qber_alice_sat = []
    qber_bob_sat = []
    qber_tfqkd = []
    skr_alice_sat = []
    skr_bob_sat = []
    skr_tfqkd = []
    
    # Store original distances
    orig_dist_ac = cfg.linkdistances.get((0, 1), 0)
    orig_dist_cb = cfg.linkdistances.get((1, 2), 0)
    
    # Physics parameters
    lam = 1550e-9
    eta_det = cfg.tfqkd_detector_eff
    pd = cfg.tfqkd_darkcount
    ed = cfg.tfqkd_misalignment
    F = cfg.tfqkd_repetition_rate
    fec = cfg.tfqkd_fec
    
    for dist in distances:
        # Update distances symmetrically
        cfg.linkdistances[(0, 1)] = dist
        cfg.linkdistances[(1, 2)] = dist
        
        # Calculate deterministic transmission (no random fluctuation)
        L = dist * 1000.0  # meters
        elev_rad = math.radians(max(sat.elevation, 5.0))
        
        # Geometric transmission: tau_geom = (pi * D_OGS * D_sat / (4 * lambda * L))^2
        tau_geom = min(1.0, (math.pi * sat.DOGS * sat.Dsat / (4.0 * lam * L)) ** 2)
        
        # System loss (Table I: 2.8 dB)
        tau_syst = 10 ** (-sat.tau_syst_dB / 10.0)
        
        # Atmospheric absorption through slant path
        atmos_km = 20.0 / math.sin(elev_rad)
        vis = cfg.atmosphere.visibility
        wl_nm = lam * 1e9
        q = 1.6 if vis > 50 else (1.3 if vis > 6 else 0.585 * vis ** (1 / 3))
        alpha_db = (3.91 / vis) * (wl_nm / 550) ** (-q)
        tau_abs = 10 ** (-alpha_db * min(atmos_km, 50) / 10.0)
        
        # Turbulence efficiency (interpolated from Table II)
        D_cm = sat.DOGS * 100.0
        if D_cm <= 20:
            eta_turb = 0.73
        elif D_cm <= 40:
            eta_turb = 0.73 - (D_cm - 20) / 20 * 0.07
        elif D_cm <= 60:
            eta_turb = 0.66 - (D_cm - 40) / 20 * 0.05
        elif D_cm <= 80:
            eta_turb = 0.61 - (D_cm - 60) / 20 * 0.03
        elif D_cm <= 100:
            eta_turb = 0.58 - (D_cm - 80) / 20 * 0.02
        else:
            eta_turb = max(0.35, 0.56 - (D_cm - 100) / 200 * 0.1)
        
        # Jitter efficiency
        theta_div = lam / sat.DOGS
        ratio = sat.theta_jitter / theta_div if theta_div > 0 else 0
        eta_jitter = math.exp(-2.0 * ratio ** 2)
        
        # Total channel transmission (deterministic - no random fluctuation)
        trans = max(1e-30, eta_turb * eta_jitter * tau_abs * tau_syst * tau_geom)
        
        eta_link = trans * eta_det
        
        # --- BB84 QBER model ---
        # At long distances, QBER increases toward 0.5 as dark counts dominate
        if eta_link > 1e-15:
            # QBER = (dark_count_contribution + misalignment) / total_detections
            signal_prob = eta_link
            dark_prob = pd
            qber_bb84 = (0.5 * dark_prob + ed * signal_prob) / (dark_prob + signal_prob)
        else:
            qber_bb84 = 0.5
        qber_bb84 = max(0.0, min(0.5, qber_bb84))
        
        # --- BB84 SKR model ---
        # R_BB84 ~ eta * (1 - H(e) - f*H(e)) for simple asymptotic
        if qber_bb84 < 0.11:
            h_e = CryptoMath.h(qber_bb84)
            skr_bb84_factor = max(0, 1.0 - 2.0 * h_e)
            skr_bb84 = F * eta_link * 0.5 * skr_bb84_factor  # 0.5 for sifting
        else:
            skr_bb84 = 0.0
        
        qber_alice_sat.append(qber_bb84)
        qber_bob_sat.append(qber_bb84 * 1.02)  # Slight asymmetry
        skr_alice_sat.append(skr_bb84)
        skr_bob_sat.append(skr_bb84 * 0.98)
        
        # --- TF-QKD metrics (deterministic calculation) ---
        # eta_A = eta_B = trans * eta_det
        eta_A = trans * eta_det
        eta_B = trans * eta_det
        
        # Optimize mu
        best_mu, best_R = 0.5, 0.0
        for mu in np.linspace(0.1, 1.5, 30):
            gA = mu * eta_A
            gB = mu * eta_B
            sg = math.sqrt(gA * gB)
            
            # p_XX (X-basis gain)
            p_xx = 0.5 * (1 - pd) * (math.exp(-sg) + math.exp(sg)) * math.exp(-0.5 * (gA + gB))
            p_xx -= (1 - pd) ** 2 * math.exp(-(gA + gB))
            p_xx += pd * (1 - pd) * math.exp(-0.5 * (gA + gB))
            p_xx = max(1e-15, min(1.0, p_xx))
            
            # e_XX (X-basis QBER)
            num = math.exp(-sg) - (1 - pd) * math.exp(-0.5 * (gA + gB))
            den = math.exp(-sg) + math.exp(sg) - 2 * (1 - pd) * math.exp(-0.5 * (gA + gB))
            e_xx = abs(num / den) + ed if den > 0 else 0.5
            e_xx = max(0.0, min(0.5, e_xx))
            
            # e_ZZ (Z-basis phase error)
            asym = abs(gA - gB) / max(gA + gB, 1e-12)
            e_zz = min(0.49, 0.15 + 0.10 * asym + min(0.2, pd / max(gA + gB, 1e-12)) + ed)
            
            if e_xx < 0.5 and e_zz < 0.5:
                R = 2.0 * p_xx * (1.0 - fec * CryptoMath.h(e_xx) - CryptoMath.h(e_zz))
                if R > best_R:
                    best_R = R
                    best_mu = mu
        
        # Calculate final TF-QKD metrics with optimal mu
        mu_opt = best_mu
        gA = mu_opt * eta_A
        gB = mu_opt * eta_B
        sg = math.sqrt(gA * gB)
        
        p_xx = 0.5 * (1 - pd) * (math.exp(-sg) + math.exp(sg)) * math.exp(-0.5 * (gA + gB))
        p_xx -= (1 - pd) ** 2 * math.exp(-(gA + gB))
        p_xx += pd * (1 - pd) * math.exp(-0.5 * (gA + gB))
        p_xx = max(1e-15, min(1.0, p_xx))
        
        num = math.exp(-sg) - (1 - pd) * math.exp(-0.5 * (gA + gB))
        den = math.exp(-sg) + math.exp(sg) - 2 * (1 - pd) * math.exp(-0.5 * (gA + gB))
        e_xx_tf = abs(num / den) + ed if den > 0 else 0.5
        e_xx_tf = max(0.0, min(0.5, e_xx_tf))
        
        asym = abs(gA - gB) / max(gA + gB, 1e-12)
        e_zz_tf = min(0.49, 0.15 + 0.10 * asym + min(0.2, pd / max(gA + gB, 1e-12)) + ed)
        
        R_pulse = 2.0 * p_xx * (1.0 - fec * CryptoMath.h(e_xx_tf) - CryptoMath.h(e_zz_tf))
        R_pulse = max(0.0, R_pulse)
        skr_tfqkd_val = R_pulse * F
        
        qber_tfqkd.append(e_xx_tf)
        skr_tfqkd.append(skr_tfqkd_val)
    
    # Restore original distances
    cfg.linkdistances[(0, 1)] = orig_dist_ac
    cfg.linkdistances[(1, 2)] = orig_dist_cb
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Convert to arrays
    distances_km = np.array(distances)
    qber_alice_arr = np.array(qber_alice_sat)
    qber_bob_arr = np.array(qber_bob_sat)
    qber_tfqkd_arr = np.array(qber_tfqkd)
    skr_alice_arr = np.array(skr_alice_sat)
    skr_bob_arr = np.array(skr_bob_sat)
    skr_tfqkd_arr = np.array(skr_tfqkd)
    
    # --- QBER vs Distance ---
    ax1.plot(distances_km, qber_alice_arr, 'b-', linewidth=2, label='QBER Alice→Sat')
    ax1.plot(distances_km, qber_bob_arr, color='orange', linewidth=2, label='QBER Bob→Sat')
    ax1.plot(distances_km, qber_tfqkd_arr, 'g-', linewidth=2.5, label='QBER TF-QKD')
    ax1.axhline(y=0.11, color='red', linestyle='--', linewidth=2, label='Security threshold')
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('QBER', fontsize=12)
    ax1.set_title('QBER vs Distance', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.55)
    
    # --- SKR vs Distance ---
    # Normalize to max value for display
    max_skr_bb84 = max(skr_alice_arr.max(), skr_bob_arr.max(), 1e-10)
    max_skr_tf = max(skr_tfqkd_arr.max(), 1e-10)
    
    # Normalize both to range [0, 0.25] for comparable display
    skr_alice_norm = (skr_alice_arr / max_skr_bb84) * 0.25 if max_skr_bb84 > 0 else np.zeros_like(distances_km)
    skr_bob_norm = (skr_bob_arr / max_skr_bb84) * 0.25 if max_skr_bb84 > 0 else np.zeros_like(distances_km)
    skr_tfqkd_norm = (skr_tfqkd_arr / max_skr_tf) * 0.25 if max_skr_tf > 0 else np.zeros_like(distances_km)
    
    ax2.plot(distances_km, skr_alice_norm, 'b-', linewidth=2, label='SKR Alice→Sat')
    ax2.plot(distances_km, skr_bob_norm, color='orange', linewidth=2, label='SKR Bob→Sat')
    ax2.plot(distances_km, skr_tfqkd_norm, 'g-', linewidth=2.5, label='SKR TF-QKD')
    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel('Secure Key Rate (normalized)', fontsize=12)
    ax2.set_title('SKR vs Distance', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.30)
    
    plt.subplots_adjust(hspace=0.35, top=0.95)
    _safe_save(fig, savepath)


# ============================================================================
# TEXT OUTPUT
# ============================================================================

def print_link_results(cfg, results, elapsed):
    print("\n" + "=" * 70)
    print("                    QKD SIMULATION RESULTS")
    print("=" * 70)
    print(f" Protocol:       {cfg.protocol}")
    print(f" Channel:        {cfg.channeltype.value.upper()}")
    print(f" Nodes:          {cfg.nodecount}")
    print(f" Topology:       {cfg.topology}")
    print(f" Finite-Size:    {'ENABLED' if cfg.usefinitesize else 'DISABLED'}")
    print(f" Security Model: {cfg.securitymodel.upper()}")
    if cfg.securitymodel.lower() == "gllp" and cfg.protocol.upper() == "BB84":
        if cfg.gllp_optimize:
            print(" GLLP mu/nu:     AUTO-OPTIMIZED per link")
        else:
            print(f" GLLP mu/nu:     {cfg.gllp_mu:.3f} / {cfg.gllp_nu:.3f}")
    if cfg.protocol.upper() == "BB84":
        print(" Security Gap:   Showing Shannon vs GLLP side-by-side")
    used_seq = any(r.get("sequence_used") for r in results if r.get("success"))
    print(f" Simulator:      {'SeQUeNCe (Real)' if used_seq else 'Analytical'}")
    print("\n --- LINK RESULTS ---")
    tot_sec, tot_q, cnt = 0, 0.0, 0
    tot_shannon, tot_gllp = 0, 0
    for r in results:
        if r.get("success"):
            lk = r["link"]
            tag = "[OK] SeQUeNCe" if r.get("sequence_used") else "o Analytical"
            print(f"\n Link: N{lk[0]} <-> N{lk[1]} ({r.get('distkm',0):.1f} km) [{tag}]")
            print(f"   Transmission: {r.get('transmission',0):.2e}")
            print(f"   Sifted Bits:  {r.get('siftedlength',0)}")
            print(f"   QBER:         {r.get('qber',0)*100:.2f}%")
            print(f"   Secure Bits:  {r.get('securelength',0)}")
            print(f"   Secret Rate:  {r.get('secretrate',0):.1f} bps")
            print(f"   Status:       {'SECURE' if r.get('issecure') else 'INSECURE'}")
            if cfg.protocol.upper() == "BB84":
                sifted = int(r.get("siftedlength", 0))
                qber = float(r.get("qber", 0.0))
                cctx = {"transmission": float(r.get("transmission", 1.0))}
                shannon_bits = compute_secure_bits_model(
                    cfg, "BB84", sifted, qber, "shannon", context=cctx
                )
                gllp_bits = compute_secure_bits_model(
                    cfg, "BB84", sifted, qber, "gllp", context=cctx
                )
                shannon_rate = shannon_bits / cfg.duration
                gllp_rate = gllp_bits / cfg.duration
                if shannon_bits > 0:
                    drop = (shannon_bits - gllp_bits) * 100.0 / shannon_bits
                    drop_txt = f"{drop:.1f}%"
                else:
                    drop_txt = "N/A"
                print(f"   Security Gap: Shannon={shannon_bits} ({shannon_rate:.1f} bps), "
                      f"GLLP={gllp_bits} ({gllp_rate:.1f} bps), Drop={drop_txt}")
                tot_shannon += shannon_bits
                tot_gllp += gllp_bits
            tot_sec += r.get("securelength", 0)
            tot_q += r.get("qber", 0)
            cnt += 1
        else:
            lk = r.get("link", ("?", "?"))
            print(f"\n Link: N{lk[0]} <-> N{lk[1]} -- FAILED: {r.get('error','?')}")
    print("\n --- SUMMARY ---")
    if cnt:
        print(f" Total Secure:   {tot_sec}")
        print(f" Avg QBER:       {tot_q/cnt*100:.2f}%")
        print(f" Net Rate:       {tot_sec/cfg.duration:.1f} bps")
        if cfg.protocol.upper() == "BB84":
            if tot_shannon > 0:
                tot_drop = (tot_shannon - tot_gllp) * 100.0 / tot_shannon
                tot_drop_txt = f"{tot_drop:.1f}%"
            else:
                tot_drop_txt = "N/A"
            print(f" Security Gap:   Shannon={tot_shannon}, GLLP={tot_gllp}, Drop={tot_drop_txt}")
    print(f" Runtime:        {elapsed:.3f} s")
    print("=" * 70)


def print_e2e(e2e, cfg):
    print("\n" + "=" * 70)
    print("                 END-TO-END KEY EXCHANGE RESULTS")
    print("                    (XOR Trusted Relay Protocol)")
    print("=" * 70)
    for r in e2e:
        print(f"\n Path: N{r['source']} -> N{r['dest']}")
        print(f"   Route:        {' -> '.join(f'N{n}' for n in r.get('path',[]))}")
        print(f"   Hops:         {r.get('hops',0)}")
        print(f"   Distance:     {r.get('totaldistance',0):.1f} km")
        if r.get("success"):
            print(f"   Link Rates:   {[f'{x:.0f}' for x in r.get('linkrates',[])]}")
            if r.get("linksiftedrates"):
                print(f"   Sifted Rates: {[f'{x:.0f}' for x in r.get('linksiftedrates',[])]}")
            print(f"   Link QBERs:   {[f'{x*100:.2f}%' for x in r.get('linkqbers',[])]}")
            print(f"   Bottleneck:   Link {r.get('bottlenecklink',-1)} "
                  f"({r.get('bottleneckrate',0):.1f} bps secret, "
                  f"{r.get('bottlenecksiftedrate',0):.1f} bps sifted)")
            print(f"   E2E Rate:     {r.get('e2erate',0):.1f} bps")
            print(f"   E2E QBER:     {r.get('e2eqber',0)*100:.2f}%")
            if r.get("qbersteps"):
                qber_chain = " -> ".join(
                    f"{s['cumulative_qber']*100:.2f}%" for s in r["qbersteps"]
                )
                print(f"   QBER Chain:   {qber_chain}")
            print(f"   E2E Sifted:   {r.get('e2esiftedbits',0)} bits")
            print(f"   E2E Secure:   {r.get('e2esecbits',0)} bits")
            print(f"   Status:       {'SECURE' if r.get('issecure') else 'NEEDS REVIEW'}")
        else:
            print(f"   Status:       FAILED")
            if r.get("warning"):
                print(f"   [!]  {r['warning']}")
    print("=" * 70)


def print_tfqkd(result):
    print("\n" + "=" * 70)
    print("               TWIN-FIELD QKD RESULTS")
    print("            [!]  ANALYTICAL (SeQUeNCe doesn't support TF-QKD)")
    print("=" * 70)
    if not result.get("success"):
        print(f" FAILED: {result.get('error','?')}")
        return
    print(f" Protocol: TF-QKD (Twin-Field, sending-or-not-sending)")
    print(f" Topology: Alice(N{result['alice']}) -- Charlie(N{result['charlie']}*)"
          f" -- Bob(N{result['bob']})")
    print(f" Charlie:  {result['charlietrust']}")
    print(f"\n Channel:        {result['channeltype'].upper()}")
    print(f" Dist A->C:       {result['distac']:.1f} km")
    print(f" Dist C->B:       {result['distcb']:.1f} km")
    print(f" Total:          {result['totaldist']:.1f} km")
    print(f"\n --- Channel Parameters ---")
    print(f" Trans(A->C):     {result['transac']:.6e}")
    print(f" Trans(C->B):     {result['transcb']:.6e}")
    print(f" eta__A:            {result['eta_A']:.6e}")
    print(f" eta__B:            {result['eta_B']:.6e}")
    print(f" u_optimal:      {result['mu_optimal']:.4f} photon/pulse")
    print(f"\n --- TF-QKD Metrics (Eq. B2-B6) ---")
    print(f" p_XX:           {result['p_XX']:.6e}")
    print(f" e_XX:           {result['e_XX']*100:.2f}%")
    print(f" e_ZZ:           {result['e_ZZ']*100:.2f}%")
    print(f" R/pulse:        {result['R_per_pulse']:.2e} bit/pulse")
    print(f"\n --- Key Generation ---")
    if result.get("detector_model", "none") != "none":
        print(f" Detector Model:  {result.get('detector_model','none')}")
        print(f" Detector Count:  {result.get('detector_count',0)}")
        print(f" Dead Time:       {result.get('detector_deadtime_ns',0):.2f} ns")
        print(f" ClickRate Raw:   {result.get('click_rate_raw',0):.3e} Hz")
        print(f" ClickRate Sat:   {result.get('click_rate_sat',0):.3e} Hz")
        print(f" Exp Clicks Raw:  {result.get('expected_clicks_raw',0):.3e}")
        print(f" Exp Clicks Sat:  {result.get('expected_clicks_sat',0):.3e}")
    print(f" Total Pulses:   {result['totalpulses']:,}")
    print(f" Clicks:         {result['clickcount']:,}")
    print(f" Sifted Bits:    {result['siftedlength']:,}")
    print(f" Secure Bits:    {result['securelength']:,}")
    print(f"\n --- Final ---")
    print(f" Secret Rate:    {result['secretrate']:.1f} bps")
    print(f" Theoretical:    {result['theoretical_rate']:.1f} bps")
    if "theoretical_rate_raw" in result:
        print(f" Theo (No Sat):  {result['theoretical_rate_raw']:.1f} bps")
    print(f" SECURE:         {'YES [OK]' if result['issecure'] else 'NO [X]'}")
    if result["issecure"]:
        print(f"\n [OK] Charlie performs interference but NEVER learns the key!")
    print("=" * 70)


# ============================================================================
# MAIN INTERACTIVE FUNCTION
# ============================================================================

def maininteract():
    import os
    outdir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 70)
    print("       QKD NETWORK SIMULATOR -- ENHANCED")
    print("       SeQUeNCe for BB84 | Analytical for TF-QKD/E91")
    print("       arXiv:2507.23466 for Satellite TF-QKD")
    print("=" * 70)
    if SEQUENCE_AVAILABLE:
        print(" [OK] SeQUeNCe: LOADED")
    else:
        print(" [X] SeQUeNCe: NOT AVAILABLE -- BB84 will use E91 analytical fallback")

    print("\n Select Simulation Mode:")
    print(" [1] Standard Link Simulation (BB84/E91 -- Mesh)")
    print(" [2] End-to-End Multi-Hop (XOR Trusted Relay -- Linear/Star)")
    print(" [3] Untrusted Node (TF-QKD -- 3 Nodes, Linear) [!] Analytical")
    print(" [4] UAV-Based QKD (BB84/E91)")
    print(" [5] Satellite-Based QKD (TF-QKD -- per arXiv:2507.23466) [!] Analytical")

    mode = input("\n Mode (1-5, default 1): ").strip()
    mode = int(mode) if mode.isdigit() and 1 <= int(mode) <= 5 else 1

    # ---- channel type ----
    if mode == 4:
        channeltype = ChannelType.FSO_UAV
        print(" Channel: FSO_UAV (automatic)")
    elif mode == 5:
        channeltype = ChannelType.FSO_SATELLITE
        print(" Channel: FSO_SATELLITE (automatic -- GEO link)")
    elif mode == 3:
        print(" Select Channel for TF-QKD:")
        print(" [1] Fiber  [2] FSO Ground")
        c = input(" Choice (1/2, default 1): ").strip()
        cmap = {"1": ChannelType.FIBER, "2": ChannelType.FSO_GROUND}
        channeltype = cmap.get(c, ChannelType.FIBER)
    else:
        print(" Select Channel:")
        print(" [1] Fiber Optic  [2] Free-Space (Ground)")
        c = input(" Choice (1/2, default 1): ").strip()
        channeltype = ChannelType.FSO_GROUND if c == "2" else ChannelType.FIBER

    # ---- protocol ----
    if mode == 3 or mode == 5:
        proto = "TF-QKD"
        print(f" Protocol: TF-QKD (Analytical -- arXiv:2507.23466)")
    else:
        print(" Select Protocol:")
        print(" [1] BB84 (SeQUeNCe simulation)")
        print(" [2] E91  (Analytical with SeQUeNCe channel params)")
        p = input(" Choice (1/2, default 1): ").strip()
        proto = "E91" if p == "2" else "BB84"

    # ---- security model ----
    securitymodel = "shannon"
    gllp_mu = 0.5
    gllp_nu = 0.1
    gllp_optimize = True
    if proto == "BB84":
        print("\n Security Model (BB84):")
        print(" [1] Shannon asymptotic (default)")
        print(" [2] GLLP + vacuum/weak decoy bound")
        sm = input(" Choice (1/2, default 1): ").strip()
        if sm == "2":
            securitymodel = "gllp"
            print(" GLLP intensities:")
            print(" [1] Auto-optimize mu/nu per link (recommended)")
            print(" [2] Manual mu/nu input")
            gmode = input(" Choice (1/2, default 1): ").strip()
            gllp_optimize = (gmode != "2")
            if not gllp_optimize:
                mu_txt = input(" Signal intensity mu (default 0.5): ").strip()
                nu_txt = input(" Weak decoy intensity nu (default 0.1): ").strip()
                gllp_mu = float(mu_txt) if mu_txt else 0.5
                gllp_nu = float(nu_txt) if nu_txt else 0.1
                if gllp_mu <= 0:
                    gllp_mu = 0.5
                if gllp_nu <= 0 or gllp_nu >= gllp_mu:
                    gllp_nu = max(0.01, min(0.1, 0.5 * gllp_mu))
    else:
        print("\n Security Model: Shannon asymptotic")

    # ---- nodes ----
    if mode == 3 or mode == 5:
        nnodes = 3
        print(f" Nodes: 3 (Alice -- Charlie* -- Bob)")
    else:
        n = input(f" Nodes (2-10, default 4): ").strip()
        nnodes = max(2, min(10, int(n) if n.isdigit() else 4))

    # ---- topology ----
    if mode == 1:
        topo = "mesh"
        print(" Topology: Mesh")
    elif mode in (3, 5):
        topo = "linear"
    elif mode == 2:
        print(" [1] Linear  [2] Star")
        t = input(" Topology (1/2, default 1): ").strip()
        topo = "star" if t == "2" else "linear"
    else:
        print(" [1] Linear  [2] Star  [3] Mesh")
        t = input(" Topology (1-3, default 1): ").strip()
        topo = {"1": "linear", "2": "star", "3": "mesh"}.get(t, "linear")

    # ---- distances ----
    links = getlinklist(topo, nnodes)
    dists = {}
    nodetrust = {}
    satconfig = SatelliteConfig()
    uavconfig = UAVConfig()

    if mode == 5:
        # Satellite mode -- derive distances from OGS-sat geometry
        print("\n Satellite (GEO) Parameters:")
        print(" Orbit: [1] LEO (500km) [2] MEO (2000km) [3] GEO (35786km)")
        o = input(" Choice (1-3, default 3): ").strip()
        omap = {"1": ("LEO", 500), "2": ("MEO", 2000), "3": ("GEO", 35786)}
        orb, alt = omap.get(o, ("GEO", 35786))
        satconfig.orbittype = orb
        satconfig.altitude = alt
        elev = input(f" Elevation angle (deg, default 30): ").strip()
        satconfig.elevation = float(elev) if elev else 30.0
        dogs = input(f" OGS aperture diameter (m, default 1.0): ").strip()
        satconfig.DOGS = float(dogs) if dogs else 1.0
        dsat = input(f" Satellite aperture (m, default 0.50): ").strip()
        satconfig.Dsat = float(dsat) if dsat else 0.50
        # Compute OGS-sat distance  L = h / sin(theta__elev)
        L_ogs_sat = satconfig.altitude / math.sin(math.radians(max(satconfig.elevation, 5)))
        print(f" OGS-Satellite distance: {L_ogs_sat:.1f} km")
        dists[(0, 1)] = L_ogs_sat   # Alice -> Charlie (satellite)
        dists[(1, 2)] = L_ogs_sat   # Charlie -> Bob
        nodetrust[1] = NodeTrust.UNTRUSTED
        print(f" N1 (Charlie / satellite) is UNTRUSTED")
    elif mode == 3:
        print("\n Enter link distances (km):")
        for a, b in links:
            default = 10
            d = input(f"  N{a}<->N{b} (default {default}): ").strip()
            dists[(a, b)] = float(d) if d else float(default)
        nodetrust[1] = NodeTrust.UNTRUSTED
        print(f" N1 (Charlie) is UNTRUSTED")
    else:
        print("\n Enter link distances (km):")
        for a, b in links:
            default = 5 if channeltype == ChannelType.FSO_UAV else 10
            d = input(f"  N{a}<->N{b} (default {default}): ").strip()
            dists[(a, b)] = float(d) if d else float(default)

    # UAV altitude
    if mode == 4:
        alt = input("\n UAV altitude (km, default 1): ").strip()
        uavconfig.altitude = float(alt) if alt else 1.0

    # Finite-size
    print("\n [1] Disable finite-size correction  [2] Enable finite-size correction")
    fs = input(" Choice (1/2, default 1): ").strip()
    usefinitesize = (fs == "2")

    # ---- Build config ----
    cfg = SimConfig(
        nodecount=nnodes, topology=topo, protocol=proto,
        keylength=256, keycount=50, duration=10.0,
        securitymodel=securitymodel, gllp_mu=gllp_mu, gllp_nu=gllp_nu,
        gllp_optimize=gllp_optimize,
        channeltype=channeltype, linkdistances=dists,
        nodetrust=nodetrust, uavconfig=uavconfig,
        satelliteconfig=satconfig, usefinitesize=usefinitesize,
    )

    print(f"\n{'='*70}")
    tag = "SeQUeNCe" if proto == "BB84" and SEQUENCE_AVAILABLE else "Analytical"
    print(f" Running {proto} via {tag}...")
    print(f"{'='*70}")
    start = time.time()

    # ========== MODE 1 : Mesh link simulation ==========
    if mode == 1:
        if proto == "BB84" and SEQUENCE_AVAILABLE:
            sim = BB84SimSequence(cfg)
        else:
            sim = E91Sim(cfg)
        results = sim.runsim()
        elapsed = time.time() - start
        print_link_results(cfg, results, elapsed)
        drawnetwork(results, cfg, savepath=f"{outdir}/network.png")
        drawqber(results, cfg, savepath=f"{outdir}/qber_bar.png")
        drawrates(results, cfg, savepath=f"{outdir}/rates.png")
        drawscatter_qber(results, cfg, savepath=f"{outdir}/scatter_qber.png")
        drawscatter_rate(results, cfg, savepath=f"{outdir}/scatter_rate.png")
        if proto == "E91":
            draw_chsh_formula_examples(results, savepath=f"{outdir}/chsh_formula_examples.png")

    # ========== MODE 2 : Multi-Hop XOR Relay ==========
    elif mode == 2:
        relay = TrustedRelayNetwork(cfg)
        e2e = relay.run_all_e2e()
        elapsed = time.time() - start
        print_link_results(cfg, relay.linkresults, elapsed)
        print_e2e(e2e, cfg)
        drawnetwork(relay.linkresults, cfg, "XOR Trusted Relay", e2e,
                    savepath=f"{outdir}/network.png")
        drawqber(relay.linkresults, cfg, e2e, savepath=f"{outdir}/qber_bar.png")
        drawrates(relay.linkresults, cfg, e2e, savepath=f"{outdir}/rates.png")
        drawscatter_qber(relay.linkresults, cfg, e2e, savepath=f"{outdir}/scatter_qber.png")
        drawscatter_rate(relay.linkresults, cfg, e2e, savepath=f"{outdir}/scatter_rate.png")
        drawe2e_rates(e2e, cfg, savepath=f"{outdir}/e2e_rates.png")
        if proto == "E91":
            draw_chsh_formula_examples(relay.linkresults, savepath=f"{outdir}/chsh_formula_examples.png")

    # ========== MODE 3 : TF-QKD Untrusted Node ==========
    elif mode == 3:
        tfqkd = TwinFieldQKD(cfg)
        result = tfqkd.runsim(0, 1, 2)
        elapsed = time.time() - start
        print_tfqkd(result)
        mock_links = [
            dict(link=(0, 1), success=True, distkm=result["distac"],
                 qber=result["e_XX"], secretrate=result["secretrate"] / 2,
                 issecure=result["issecure"]),
            dict(link=(1, 2), success=True, distkm=result["distcb"],
                 qber=result["e_XX"], secretrate=result["secretrate"] / 2,
                 issecure=result["issecure"]),
        ]
        drawnetwork(mock_links, cfg, "TF-QKD (Untrusted N1)",
                    savepath=f"{outdir}/network.png")
        draw_tfqkd(result, cfg, savepath=f"{outdir}/tfqkd_analysis.png")

    # ========== MODE 4 : UAV-Based QKD ==========
    elif mode == 4:
        print(f" UAV Altitude: {cfg.uavconfig.altitude} km")
        if proto == "BB84" and SEQUENCE_AVAILABLE:
            sim = BB84SimSequence(cfg)
        else:
            sim = E91Sim(cfg)
        results = sim.runsim()
        elapsed = time.time() - start
        print_link_results(cfg, results, elapsed)
        drawnetwork(results, cfg, f"UAV @ {cfg.uavconfig.altitude}km",
                    savepath=f"{outdir}/network.png")
        drawqber(results, cfg, savepath=f"{outdir}/qber_bar.png")
        drawrates(results, cfg, savepath=f"{outdir}/rates.png")
        drawscatter_qber(results, cfg, savepath=f"{outdir}/scatter_qber.png")
        drawscatter_rate(results, cfg, savepath=f"{outdir}/scatter_rate.png")
        if proto == "E91":
            draw_chsh_formula_examples(results, savepath=f"{outdir}/chsh_formula_examples.png")

    # ========== MODE 5 : Satellite TF-QKD (arXiv:2507.23466) ==========
    elif mode == 5:
        print(f" Orbit: {cfg.satelliteconfig.orbittype} "
              f"({cfg.satelliteconfig.altitude} km)")
        print(f" Elevation: {cfg.satelliteconfig.elevation}deg")
        print(f" OGS aperture: {cfg.satelliteconfig.DOGS*100:.0f} cm")
        print(f" Sat aperture: {cfg.satelliteconfig.Dsat*100:.0f} cm")
        tfqkd = TwinFieldQKD(cfg)
        result = tfqkd.runsim(0, 1, 2)
        elapsed = time.time() - start
        print_tfqkd(result)
        mock_links = [
            dict(link=(0, 1), success=True, distkm=result["distac"],
                 qber=result["e_XX"], secretrate=result["secretrate"] / 2,
                 issecure=result["issecure"]),
            dict(link=(1, 2), success=True, distkm=result["distcb"],
                 qber=result["e_XX"], secretrate=result["secretrate"] / 2,
                 issecure=result["issecure"]),
        ]
        drawnetwork(mock_links, cfg,
                    f"Satellite TF-QKD ({cfg.satelliteconfig.orbittype})",
                    savepath=f"{outdir}/network.png")
        draw_tfqkd(result, cfg, savepath=f"{outdir}/tfqkd_analysis.png")
        draw_satellite_distance_analysis(cfg, savepath=f"{outdir}/satellite_distance_analysis.png")

    runtime = time.time() - start
    print(f"\n Runtime: {runtime:.3f} s")
    print(f" Figures saved to: {outdir}/")
    if plt.get_fignums():
        plt.show()  # Show all figures at once
    print("=" * 70)


if __name__ == "__main__":
    maininteract()
