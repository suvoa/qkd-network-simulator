#!/usr/bin/env python3
"""
================================================================================
E91 QKD NETWORK SIMULATOR using QuTiP
================================================================================

Full network simulator for the E91 (Ekert 1991) entanglement-based QKD protocol.

Quantum Backend: QuTiP (Quantum Toolbox in Python)
  - Density matrix representation of Bell / Werner states
  - Projective measurement operators built from QuTiP kets
  - Born rule: P(a,b) = Tr(rho . Pi_a^A (x) Pi_b^B)
  - CHSH inequality for eavesdropper detection

Features:
  - Multi-node networks (2-10 nodes)
  - Topologies: linear, star, mesh
  - Fiber and free-space optical (FSO) channel models
  - Per-link CHSH Bell test
  - End-to-end XOR trusted relay
  - Eavesdropper detection mode
  - Distance sweep analysis
  - Full visualization suite

Install:  pip install qutip numpy matplotlib networkx

Reference: A. Ekert, Phys. Rev. Lett. 67, 661 (1991)
================================================================================
"""

import os
import sys
import math
import time
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

# ── QuTiP ───────────────────────────────────────────────────────────────────
try:
    from qutip import basis, tensor, qeye, ket2dm, expect, Qobj
    QUTIP_OK = True
    print("[OK] QuTiP loaded successfully")
except ImportError as e:
    QUTIP_OK = False
    print(f"[X] QuTiP not available: {e}")
    print("    Install: pip install qutip")
    sys.exit(1)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
import networkx as nx


# ============================================================================
# E91 PROTOCOL CONSTANTS
# ============================================================================
# Measurement angles chosen for maximal CHSH violation on |Phi+>
#
#   Alice: a1 = 0,    a2 = pi/8,   a3 = pi/4
#   Bob:   b1 = pi/8, b2 = pi/4,   b3 = 3*pi/8
#
# Key-generating pairs (same angle):
#   (a2, b1) both at pi/8
#   (a3, b2) both at pi/4
#
# CHSH: S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3)
# For ideal |Phi+>:  S = 2*sqrt(2) ~ 2.828

ALICE_ANGLES = [0.0, math.pi / 8, math.pi / 4]
BOB_ANGLES   = [math.pi / 8, math.pi / 4, 3 * math.pi / 8]

KEY_BASIS_PAIRS = [(1, 0), (2, 1)]

CHSH_TERMS = [
    (+1, 0, 0),   # +E(a1, b1)
    (-1, 0, 2),   # -E(a1, b3)
    (+1, 2, 0),   # +E(a3, b1)
    (+1, 2, 2),   # +E(a3, b3)
]


# ============================================================================
# ENUMS & CONFIG
# ============================================================================

class ChannelType(Enum):
    FIBER = "fiber"
    FSO_GROUND = "fso_ground"


class NodeTrust(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass
class FiberPhysics:
    attenuation: float = 0.2       # dB/km
    detector_eff: float = 0.9
    dark_count_prob: float = 0.001
    base_visibility: float = 0.98  # Werner-state visibility at 0 km
    vis_decay_per_km: float = 0.002  # visibility loss per km


@dataclass
class FSOPhysics:
    wavelength: float = 1550e-9
    tx_aperture: float = 0.1       # m
    rx_aperture: float = 0.3       # m
    beam_divergence: float = 10e-6 # rad
    pointing_error: float = 1e-6   # rad
    detector_eff: float = 0.85
    dark_count_prob: float = 0.001
    visibility_km: float = 23.0    # atmospheric visibility
    base_visibility: float = 0.96
    vis_decay_per_km: float = 0.003


@dataclass
class SimConfig:
    nodecount: int = 4
    topology: str = "linear"
    num_pairs: int = 10000         # entangled pairs per link
    duration: float = 10.0         # seconds (for rate calculation)
    seed: int = 42
    channeltype: ChannelType = ChannelType.FIBER
    fiber: FiberPhysics = field(default_factory=FiberPhysics)
    fso: FSOPhysics = field(default_factory=FSOPhysics)
    linkdistances: Dict[Tuple[int, int], float] = field(default_factory=dict)
    nodetrust: Dict[int, NodeTrust] = field(default_factory=dict)

    def get_distance(self, a: int, b: int) -> float:
        if (a, b) in self.linkdistances:
            return self.linkdistances[(a, b)]
        if (b, a) in self.linkdistances:
            return self.linkdistances[(b, a)]
        return 10.0

    def get_trust(self, nid: int) -> NodeTrust:
        return self.nodetrust.get(nid, NodeTrust.TRUSTED)


# ============================================================================
# UTILITY
# ============================================================================

def h_binary(p: float) -> float:
    """Binary Shannon entropy H(p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def get_links(topology: str, n: int) -> List[Tuple[int, int]]:
    t = topology.lower()
    if t == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    elif t == "star":
        return [(0, i) for i in range(1, n)]
    elif t == "mesh":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return []


def set_seed(seed: int):
    """Set global RNG seeds for reproducible simulation output."""
    np.random.seed(seed)
    random.seed(seed)


# ============================================================================
# CHANNEL MODELS
# ============================================================================

class ChannelModel:
    """Compute (transmission, visibility) for a link."""

    @staticmethod
    def fiber(dist_km: float, ph: FiberPhysics) -> Tuple[float, float]:
        loss_db = ph.attenuation * dist_km
        transmission = 10 ** (-loss_db / 10) * ph.detector_eff
        visibility = max(0.5, ph.base_visibility - ph.vis_decay_per_km * dist_km)
        return transmission, visibility

    @staticmethod
    def fso_ground(dist_km: float, ph: FSOPhysics) -> Tuple[float, float]:
        vis = ph.visibility_km
        wl_nm = ph.wavelength * 1e9
        q = 1.6 if vis > 50 else (1.3 if vis > 6 else 0.585 * vis ** (1 / 3))
        alpha = (3.91 / vis) * (wl_nm / 550) ** (-q)
        dm = dist_km * 1000.0
        br = dm * ph.beam_divergence
        geom = min(1.0, (ph.rx_aperture / (2 * br)) ** 2)
        geom_db = -10 * math.log10(max(geom, 1e-10))
        point = math.exp(-(ph.pointing_error ** 2) / (ph.beam_divergence ** 2))
        point_db = -10 * math.log10(max(point, 1e-10))
        total_db = alpha * dist_km + geom_db + point_db
        transmission = max(1e-15, 10 ** (-total_db / 10) * ph.detector_eff)
        visibility = max(0.5, ph.base_visibility - ph.vis_decay_per_km * dist_km)
        return transmission, visibility

    @staticmethod
    def get_params(dist_km: float, cfg: SimConfig) -> Tuple[float, float]:
        if cfg.channeltype == ChannelType.FSO_GROUND:
            return ChannelModel.fso_ground(dist_km, cfg.fso)
        return ChannelModel.fiber(dist_km, cfg.fiber)


# ============================================================================
# QuTiP QUANTUM ENGINE
# ============================================================================

class E91QuantumEngine:
    """
    QuTiP-powered E91 quantum engine.

    For a given Werner-state visibility V, this engine:
      1. Builds rho = V|Phi+><Phi+| + (1-V)I/4   (QuTiP density matrix)
      2. Constructs projective measurement operators Pi(theta)
      3. Computes P(a,b) = Tr(rho . Pi_a (x) Pi_b)  via QuTiP expect()
      4. Returns all 36 joint probabilities (9 basis combos x 4 outcomes)
    """

    def __init__(self, visibility: float = 0.98):
        self.visibility = max(0.5, min(1.0, visibility))
        self.rho = self._build_state()
        self.joint_probs = self._precompute()

    def _build_state(self) -> Qobj:
        phi_plus = (
            tensor(basis(2, 0), basis(2, 0))
            + tensor(basis(2, 1), basis(2, 1))
        ) / np.sqrt(2)
        rho_pure = ket2dm(phi_plus)
        noise = tensor(qeye(2), qeye(2)) / 4.0
        V = self.visibility
        return V * rho_pure + (1 - V) * noise

    @staticmethod
    def _projector(angle: float, outcome: int) -> Qobj:
        """
        |+theta> =  cos(theta)|0> + sin(theta)|1>  -> outcome 0
        |-theta> = -sin(theta)|0> + cos(theta)|1>  -> outcome 1
        """
        if outcome == 0:
            st = np.cos(angle) * basis(2, 0) + np.sin(angle) * basis(2, 1)
        else:
            st = -np.sin(angle) * basis(2, 0) + np.cos(angle) * basis(2, 1)
        return ket2dm(st)

    def _precompute(self) -> Dict:
        probs = {}
        for ai, a_angle in enumerate(ALICE_ANGLES):
            for bi, b_angle in enumerate(BOB_ANGLES):
                p = np.zeros((2, 2))
                for a_out in range(2):
                    for b_out in range(2):
                        proj = tensor(
                            self._projector(a_angle, a_out),
                            self._projector(b_angle, b_out),
                        )
                        p[a_out, b_out] = max(0.0,
                            float(np.real(expect(proj, self.rho))))
                total = p.sum()
                if total > 0:
                    p /= total
                probs[(ai, bi)] = p
        return probs

    def state_fidelity(self) -> float:
        phi_plus = (
            tensor(basis(2, 0), basis(2, 0))
            + tensor(basis(2, 1), basis(2, 1))
        ) / np.sqrt(2)
        return float(np.real(expect(ket2dm(phi_plus), self.rho)))

    def state_purity(self) -> float:
        return float(np.real((self.rho * self.rho).tr()))


# ============================================================================
# E91 LINK SIMULATOR
# ============================================================================

class E91LinkSim:
    """Simulate E91 on a single link using QuTiP."""

    @staticmethod
    def run_link(a: int, b: int, cfg: SimConfig) -> dict:
        dist = cfg.get_distance(a, b)
        transmission, visibility = ChannelModel.get_params(dist, cfg)
        dark_prob = (cfg.fiber.dark_count_prob if cfg.channeltype == ChannelType.FIBER
                     else cfg.fso.dark_count_prob)

        # Pair survival (both photons must arrive)
        pair_survival = max(0.0, min(1.0, transmission))
        n_surviving = max(20, int(np.random.binomial(cfg.num_pairs, pair_survival)))

        # QuTiP engine for this link's visibility
        engine = E91QuantumEngine(visibility)

        # Random basis choices
        alice_bases = np.random.randint(0, 3, n_surviving)
        bob_bases   = np.random.randint(0, 3, n_surviving)

        # Sample outcomes using Born-rule probabilities from QuTiP
        alice_bits = np.zeros(n_surviving, dtype=int)
        bob_bits   = np.zeros(n_surviving, dtype=int)

        for (ai, bi), p_matrix in engine.joint_probs.items():
            mask = (alice_bases == ai) & (bob_bases == bi)
            count = int(mask.sum())
            if count == 0:
                continue
            flat = p_matrix.flatten()
            outcomes = np.random.choice(4, size=count, p=flat)
            alice_bits[mask] = outcomes // 2
            bob_bits[mask]   = outcomes % 2

        # Dark count flips
        dark_mask = np.random.random(n_surviving) < dark_prob
        n_dark = int(dark_mask.sum())
        if n_dark > 0:
            dark_idx = np.where(dark_mask)[0]
            alice_side = np.random.random(n_dark) < 0.5
            alice_idx = dark_idx[alice_side]
            bob_idx = dark_idx[~alice_side]
            if len(alice_idx) > 0:
                alice_bits[alice_idx] = np.random.randint(0, 2, len(alice_idx))
            if len(bob_idx) > 0:
                bob_bits[bob_idx] = np.random.randint(0, 2, len(bob_idx))

        # Sift key
        key_alice, key_bob = [], []
        for ai, bi in KEY_BASIS_PAIRS:
            mask = (alice_bases == ai) & (bob_bases == bi)
            key_alice.extend(alice_bits[mask].tolist())
            key_bob.extend(bob_bits[mask].tolist())

        # Correlations
        correlations = {}
        for ai in range(3):
            for bi in range(3):
                mask = (alice_bases == ai) & (bob_bases == bi)
                n = int(mask.sum())
                if n == 0:
                    correlations[(ai, bi)] = 0.0
                    continue
                a_vals = 1 - 2 * alice_bits[mask].astype(float)
                b_vals = 1 - 2 * bob_bits[mask].astype(float)
                correlations[(ai, bi)] = float(np.mean(a_vals * b_vals))

        # CHSH
        S = sum(sign * correlations.get((ai, bi), 0.0)
                for sign, ai, bi in CHSH_TERMS)

        # QBER
        errors = sum(a != b for a, b in zip(key_alice, key_bob))
        sifted = len(key_alice)
        qber = errors / sifted if sifted > 0 else 0.5

        # Secure key
        if qber < 0.11 and abs(S) > 2.0:
            sec_frac = max(0.0, 1.0 - 2.0 * h_binary(qber))
            secure_bits = int(sifted * sec_frac)
        else:
            sec_frac = 0.0
            secure_bits = 0

        return dict(
            link=(a, b), alice=f"N{a}", bob=f"N{b}", success=True,
            distkm=dist, channeltype=cfg.channeltype.value,
            transmission=transmission,
            visibility=visibility,
            state_fidelity=engine.state_fidelity(),
            state_purity=engine.state_purity(),
            pairsgen=cfg.num_pairs, pairsrec=n_surviving,
            siftedlength=sifted, securelength=secure_bits,
            qber=qber, errors=errors,
            chsh_S=S, chsh_violated=abs(S) > 2.0,
            correlations=correlations,
            secure_fraction=sec_frac,
            siftedrate=sifted / cfg.duration,
            secretrate=secure_bits / cfg.duration,
            issecure=abs(S) > 2.0 and qber < 0.11 and secure_bits > 0,
            key_alice=key_alice[:16],
            key_bob=key_bob[:16],
        )


# ============================================================================
# E91 NETWORK SIMULATOR
# ============================================================================

class E91NetworkSim:
    """Run E91 across all links in a network topology."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.results: List[dict] = []

    def runsim(self) -> List[dict]:
        links = get_links(self.cfg.topology, self.cfg.nodecount)
        results = []
        for a, b in links:
            print(f"  Link N{a}<->N{b} ({self.cfg.get_distance(a,b):.1f} km) ... ",
                  end="", flush=True)
            r = E91LinkSim.run_link(a, b, self.cfg)
            status = "SECURE" if r["issecure"] else "INSECURE"
            print(f"QBER={r['qber']*100:.2f}%  |S|={abs(r['chsh_S']):.3f}  {status}")
            results.append(r)
        self.results = results
        return results


# ============================================================================
# TRUSTED RELAY (XOR key relay for multi-hop)
# ============================================================================

class TrustedRelayNetwork:

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.linkresults: List[dict] = []
        self.e2eresults: List[dict] = []

    def run_links(self) -> List[dict]:
        sim = E91NetworkSim(self.cfg)
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

    @staticmethod
    def combine_qber_xor(qbers: List[float]) -> Tuple[float, List[dict]]:
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

    def calc_e2e(self, src, dst) -> dict:
        path = self._find_path(src, dst)
        if len(path) < 2:
            return dict(source=src, dest=dst, success=False, error="No path")

        secret_rates, sifted_rates, qbers, details = [], [], [], []
        total_d = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            r_secret = self._link_metric(a, b, "secretrate")
            r_sifted = self._link_metric(a, b, "siftedrate")
            if r_sifted <= 0:
                r_sifted = r_secret
            q = self._link_metric(a, b, "qber")
            d = self.cfg.get_distance(a, b)
            secret_rates.append(r_secret)
            sifted_rates.append(r_sifted)
            qbers.append(q)
            details.append(dict(link=(a, b), secretrate=r_secret,
                                siftedrate=r_sifted, qber=q, distance=d))
            total_d += d

        untrusted = [path[i] for i in range(1, len(path) - 1)
                     if self.cfg.get_trust(path[i]) == NodeTrust.UNTRUSTED]

        eq, qber_steps = self.combine_qber_xor(qbers)

        if all(r > 0 for r in sifted_rates):
            bn = sifted_rates.index(min(sifted_rates))
            bottleneck_sifted = min(sifted_rates)
            ok = True
        else:
            bottleneck_sifted, bn, ok = 0.0, -1, False

        e2e_sifted_bits = int(bottleneck_sifted * self.cfg.duration)
        if eq < 0.11 and e2e_sifted_bits > 0:
            e2e_sec = int(e2e_sifted_bits * max(0, 1 - 2 * h_binary(eq)))
        else:
            e2e_sec = 0
        e2e_rate = e2e_sec / self.cfg.duration if self.cfg.duration > 0 else 0

        return dict(
            source=src, dest=dst, path=path,
            hops=len(path) - 1, totaldistance=total_d,
            linkdetails=details, linkrates=secret_rates,
            linksiftedrates=sifted_rates, linkqbers=qbers,
            bottlenecklink=bn,
            bottleneckrate=secret_rates[bn] if bn >= 0 else 0,
            bottlenecksiftedrate=bottleneck_sifted,
            e2erate=e2e_rate, e2eqber=eq, e2esecbits=e2e_sec,
            e2esiftedbits=e2e_sifted_bits,
            qbersteps=qber_steps,
            untrusted=untrusted,
            warning="Path has untrusted intermediate nodes!" if untrusted else None,
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
# EAVESDROPPER COMPARISON
# ============================================================================

def run_eavesdropper_comparison(cfg: SimConfig, existing_results=None):
    """
    Compare normal E91 vs eavesdropped channel.
    Eve's intercept-resend destroys entanglement -> visibility drops to ~0.5.
    CHSH falls below 2 -> detected.

    If existing_results is provided, reuse them as the 'no Eve' baseline
    instead of running a new simulation.
    """
    links = get_links(cfg.topology, cfg.nodecount)
    safe_results = []
    eve_results = []

    # Build lookup from existing results if provided
    existing_lookup = {}
    if existing_results:
        for r in existing_results:
            if r.get("success"):
                existing_lookup[tuple(r["link"])] = r

    for a, b in links:
        dist = cfg.get_distance(a, b)
        print(f"\n  Link N{a}<->N{b} ({dist:.1f} km):")

        # Normal — reuse existing results if available
        if (a, b) in existing_lookup:
            r_safe = existing_lookup[(a, b)]
            print(f"    Without Eve ... |S|={abs(r_safe['chsh_S']):.3f}  "
                  f"QBER={r_safe['qber']*100:.2f}%  (reused from network sim)")
        else:
            print(f"    Without Eve ... ", end="", flush=True)
            r_safe = E91LinkSim.run_link(a, b, cfg)
            print(f"|S|={abs(r_safe['chsh_S']):.3f}  QBER={r_safe['qber']*100:.2f}%")
        safe_results.append(r_safe)

        # With Eve (reduce visibility to 0.50)
        eve_cfg = SimConfig(
            nodecount=cfg.nodecount, topology=cfg.topology,
            num_pairs=cfg.num_pairs, duration=cfg.duration,
            channeltype=cfg.channeltype, fiber=cfg.fiber, fso=cfg.fso,
            linkdistances=cfg.linkdistances, nodetrust=cfg.nodetrust,
        )
        if eve_cfg.channeltype == ChannelType.FIBER:
            eve_cfg.fiber = FiberPhysics(
                attenuation=cfg.fiber.attenuation,
                detector_eff=cfg.fiber.detector_eff,
                dark_count_prob=cfg.fiber.dark_count_prob,
                base_visibility=0.50,
                vis_decay_per_km=0.0,
            )
        else:
            eve_cfg.fso = FSOPhysics(
                wavelength=cfg.fso.wavelength,
                tx_aperture=cfg.fso.tx_aperture,
                rx_aperture=cfg.fso.rx_aperture,
                beam_divergence=cfg.fso.beam_divergence,
                pointing_error=cfg.fso.pointing_error,
                detector_eff=cfg.fso.detector_eff,
                dark_count_prob=cfg.fso.dark_count_prob,
                visibility_km=cfg.fso.visibility_km,
                base_visibility=0.50,
                vis_decay_per_km=0.0,
            )
        print(f"    With Eve    ... ", end="", flush=True)
        r_eve = E91LinkSim.run_link(a, b, eve_cfg)
        print(f"|S|={abs(r_eve['chsh_S']):.3f}  QBER={r_eve['qber']*100:.2f}%")
        eve_results.append(r_eve)

    return safe_results, eve_results


# ============================================================================
# DISTANCE SWEEP
# ============================================================================

def run_distance_sweep(cfg: SimConfig, distances=None):
    if distances is None:
        distances = [1, 5, 10, 20, 30, 50, 75, 100]
    results = []
    for d in distances:
        print(f"  {d:6.1f} km ... ", end="", flush=True)
        sweep_cfg = SimConfig(
            nodecount=2, topology="linear",
            num_pairs=cfg.num_pairs, duration=cfg.duration,
            channeltype=cfg.channeltype, fiber=cfg.fiber, fso=cfg.fso,
            linkdistances={(0, 1): d},
        )
        r = E91LinkSim.run_link(0, 1, sweep_cfg)
        print(f"QBER={r['qber']*100:5.2f}%  |S|={abs(r['chsh_S']):.3f}  "
              f"Key={r['securelength']}")
        results.append(r)
    return results


# ============================================================================
# VISUALISATION
# ============================================================================

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   Saved: {path}")


def draw_network(results, cfg, title_extra="", e2eresults=None,
                 savepath="network.png"):
    g = nx.Graph()
    g.add_nodes_from(range(cfg.nodecount))
    edge_colors, edge_labels = [], {}

    for r in results:
        if not r.get("success"):
            continue
        u, v = r["link"]
        rate = r.get("secretrate", 0)
        q = r.get("qber", 0)
        s = abs(r.get("chsh_S", 0))
        g.add_edge(u, v, weight=rate)
        edge_labels[(u, v)] = f"{rate:.0f} bps\nQBER:{q*100:.1f}%\n|S|={s:.2f}"
        if q >= 0.11 or not r.get("issecure"):
            edge_colors.append("red")
        elif q > 0.05:
            edge_colors.append("orange")
        else:
            edge_colors.append("green")

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

    # E2E arcs
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
            off = 0.8 + (e.get("hops", 2) - 2) * 0.4
            cx, cy = mx + px * off, my + py * off
            eqber = e.get("e2eqber", 0)
            has_u = bool(e.get("untrusted"))
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
                                    edgecolor=pc, ls=":", lw=2.5, alpha=0.7, zorder=5)
            ax.add_patch(pp)
            lx = 0.25 * x1 + 0.5 * cx + 0.25 * x2
            ly = 0.25 * y1 + 0.5 * cy + 0.25 * y2
            if e.get("success") and not has_u:
                lbl = (f"N{s}<->N{d} (E2E)\n{e.get('e2erate',0):.0f} bps\n"
                       f"QBER:{eqber*100:.1f}%\n({e['hops']} hops)")
            else:
                lbl = f"N{s}<->N{d} (E2E)\nFAILED\n({e.get('hops',0)} hops)"
            ax.annotate(lbl, xy=(lx, ly), fontsize=9, color=pc, ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                                  edgecolor=pc, alpha=0.9), zorder=10)

    nc = ["salmon" if cfg.get_trust(i) == NodeTrust.UNTRUSTED else "lightblue"
          for i in range(cfg.nodecount)]
    nx.draw_networkx_nodes(g, pos, node_color=nc, node_size=1500,
                           edgecolors="darkblue", linewidths=2, ax=ax)
    nl = {i: f"N{i}{'*' if cfg.get_trust(i)==NodeTrust.UNTRUSTED else ''}"
          for i in range(cfg.nodecount)}
    nx.draw_networkx_labels(g, pos, nl, font_size=14, font_weight="bold", ax=ax)
    if edge_colors:
        nx.draw_networkx_edges(g, pos, edgelist=list(g.edges()), width=4,
                               edge_color=edge_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels, font_size=8,
                                  bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor="white", edgecolor="gray",
                                            alpha=0.9), ax=ax)
    ax.set_title(
        f"E91 QKD Network ({cfg.topology.upper()}, "
        f"{cfg.channeltype.value.upper()}) [QuTiP]\n{title_extra}",
        fontsize=12, fontweight="bold")
    leg = [Patch(facecolor="green", label="Secure (QBER<5%)"),
           Patch(facecolor="orange", label="Marginal (5-11%)"),
           Patch(facecolor="red", label="Insecure (>=11%)"),
           Patch(facecolor="lightblue", edgecolor="darkblue", label="Trusted"),
           Patch(facecolor="salmon", edgecolor="darkblue", label="Untrusted"),
           Line2D([0], [0], color="green", lw=4, label="Direct Link"),
           Line2D([0], [0], color="purple", lw=2.5, ls=":", label="E2E Multi-Hop")]
    ax.legend(handles=leg, loc="upper left", fontsize=9)
    ax.margins(0.15); ax.axis("off"); fig.tight_layout()
    _save(fig, savepath)


def draw_qber(results, cfg, e2eresults=None, savepath="qber_bar.png"):
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
                labels.append(f"N{e['source']}<->N{e['dest']}\n(E2E)")
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
    for bar, d, v in zip(bars, dists, qbers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{d:.1f}km", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=11, color="red", ls="--", lw=2, label="Security Threshold (11%)")
    ax.set_ylabel("QBER (%)")
    ax.set_xlabel("Link / Path")
    ax.set_title(f"E91 QBER ({cfg.channeltype.value.upper()} Channel) [QuTiP]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)


def draw_chsh(results, cfg, savepath="chsh_bar.png"):
    labels, s_vals, secure = [], [], []
    for r in results:
        if not r.get("success"):
            continue
        labels.append(f"N{r['link'][0]}<->N{r['link'][1]}\n({r['distkm']:.0f}km)")
        s_vals.append(abs(r.get("chsh_S", 0)))
        secure.append(r.get("issecure", False))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    colors = ["green" if s > 2 else "red" for s in s_vals]
    bars = ax.bar(x, s_vals, color=colors, edgecolor="black")
    for b, v in zip(bars, s_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(y=2.0, color="red", ls="--", lw=2, label="Classical bound (2.0)")
    ax.axhline(y=2 * math.sqrt(2), color="blue", ls=":", lw=2,
               label=f"Tsirelson ({2*math.sqrt(2):.3f})")
    ax.set_ylabel("|S|")
    ax.set_xlabel("Link")
    ax.set_title("E91 CHSH Bell Test per Link [QuTiP]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 3.2)
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)


def draw_rates(results, cfg, e2eresults=None, savepath="rates.png"):
    labels, sifted, secret, types = [], [], [], []
    for r in results:
        if not r.get("success"):
            continue
        labels.append(f"N{r['link'][0]}<->N{r['link'][1]}")
        sifted.append(r.get("siftedrate", 0))
        secret.append(r.get("secretrate", 0))
        types.append("direct")
    if e2eresults:
        for e in e2eresults:
            if e.get("success") and e.get("hops", 0) > 1:
                labels.append(f"N{e['source']}<->N{e['dest']}\n(E2E)")
                sifted.append(e.get("e2esiftedbits", 0) / cfg.duration)
                secret.append(e.get("e2esecbits", 0) / cfg.duration)
                types.append("e2e")
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    w = 0.35
    sc = ["lightblue" if t == "direct" else "plum" for t in types]
    rc = ["green" if s > 0 else "red" for s in secret]
    ax.bar(x - w / 2, sifted, w, label="Sifted Rate", color=sc, edgecolor="blue")
    ax.bar(x + w / 2, secret, w, label="Secret Rate", color=rc, edgecolor="black")
    ax.set_ylabel("Rate (bps)")
    ax.set_xlabel("Link / Path")
    ax.set_title(f"E91 Key Rates ({cfg.channeltype.value.upper()}) [QuTiP]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    _save(fig, savepath)


def draw_correlations(r, savepath="correlations.png"):
    """Heatmap of simulated vs theoretical E(a_i, b_j) for one link."""
    V = r.get("visibility", 0.98)
    sim_m = np.zeros((3, 3))
    theo_m = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            sim_m[i, j] = r["correlations"].get((i, j), 0)
            theo_m[i, j] = V * math.cos(2 * (ALICE_ANGLES[i] - BOB_ANGLES[j]))

    a_labels = [f"a{i+1}={ALICE_ANGLES[i]*180/math.pi:.1f}" for i in range(3)]
    b_labels = [f"b{j+1}={BOB_ANGLES[j]*180/math.pi:.1f}" for j in range(3)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, matrix, title in [
        (ax1, sim_m, "Simulated (QuTiP Born Rule)"),
        (ax2, theo_m, f"Theoretical V*cos(2(thA-thB)), V={V:.2f}"),
    ]:
        im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                tag = "\n(KEY)" if (i, j) in KEY_BASIS_PAIRS else ""
                ax.text(j, i, f"{matrix[i,j]:+.3f}{tag}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold" if (i, j) in KEY_BASIS_PAIRS else "normal")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(b_labels)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(a_labels)
        ax.set_xlabel("Bob's basis"); ax.set_ylabel("Alice's basis")
        fig.colorbar(im, ax=ax, label="E(a,b)", shrink=0.8)
    fig.suptitle(f"E91 Correlations: N{r['link'][0]}<->N{r['link'][1]} "
                 f"({r['distkm']:.0f} km)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, savepath)


def draw_eavesdropper(safe_results, eve_results, savepath="eavesdropper.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n = len(safe_results)
    labels_s = [f"N{r['link'][0]}-N{r['link'][1]}\nNo Eve" for r in safe_results]
    labels_e = [f"N{r['link'][0]}-N{r['link'][1]}\nWith Eve" for r in eve_results]
    labels = []
    for ls, le in zip(labels_s, labels_e):
        labels.extend([ls, le])

    # CHSH
    ax = axes[0]
    vals, cols = [], []
    for rs, re in zip(safe_results, eve_results):
        vs = abs(rs.get("chsh_S", 0))
        ve = abs(re.get("chsh_S", 0))
        vals.extend([vs, ve])
        cols.extend(["green" if vs > 2 else "red", "green" if ve > 2 else "red"])
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=cols, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(2.0, color="red", ls="--", lw=2, label="Classical bound")
    ax.axhline(2 * math.sqrt(2), color="blue", ls=":", lw=2, label="Tsirelson")
    ax.set_ylabel("|S|"); ax.set_ylim(0, 3.2); ax.set_title("CHSH Bell Test")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.legend(fontsize=8)

    # QBER
    ax = axes[1]
    vals, cols = [], []
    for rs, re in zip(safe_results, eve_results):
        qs = rs.get("qber", 0) * 100
        qe = re.get("qber", 0) * 100
        vals.extend([qs, qe])
        cols.extend(["green" if qs < 11 else "red", "green" if qe < 11 else "red"])
    bars = ax.bar(np.arange(len(vals)), vals, color=cols, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                f"{v:.1f}%", ha="center", fontsize=8)
    ax.axhline(11, color="red", ls="--", lw=2, label="Threshold 11%")
    ax.set_ylabel("QBER (%)"); ax.set_title("QBER")
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.legend()

    # Secure key
    ax = axes[2]
    vals, cols = [], []
    for rs, re in zip(safe_results, eve_results):
        ks = rs.get("securelength", 0)
        ke = re.get("securelength", 0)
        vals.extend([ks, ke])
        cols.extend(["green" if ks > 0 else "red", "green" if ke > 0 else "red"])
    bars = ax.bar(np.arange(len(vals)), vals, color=cols, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(max(vals), 1) * 0.01,
                f"{v:,}", ha="center", fontsize=8)
    ax.set_ylabel("Secure Key (bits)"); ax.set_title("Secure Key Length")
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels, fontsize=7, rotation=45)

    fig.suptitle("E91 Eavesdropper Detection via CHSH Inequality [QuTiP]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, savepath)


def draw_distance_sweep(results, savepath="distance.png"):
    dists = [r["distkm"] for r in results]
    qbers = [r["qber"] * 100 for r in results]
    chsh  = [abs(r["chsh_S"]) for r in results]
    keys  = [r["securelength"] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax1.plot(dists, qbers, "o-", color="steelblue", lw=2, ms=8)
    ax1.axhline(11, color="red", ls="--", lw=2, label="Threshold 11%")
    ax1.set_ylabel("QBER (%)"); ax1.set_title("QBER vs Distance")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(dists, chsh, "s-", color="purple", lw=2, ms=8)
    ax2.axhline(2.0, color="red", ls="--", lw=2, label="Classical bound")
    ax2.axhline(2 * math.sqrt(2), color="blue", ls=":", lw=2, label="Tsirelson")
    ax2.set_ylabel("|S|"); ax2.set_title("CHSH Value vs Distance")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    cols = ["green" if k > 0 else "red" for k in keys]
    w = max(1, (max(dists) - min(dists)) / len(dists) * 0.6) if len(dists) > 1 else 2
    ax3.bar(dists, keys, width=w, color=cols, edgecolor="black")
    ax3.set_ylabel("Secure Key (bits)"); ax3.set_xlabel("Distance (km)")
    ax3.set_title("Secure Key vs Distance"); ax3.grid(True, alpha=0.3)

    # Reserve top margin for suptitle so it does not overlap subplot titles.
    fig.suptitle("E91 QKD Distance Analysis (QuTiP)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, savepath)


def draw_e2e_rates(e2eresults, cfg, savepath="e2e_rates.png"):
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
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                 f"{h} hops\n{d:.0f}km", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("E2E Secret Rate (bps)"); ax1.set_xlabel("Node Pair")
    ax1.set_title("End-to-End Key Rates"); ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)

    c2 = ["green" if q < 5 else "orange" if q < 11 else "red" for q in qbers]
    bars2 = ax2.bar(x, qbers, color=c2, edgecolor="black")
    ax2.axhline(y=11, color="red", ls="--", lw=2, label="Threshold")
    for b, h in zip(bars2, hops):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 f"{h} hops", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("E2E QBER (%)"); ax2.set_xlabel("Node Pair")
    ax2.set_title("E2E QBER (XOR Combined)"); ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45); ax2.legend()
    fig.suptitle("E91 End-to-End Analysis (XOR Trusted Relay) [QuTiP]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, savepath)


def draw_xor_relay(e2eresult, linkresults, cfg, savepath="xor_relay.png"):
    """
    Visualise the XOR trusted-relay key exchange step by step.

    Shows: Key_A on link 1, Key_B on link 2, XOR at relay node,
    and the final end-to-end shared key.
    """
    if not e2eresult.get("success") or e2eresult.get("hops", 0) < 2:
        return

    path = e2eresult["path"]
    hops = len(path) - 1

    # Collect per-link key samples from linkresults
    link_keys = []
    link_lookup = {}
    for r in linkresults:
        if r.get("success"):
            link_lookup[tuple(r["link"])] = r
            link_lookup[tuple(reversed(r["link"]))] = r

    for i in range(hops):
        a, b = path[i], path[i + 1]
        r = link_lookup.get((a, b)) or link_lookup.get((b, a))
        if r and r.get("key_alice"):
            # Use 8 bits for clean display
            ka = r["key_alice"][:8]
            kb = r["key_bob"][:8]
            link_keys.append(dict(link=(a, b), key_alice=ka, key_bob=kb,
                                  qber=r.get("qber", 0),
                                  distance=r.get("distkm", 0)))
        else:
            link_keys.append(dict(link=(a, b),
                                  key_alice=[0]*8, key_bob=[0]*8,
                                  qber=0, distance=0))

    n_bits = 8

    # Compute XOR relay chain
    # Step 1: source holds K1 (alice side of link 0)
    # Step 2: relay node XORs K1 ^ K2 for each hop
    # Step 3: destination recovers end-to-end key
    relay_keys = []
    # Key at source = alice's key on the first link
    e2e_key = list(link_keys[0]["key_alice"][:n_bits])
    relay_keys.append(dict(node=f"N{path[0]} (Source)",
                           key=list(e2e_key), label="K_source"))
    for i in range(hops):
        # Bob's key at this link (held by relay node path[i+1])
        k_link = list(link_keys[i]["key_bob"][:n_bits])
        if i < hops - 1:
            # Intermediate relay: XOR with next link's alice key
            k_next = list(link_keys[i + 1]["key_alice"][:n_bits])
            xor_result = [b1 ^ b2 for b1, b2 in zip(k_link, k_next)]
            relay_keys.append(dict(
                node=f"N{path[i+1]} (Relay)",
                key_in1=k_link,
                key_in2=k_next,
                key=xor_result,
                label=f"K{i+1} XOR K{i+2}"))
            e2e_key = xor_result
        else:
            # Final destination
            relay_keys.append(dict(
                node=f"N{path[i+1]} (Dest)",
                key=k_link,
                label="K_dest"))

    # Also compute the end-to-end XOR of all keys
    final_src = list(link_keys[0]["key_alice"][:n_bits])
    final_dst = list(link_keys[-1]["key_bob"][:n_bits])
    e2e_xor = [a ^ b for a, b in zip(final_src, final_dst)]
    e2e_match = sum(1 for a, b in zip(final_src, final_dst) if a == b)

    # ── Precompute QBER steps for the formula box ──
    qber_steps = e2eresult.get("qbersteps", [])
    n_qsteps = max(len(qber_steps), 1)
    qber_box_h = 1.4 + n_qsteps * 0.5

    # ── Drawing ──
    fig_height = max(8, 2.5 * hops + 4 + qber_box_h + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1 - qber_box_h - 1.5, hops * 2.5 + 4)
    ax.axis("off")

    y_top = hops * 2.5 + 2.5

    # Title
    src, dst = path[0], path[-1]
    route_str = " -> ".join(f"N{n}" for n in path)
    ax.text(7, y_top + 0.8,
            f"XOR Trusted Relay: N{src} to N{dst}",
            fontsize=16, fontweight="bold", ha="center", va="center")
    ax.text(7, y_top + 0.2,
            f"Route: {route_str}  |  Hops: {hops}  |  "
            f"E2E QBER: {e2eresult.get('e2eqber', 0)*100:.1f}%",
            fontsize=11, ha="center", va="center", color="gray")
    # (QBER derivation is shown in the dedicated box at the bottom)

    def draw_bits(ax, x, y, bits, color="black", bg="white",
                  highlight_errors=None, label=None):
        """Draw a row of bit boxes."""
        for j, bit in enumerate(bits):
            ec = "red" if highlight_errors and highlight_errors[j] else "gray"
            lw = 2.5 if highlight_errors and highlight_errors[j] else 1
            fc = "#ffe0e0" if highlight_errors and highlight_errors[j] else bg
            rect = mpatches.FancyBboxPatch(
                (x + j * 0.55, y - 0.22), 0.45, 0.44,
                boxstyle="round,pad=0.05", facecolor=fc,
                edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)
            ax.text(x + j * 0.55 + 0.225, y,
                    str(bit), fontsize=12, fontweight="bold",
                    ha="center", va="center", color=color,
                    fontfamily="monospace")
        if label:
            ax.text(x - 0.3, y, label, fontsize=10, ha="right",
                    va="center", fontweight="bold", color="darkblue")

    # Draw each hop
    for i in range(hops):
        y = y_top - 1.2 - i * 2.5
        a, b = link_keys[i]["link"]
        ka = link_keys[i]["key_alice"][:n_bits]
        kb = link_keys[i]["key_bob"][:n_bits]
        qber = link_keys[i]["qber"]
        dist = link_keys[i]["distance"]

        # Errors between alice and bob keys
        errors = [1 if ka[j] != kb[j] else 0 for j in range(n_bits)]

        # Link header
        ax.text(0.5, y + 0.7,
                f"Link {i+1}: N{a} <-> N{b}  "
                f"({dist:.0f} km, QBER={qber*100:.1f}%)",
                fontsize=11, fontweight="bold", color="darkgreen",
                va="center")

        # Alice key
        draw_bits(ax, 2.5, y, ka, color="blue", bg="#e0e8ff",
                  label=f"N{a}:")

        # Bob key
        draw_bits(ax, 2.5, y - 0.65, kb, color="red", bg="#ffe8e0",
                  highlight_errors=errors, label=f"N{b}:")

        # Arrow between hops
        if i < hops - 1:
            ax.annotate("", xy=(7, y - 1.4), xytext=(7, y - 1.0),
                        arrowprops=dict(arrowstyle="->", color="purple",
                                        lw=2.5))
            ax.text(8.5, y - 1.2, "XOR relay at "
                    f"N{path[i+1]}",
                    fontsize=10, color="purple", fontweight="bold",
                    va="center")

    # ── Final E2E summary box ──
    y_bottom = y_top - 1.2 - hops * 2.5
    box = mpatches.FancyBboxPatch(
        (1, y_bottom - 0.2), 12, 1.8,
        boxstyle="round,pad=0.3", facecolor="#f0f0ff",
        edgecolor="darkblue", linewidth=2.5)
    ax.add_patch(box)

    ax.text(7, y_bottom + 1.2,
            f"End-to-End Key: N{src} <-> N{dst}",
            fontsize=13, fontweight="bold", ha="center", va="center",
            color="darkblue")

    draw_bits(ax, 2.5, y_bottom + 0.6, final_src, color="blue",
              bg="#d0d8ff", label="Src:")
    draw_bits(ax, 2.5, y_bottom, final_dst, color="red",
              bg="#ffd8d0",
              highlight_errors=[1 if final_src[j] != final_dst[j]
                                else 0 for j in range(n_bits)],
              label="Dst:")

    match_pct = e2e_match / n_bits * 100
    status = "SECURE" if e2eresult.get("issecure") else "INSECURE"
    sc = "green" if e2eresult.get("issecure") else "red"
    ax.text(8.5, y_bottom + 0.3,
            f"Match: {e2e_match}/{n_bits} ({match_pct:.0f}%)  |  "
            f"Status: {status}",
            fontsize=11, fontweight="bold", ha="left", va="center",
            color=sc)

    # Legend
    leg_y = y_bottom - 0.8
    ax.text(1.5, leg_y,
            "Blue = Alice's bits    Red = Bob's bits    "
            "Pink cell = bit error    Arrow = XOR relay hop",
            fontsize=9, color="gray", va="center")

    # ── E2E QBER Calculation Box ──
    qb_top = leg_y - 0.8
    qb_h = qber_box_h
    qber_box = mpatches.FancyBboxPatch(
        (0.5, qb_top - qb_h), 13, qb_h,
        boxstyle="round,pad=0.3", facecolor="#fffbe6",
        edgecolor="#b8860b", linewidth=2)
    ax.add_patch(qber_box)

    # Header
    ax.text(7, qb_top - 0.25,
            "E2E QBER Calculation  (XOR Trusted Relay)",
            fontsize=13, fontweight="bold", ha="center", va="center",
            color="#8b4513")

    # Formula
    ax.text(7, qb_top - 0.7,
            r"Formula:  $Q_{combined} = Q_1 + Q_2 - 2 \cdot Q_1 \cdot Q_2$",
            fontsize=12, ha="center", va="center", color="black")

    # Step-by-step calculations
    y_step = qb_top - 1.15
    if len(qber_steps) >= 1:
        # Show first link
        q0 = qber_steps[0]["input_qber"]
        a0, b0 = link_keys[0]["link"]
        ax.text(1.5, y_step,
                f"Step 0:  Link 1 (N{a0}-N{b0})  QBER = {q0*100:.2f}%",
                fontsize=11, ha="left", va="center", color="#333333",
                fontfamily="monospace")
        y_step -= 0.5

        # Show each combination step
        for i in range(1, len(qber_steps)):
            s = qber_steps[i]
            qi = s["input_qber"]
            q_prev = qber_steps[i-1]["cumulative_qber"]
            q_comb = s["cumulative_qber"]
            a_i, b_i = link_keys[i]["link"]
            ax.text(1.5, y_step,
                    f"Step {i}:  Q_prev={q_prev*100:.2f}%  +  "
                    f"Link {i+1} (N{a_i}-N{b_i}) Q={qi*100:.2f}%  "
                    f"-  2({q_prev*100:.2f}%)({qi*100:.2f}%)  =  "
                    f"{q_comb*100:.2f}%",
                    fontsize=11, ha="left", va="center", color="#333333",
                    fontfamily="monospace",
                    fontweight="bold" if i == len(qber_steps) - 1 else "normal")
            y_step -= 0.5

    # Final result
    final_qber = e2eresult.get("e2eqber", 0)
    secure_str = "SECURE (< 11%)" if final_qber < 0.11 else "INSECURE (>= 11%)"
    res_color = "green" if final_qber < 0.11 else "red"
    ax.text(7, y_step,
            f"E2E QBER = {final_qber*100:.2f}%   =>   {secure_str}",
            fontsize=12, fontweight="bold", ha="center", va="center",
            color=res_color)

    fig.tight_layout()
    _save(fig, savepath)


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

def print_link_results(cfg, results, elapsed):
    print("\n" + "=" * 70)
    print("              E91 QKD NETWORK SIMULATION RESULTS")
    print("              Quantum Backend: QuTiP (Born Rule)")
    print("=" * 70)
    print(f"  Protocol:      E91 (Ekert 1991)")
    print(f"  Channel:       {cfg.channeltype.value.upper()}")
    print(f"  Nodes:         {cfg.nodecount}")
    print(f"  Topology:      {cfg.topology}")
    print(f"  Pairs/link:    {cfg.num_pairs}")
    print(f"  Simulator:     QuTiP (density matrix + projective measurement)")

    print("\n --- LINK RESULTS ---")
    tot_sec, tot_q, cnt = 0, 0.0, 0
    for r in results:
        if r.get("success"):
            lk = r["link"]
            print(f"\n  Link: N{lk[0]} <-> N{lk[1]} ({r['distkm']:.1f} km)")
            print(f"    Transmission:    {r['transmission']:.4e}")
            print(f"    Visibility:      {r['visibility']:.4f}")
            print(f"    State Fidelity:  {r['state_fidelity']:.4f}")
            print(f"    State Purity:    {r['state_purity']:.4f}")
            print(f"    Pairs gen/rec:   {r['pairsgen']} / {r['pairsrec']}")
            print(f"    Sifted Key:      {r['siftedlength']} bits")
            print(f"    QBER:            {r['qber']*100:.2f}%")
            print(f"    CHSH |S|:        {abs(r['chsh_S']):.4f}")
            print(f"    Bell violated:   {'YES' if r['chsh_violated'] else 'NO'}")
            print(f"    Secure Bits:     {r['securelength']}")
            print(f"    Secret Rate:     {r['secretrate']:.1f} bps")
            print(f"    Status:          {'SECURE [OK]' if r['issecure'] else 'INSECURE [X]'}")
            # Correlations
            print(f"    Correlations:")
            for i in range(3):
                row = "      "
                for j in range(3):
                    val = r['correlations'].get((i, j), 0)
                    tag = "*" if (i, j) in KEY_BASIS_PAIRS else " "
                    row += f"E(a{i+1},b{j+1})={val:+.3f}{tag} "
                print(row)
            print("      (* = key-generating basis pair)")
            # Key sample
            if r.get("key_alice"):
                a_str = ''.join(str(b) for b in r['key_alice'])
                b_str = ''.join(str(b) for b in r['key_bob'])
                match = ''.join('=' if a == b else 'X'
                                for a, b in zip(r['key_alice'], r['key_bob']))
                print(f"    Key sample:  A={a_str}")
                print(f"                 B={b_str}")
                print(f"                 M={match}")
            tot_sec += r.get("securelength", 0)
            tot_q += r.get("qber", 0)
            cnt += 1
        else:
            lk = r.get("link", ("?", "?"))
            print(f"\n  Link: N{lk[0]} <-> N{lk[1]} -- FAILED")

    print("\n --- SUMMARY ---")
    if cnt:
        print(f"  Total Secure Bits: {tot_sec}")
        print(f"  Avg QBER:          {tot_q/cnt*100:.2f}%")
        print(f"  Net Rate:          {tot_sec/cfg.duration:.1f} bps")
    print(f"  Runtime:           {elapsed:.3f} s")
    print("=" * 70)


def print_e2e(e2e, cfg):
    print("\n" + "=" * 70)
    print("              E91 END-TO-END KEY EXCHANGE RESULTS")
    print("              (XOR Trusted Relay Protocol)")
    print("=" * 70)
    for r in e2e:
        print(f"\n  Path: N{r['source']} -> N{r['dest']}")
        print(f"    Route:       {' -> '.join(f'N{n}' for n in r.get('path',[]))}")
        print(f"    Hops:        {r.get('hops',0)}")
        print(f"    Distance:    {r.get('totaldistance',0):.1f} km")
        if r.get("success"):
            print(f"    Link Rates:  {[f'{x:.0f}' for x in r.get('linkrates',[])]}")
            print(f"    Link QBERs:  {[f'{x*100:.2f}%' for x in r.get('linkqbers',[])]}")
            print(f"    Bottleneck:  Link {r.get('bottlenecklink',-1)} "
                  f"({r.get('bottleneckrate',0):.1f} bps)")
            print(f"    E2E Rate:    {r.get('e2erate',0):.1f} bps")
            print(f"    E2E QBER:    {r.get('e2eqber',0)*100:.2f}%")
            if r.get("qbersteps"):
                chain = " -> ".join(f"{s['cumulative_qber']*100:.2f}%"
                                    for s in r["qbersteps"])
                print(f"    QBER Chain:  {chain}")
            print(f"    E2E Secure:  {r.get('e2esecbits',0)} bits")
            print(f"    Status:      {'SECURE [OK]' if r.get('issecure') else 'NEEDS REVIEW'}")
        else:
            print(f"    Status:      FAILED")
            if r.get("warning"):
                print(f"    [!] {r['warning']}")
    print("=" * 70)


# ============================================================================
# MAIN INTERACTIVE
# ============================================================================

def main():
    outdir = os.path.join(os.getcwd(), "e91_outputs")
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 70)
    print("       E91 QKD NETWORK SIMULATOR -- QuTiP Backend")
    print("       Density Matrix | Born Rule | CHSH Bell Test")
    print("       Multi-Node | Multi-Topology | Full Visualisation")
    print("=" * 70)

    print("\n Select Simulation Mode:")
    print("  [1] Standard Link Simulation (Mesh)")
    print("  [2] End-to-End Multi-Hop (XOR Trusted Relay -- Linear/Star)")
    print("  [3] Eavesdropper Detection Demo")
    print("  [4] Distance Sweep Analysis")
    print("  [5] Run All")

    mode = input("\n Mode (1-5, default 5): ").strip()
    mode = int(mode) if mode.isdigit() and 1 <= int(mode) <= 5 else 5

    # Channel
    print("\n Select Channel:")
    print("  [1] Fiber Optic")
    print("  [2] Free-Space Optical (Ground)")
    c = input(" Choice (1/2, default 1): ").strip()
    channeltype = ChannelType.FSO_GROUND if c == "2" else ChannelType.FIBER

    # Nodes
    if mode == 4:
        nnodes = 2
    else:
        n = input(f" Nodes (2-10, default 4): ").strip()
        nnodes = max(2, min(10, int(n) if n.isdigit() else 4))

    # Topology
    if mode == 1 or mode == 3:
        topo = "mesh"
        print(f" Topology: Mesh")
    elif mode == 4:
        topo = "linear"
    elif mode == 2:
        print(" [1] Linear  [2] Star")
        t = input(" Topology (1/2, default 1): ").strip()
        topo = "star" if t == "2" else "linear"
    else:
        print(" [1] Linear  [2] Star  [3] Mesh")
        t = input(" Topology (1-3, default 1): ").strip()
        topo = {"1": "linear", "2": "star", "3": "mesh"}.get(t, "linear")

    # Distances
    links = get_links(topo, nnodes)
    dists = {}
    if mode != 4:
        print("\n Enter link distances (km):")
        for a, b in links:
            d = input(f"  N{a}<->N{b} (default 10): ").strip()
            dists[(a, b)] = float(d) if d else 10.0

    # Pairs
    n = input(f" Entangled pairs per link (default 10000): ").strip()
    num_pairs = int(n) if n.isdigit() and int(n) > 0 else 10000

    # Seed
    s = input(" Random seed (default 42): ").strip()
    seed = int(s) if s.isdigit() else 42

    cfg = SimConfig(
        nodecount=nnodes, topology=topo, num_pairs=num_pairs,
        channeltype=channeltype, linkdistances=dists, seed=seed,
    )

    # Ensure deterministic/reproducible output for same config + seed.
    set_seed(cfg.seed)

    print(f"\n{'='*70}")
    print(f" Running E91 via QuTiP ...")
    print(f"{'='*70}")
    start = time.time()

    # ── Run the base network simulation ONCE ──
    # All modes that need link results will share this single run.
    results = None
    if mode in (1, 2, 3, 5):
        print("\n --- BASE NETWORK SIMULATION ---")
        sim = E91NetworkSim(cfg)
        results = sim.runsim()
        elapsed = time.time() - start
        print_link_results(cfg, results, elapsed)

    # ── Mode 1/5: Link-level plots ──
    if mode in (1, 5) and results:
        draw_network(results, cfg, savepath=f"{outdir}/network.png")
        draw_qber(results, cfg, savepath=f"{outdir}/qber_bar.png")
        draw_chsh(results, cfg, savepath=f"{outdir}/chsh_bar.png")
        draw_rates(results, cfg, savepath=f"{outdir}/rates.png")
        draw_correlations(results[0], savepath=f"{outdir}/correlations.png")

    # ── Mode 2/5: Multi-Hop E2E (reuse same link results) ──
    if mode in (2, 5) and results:
        relay = TrustedRelayNetwork(cfg)
        relay.linkresults = results  # reuse the same simulation results
        e2e = []
        for i in range(cfg.nodecount):
            for j in range(i + 1, cfg.nodecount):
                e2e.append(relay.calc_e2e(i, j))
        relay.e2eresults = e2e
        print_e2e(e2e, cfg)
        draw_network(results, cfg, "XOR Trusted Relay", e2e,
                     savepath=f"{outdir}/network_e2e.png")
        draw_qber(results, cfg, e2e, savepath=f"{outdir}/qber_e2e.png")
        draw_e2e_rates(e2e, cfg, savepath=f"{outdir}/e2e_rates.png")

        # Draw XOR relay visualization for multi-hop paths
        xor_count = 0
        for er in e2e:
            if er.get("success") and er.get("hops", 0) >= 2:
                xor_count += 1
                draw_xor_relay(er, results, cfg,
                               savepath=f"{outdir}/xor_relay_N{er['source']}_N{er['dest']}.png")
        if xor_count:
            print(f"   ({xor_count} XOR relay diagram(s) saved)")

    # ── Mode 3/5: Eavesdropper (reuse same results as 'no Eve' baseline) ──
    if mode in (3, 5):
        print("\n --- EAVESDROPPER DETECTION ---")
        safe_res, eve_res = run_eavesdropper_comparison(cfg, existing_results=results)
        print("\n === WITHOUT Eavesdropper ===")
        print_link_results(cfg, safe_res, time.time() - start)
        print("\n === WITH Eavesdropper (V -> 0.50) ===")
        print_link_results(cfg, eve_res, time.time() - start)
        draw_eavesdropper(safe_res, eve_res, savepath=f"{outdir}/eavesdropper.png")

    # ── Mode 4/5: Distance sweep (inherently separate — varies distance) ──
    if mode in (4, 5):
        print("\n --- DISTANCE SWEEP ---")
        dist_results = run_distance_sweep(cfg)
        draw_distance_sweep(dist_results, savepath=f"{outdir}/distance.png")

    runtime = time.time() - start
    print(f"\n Runtime: {runtime:.2f} s")
    print(f" Figures saved to: {outdir}/")

    # Save text results
    results_path = os.path.join(outdir, "e91_qutip_results.txt")
    with open(results_path, "w") as f:
        f.write("E91 QKD Network Simulation Results (QuTiP)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Simulator: QuTiP (density matrix + Born rule)\n")
        f.write(f"Protocol:  E91 (Ekert 1991)\n")
        f.write(f"Nodes:     {cfg.nodecount}\n")
        f.write(f"Topology:  {cfg.topology}\n")
        f.write(f"Channel:   {cfg.channeltype.value}\n")
        f.write(f"Pairs:     {cfg.num_pairs}\n")
        f.write(f"Seed:      {cfg.seed}\n")
        f.write(f"Runtime:   {runtime:.2f} s\n")
    print(f" Results text: {results_path}")

    if plt.get_fignums():
        plt.show()
    print("=" * 70)


if __name__ == "__main__":
    main()
