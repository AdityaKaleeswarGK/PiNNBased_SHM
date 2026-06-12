"""Parameterised Mode-I fracture PINN, refactored from
PINN_backend/PINN_FINAL_WALLAHI.ipynb.

The notebook solves a single hardcoded SENT-like configuration
(a=20, W=100, H_half=50, sigma=100, E=30000, plane stress) on the upper
symmetric half of an edge-cracked plate, with a Williams near-tip
enrichment whose amplitude is the learned K_I. This module keeps that
formulation but takes geometry/material/load as arguments so the CV
pipeline can drive it:

    result = solve_mode_I(a_mm=20, W_mm=100, H_half_mm=50, sigma_MPa=100)

Length-scale-dependent constants from the notebook (suppression radius 6,
J-ring radius 15, crack-face strip 0.5) are expressed relative to `a` and
`H_half` so the tuning survives a change of scale.

Units: mm and MPa everywhere -> K_I in MPa*sqrt(mm).
(1 MPa*sqrt(m) = 31.62 MPa*sqrt(mm))
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_BANK_PATH = Path(__file__).resolve().parent / "pinn_bank.json"
LEFM_MAX_RATIO = 0.30   # teammate's stated LEFM validity limit for concrete


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sent_geometry_factor(a_over_W):
    """Tada/Paris handbook polynomial for a single edge crack in tension.

    F(a/W) such that K_I = F * sigma * sqrt(pi * a). Valid for a/W <= 0.6.
    At a/W = 0.2 this gives ~1.37, matching the notebook's hardcoded value.
    """
    r = a_over_W
    return (1.122 - 0.231 * r + 10.550 * r**2
            - 21.710 * r**3 + 30.382 * r**4)


class ModeIPINN(nn.Module):
    """Smooth displacement net + Williams K_I enrichment (plane stress)."""

    def __init__(self, a, W, H_half, sigma, mu, kappa, device,
                 hidden_dim=128):
        super().__init__()
        self.a, self.W, self.H_half = a, W, H_half
        self.kappa = kappa
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        # Inverse-softplus init so epoch 0 starts near the handbook K_I
        KI_rough = sigma * math.sqrt(math.pi * a) * sent_geometry_factor(a / W)
        target_val = KI_rough / (2 * mu)
        init_val = math.log(math.exp(target_val) - 1.0 + 1e-9)
        self.KI_raw = nn.Parameter(torch.tensor([init_val], device=device,
                                                dtype=torch.float32))

    def get_KI_amp(self):
        return torch.nn.functional.softplus(self.KI_raw)

    def get_smooth(self, inputs):
        x_n = (inputs[:, 0:1] / (self.W / 2)) - 1.0
        y_n = inputs[:, 1:2] / self.H_half
        return self.net(torch.cat([x_n, y_n], dim=1))

    def forward(self, inputs):
        x_tip = inputs[:, 0:1] - self.a
        y_tip = inputs[:, 1:2]
        r = torch.sqrt(x_tip**2 + y_tip**2 + 1e-9)
        theta = torch.atan2(y_tip, x_tip)

        sqr = torch.sqrt(r / (2 * math.pi))
        w_u = sqr * torch.cos(theta / 2) * (self.kappa - 1 + 2 * torch.sin(theta / 2)**2)
        w_v = sqr * torch.sin(theta / 2) * (self.kappa + 1 - 2 * torch.cos(theta / 2)**2)

        smooth = self.get_smooth(inputs)
        KI_amp = self.get_KI_amp()
        u = smooth[:, 0:1] + KI_amp * w_u
        v = smooth[:, 1:2] + KI_amp * w_v
        return torch.cat([u, v], dim=1)


def _get_physics(model, pts, E, nu, mu):
    pts = pts.detach().requires_grad_(True)
    uv = model(pts)
    u, v = uv[:, 0:1], uv[:, 1:2]
    gu = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
    gv = torch.autograd.grad(v.sum(), pts, create_graph=True)[0]
    u_x, u_y = gu[:, 0:1], gu[:, 1:2]
    v_x, v_y = gv[:, 0:1], gv[:, 1:2]
    f = E / (1 - nu**2)
    sx = f * (u_x + nu * v_y)
    sy = f * (v_y + nu * u_x)
    sxy = mu * (u_y + v_x)
    return dict(sx=sx, sy=sy, sxy=sxy, u=u, v=v,
                u_x=u_x, v_x=v_x, pts=pts)


def solve_mode_I(a_mm, W_mm, H_half_mm, sigma_MPa,
                 E_MPa=30000.0, nu=0.2,
                 epochs=8000, lr=1e-3, seed=0,
                 device=None, log_every=1000, verbose=True):
    """Train the PINN for one crack configuration and extract K_I.

    Parameters mirror the notebook: edge crack of length `a_mm` along y=0
    in a plate of width `W_mm`, upper half-height `H_half_mm`, far-field
    tension `sigma_MPa` applied on the top edge.

    Returns a dict with K_I from the learned parameter, the J-integral,
    and the Williams displacement fit, plus the handbook reference value.
    """
    if not 0 < a_mm < 0.6 * W_mm:
        raise ValueError(
            f"a/W = {a_mm / W_mm:.2f} outside the validated range (0, 0.6) "
            "of the SENT geometry factor — increase W_mm or split the crack.")

    device = pick_device(device)
    torch.manual_seed(seed)

    E, W, H_half, a, sigma = E_MPa, W_mm, H_half_mm, a_mm, sigma_MPa
    mu = E / (2 * (1 + nu))
    kappa = (3 - nu) / (1 + nu)          # plane stress (matches notebook)

    # Notebook constants rescaled to the actual geometry
    crack_strip = 0.01 * H_half          # collocation exclusion near faces
    r_suppress = 0.3 * a                 # volumetric suppression radius
    r_J = min(0.75 * a, 0.8 * min(W - a, H_half))   # J-ring inside domain
    r_williams = 0.3 * a

    model = ModeIPINN(a, W, H_half, sigma, mu, kappa, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=max(1, epochs // 4),
                                          gamma=0.3)

    if verbose:
        print(f"[pinn] device={device.type}  a={a:.1f}mm  W={W:.1f}mm  "
              f"H/2={H_half:.1f}mm  sigma={sigma:.2f}MPa  epochs={epochs}")

    N_bc = 600
    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        # Collocation points (excluding crack-face strip)
        xc = torch.rand(4000, 1, device=device) * W
        yc = torch.rand(4000, 1, device=device) * H_half
        valid = ~((xc < a) & (yc < crack_strip)).squeeze()
        pts_c = torch.cat([xc[valid], yc[valid]], dim=1)

        pts_top = torch.cat([torch.rand(N_bc, 1, device=device) * W,
                             torch.full((N_bc, 1), H_half, device=device)], dim=1)
        pts_sym = torch.cat([torch.rand(N_bc, 1, device=device) * (W - a) + a,
                             torch.zeros(N_bc, 1, device=device)], dim=1)
        pts_crack = torch.cat([torch.rand(N_bc, 1, device=device) * a,
                               torch.zeros(N_bc, 1, device=device)], dim=1)
        pts_left = torch.cat([torch.zeros(N_bc, 1, device=device),
                              torch.rand(N_bc, 1, device=device) * H_half], dim=1)
        pts_right = torch.cat([torch.full((N_bc, 1), W, device=device),
                               torch.rand(N_bc, 1, device=device) * H_half], dim=1)

        # PDE residual
        res = _get_physics(model, pts_c, E, nu, mu)
        sx_x = torch.autograd.grad(res['sx'].sum(), res['pts'], create_graph=True)[0][:, 0:1]
        sxy_y = torch.autograd.grad(res['sxy'].sum(), res['pts'], create_graph=True)[0][:, 1:2]
        sy_y = torch.autograd.grad(res['sy'].sum(), res['pts'], create_graph=True)[0][:, 1:2]
        sxy_x = torch.autograd.grad(res['sxy'].sum(), res['pts'], create_graph=True)[0][:, 0:1]
        loss_p = torch.mean((sx_x + sxy_y)**2 + (sy_y + sxy_x)**2) / (E**2)

        # Boundary conditions
        res_t = _get_physics(model, pts_top, E, nu, mu)
        loss_t = torch.mean((res_t['sy'] - sigma)**2 + res_t['sxy']**2) / (E**2)
        res_s = _get_physics(model, pts_sym, E, nu, mu)
        loss_s = torch.mean((res_s['v'] * E)**2 + res_s['sxy']**2) / (E**2)
        res_f = _get_physics(model, pts_crack, E, nu, mu)
        loss_f = torch.mean(res_f['sy']**2 + res_f['sxy']**2) / (E**2)
        res_l = _get_physics(model, pts_left, E, nu, mu)
        loss_l = torch.mean(res_l['sx']**2 + res_l['sxy']**2) / (E**2)
        res_r = _get_physics(model, pts_right, E, nu, mu)
        loss_r = torch.mean(res_r['sx']**2 + res_r['sxy']**2) / (E**2)

        # Rigid-body pin at bottom-right (E-scale matched, not /E^2)
        pt_pin = torch.tensor([[W, 0.0]], device=device)
        loss_pin = _get_physics(model, pt_pin, E, nu, mu)['u'][0, 0]**2

        # Volumetric suppression of the smooth net near the tip
        n_sup = 300
        r_s = torch.rand(n_sup, device=device) * r_suppress
        th_s = torch.rand(n_sup, device=device) * math.pi
        pts_sup = torch.cat([(a + r_s * torch.cos(th_s)).unsqueeze(1),
                             (r_s * torch.sin(th_s)).unsqueeze(1)], dim=1)
        loss_suppress = torch.mean(model.get_smooth(pts_sup)**2)

        loss = (loss_p
                + 100 * (loss_t + loss_s + loss_f + loss_l + loss_r)
                + 500 * loss_pin
                + 75 * loss_suppress)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and epoch % log_every == 0:
            KI_cur = model.get_KI_amp().item() * 2 * mu
            print(f"[pinn] epoch {epoch:5d} | loss {loss.item():.2e} | "
                  f"KI {KI_cur:.1f}")

    model.eval()
    KI_param = model.get_KI_amp().item() * 2 * mu

    # ── K_I via J-integral on a half-ring around the tip ──────────────────
    theta = torch.linspace(0.02, math.pi - 0.02, 600, device=device)
    pts = torch.cat([(a + r_J * torch.cos(theta)).view(-1, 1),
                     (r_J * torch.sin(theta)).view(-1, 1)], dim=1)
    res = _get_physics(model, pts, E, nu, mu)
    sx, sy, sxy = res['sx'].squeeze(), res['sy'].squeeze(), res['sxy'].squeeze()
    u_x, v_x = res['u_x'].squeeze(), res['v_x'].squeeze()
    W_d = (sx**2 + sy**2 - 2 * nu * sx * sy) / (2 * E) + (1 + nu) / E * sxy**2
    nx, ny = torch.cos(theta), torch.sin(theta)
    Tx, Ty = sx * nx + sxy * ny, sxy * nx + sy * ny
    integrand = W_d * nx - (Tx * u_x + Ty * v_x)
    J_full = 2.0 * torch.trapz(integrand * r_J, theta).item()
    KI_J = math.sqrt(abs(J_full) * E) * (1 if J_full > 0 else -1)

    # ── K_I via Williams displacement fit ─────────────────────────────────
    theta_w = torch.linspace(0.1, math.pi - 0.1, 300, device=device)
    pts_w = torch.cat([(a + r_williams * torch.cos(theta_w)).view(-1, 1),
                       (r_williams * torch.sin(theta_w)).view(-1, 1)], dim=1)
    with torch.no_grad():
        v_pinn = model(pts_w)[:, 1].cpu().numpy()
    th = theta_w.cpu().numpy()
    v_unit = (1.0 / (2 * mu)) * np.sqrt(r_williams / (2 * np.pi)) * \
        np.sin(th / 2) * (kappa + 1 - 2 * np.cos(th / 2)**2)
    KI_W = float(np.dot(v_pinn, v_unit) / np.dot(v_unit, v_unit))

    F = sent_geometry_factor(a / W)
    KI_handbook = sigma * math.sqrt(math.pi * a) * F

    return {
        "inputs": {"a_mm": a, "W_mm": W, "H_half_mm": H_half,
                   "sigma_MPa": sigma, "E_MPa": E, "nu": nu,
                   "epochs": epochs, "device": device.type},
        "KI_param_MPa_sqrt_mm": round(KI_param, 2),
        "KI_J_integral_MPa_sqrt_mm": round(KI_J, 2),
        "KI_williams_fit_MPa_sqrt_mm": round(KI_W, 2),
        "J_MPa_mm": round(J_full, 5),
        "KI_handbook_MPa_sqrt_mm": round(KI_handbook, 2),
        "geometry_factor_F": round(F, 3),
        "rel_error_vs_handbook_pct": round(
            abs(KI_handbook - KI_param) / KI_handbook * 100, 2),
    }


# ── F(a/W) reference bank + scaling-law evaluation ────────────────────────
#
# By LEFM similarity, one PINN solve at a given a/W is the SAME
# dimensionless problem for every crack with that ratio, at any physical
# size and load: K_I = F(a/W) * sigma * sqrt(pi*a). So the expensive
# training happens once per ratio (build_bank.py, offline) and every
# detected crack evaluates in microseconds via interpolation of F.

def load_bank(path=DEFAULT_BANK_PATH):
    """Load the trained F(a/W) bank, or None if it hasn't been built."""
    path = Path(path)
    if not path.exists():
        return None
    bank = json.loads(path.read_text())
    bank["entries"].sort(key=lambda e: e["a_over_W"])
    return bank


def geometry_factor(a_over_W, bank=None):
    """Return (F, source): PINN-bank interpolation if covered, else Tada."""
    if bank and bank.get("entries"):
        ratios = [e["a_over_W"] for e in bank["entries"]]
        Fs = [e["F_param"] for e in bank["entries"]]
        if ratios[0] <= a_over_W <= ratios[-1]:
            return float(np.interp(a_over_W, ratios, Fs)), "pinn-bank"
    return sent_geometry_factor(a_over_W), "tada-handbook"


def evaluate_crack(a_mm, W_mm, sigma_MPa, angular_dev_deg=0.0, bank=None,
                   crack_type="edge"):
    """Instant K_I / K_II for one crack via the LEFM scaling law.

    `a_mm` is always the FULL measured crack length. `crack_type` selects
    the idealisation:
      "edge"   — crack starts at the member boundary: one tip driven by the
                 full length, F from the PINN bank / Tada polynomial.
      "center" — interior crack: TWO tips each driven by half the length,
                 F = sqrt(sec(pi*a_half/W)) (Feddersen/Tada centre-crack).
                 Same measured length is ~1.6x less severe than an edge crack.

    `angular_dev_deg` is the CV-measured deviation of the crack line from
    the ideal Mode-I orientation (perpendicular to the tension axis).
    Resolved-stress relations for an inclined crack:
        K_I  = F * sigma * sqrt(pi*a) * cos^2(dev)
        K_II = F * sigma * sqrt(pi*a) * cos(dev) * sin(dev)
    K_eff = sqrt(K_I^2 + K_II^2) is a simple combined-mode ranking metric.
    """
    if crack_type == "center":
        a_eff = a_mm / 2.0
        ratio = a_eff / W_mm
        # secant formula diverges at ratio=0.5; past 0.4 it is meaningless
        F = math.sqrt(1.0 / math.cos(math.pi * min(ratio, 0.4)))
        source = "center-secant"
    else:
        a_eff = a_mm
        ratio = a_eff / W_mm
        F, source = geometry_factor(ratio, bank)
    dev = math.radians(angular_dev_deg or 0.0)
    base = sigma_MPa * math.sqrt(math.pi * a_eff) * F
    KI = base * math.cos(dev) ** 2
    KII = base * math.cos(dev) * math.sin(dev)
    return {
        "a_mm": round(a_mm, 2),
        "crack_type": crack_type,
        "a_over_W": round(ratio, 4),
        "F": round(F, 4),
        "F_source": source,
        "angular_dev_deg": round(angular_dev_deg or 0.0, 2),
        "KI_MPa_sqrt_mm": round(KI, 2),
        "KII_MPa_sqrt_mm": round(KII, 2),
        "K_eff_MPa_sqrt_mm": round(math.hypot(KI, KII), 2),
        "lefm_valid": ratio <= LEFM_MAX_RATIO,
    }
