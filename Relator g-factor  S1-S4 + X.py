# =============================================================================
# Relator g-factor (CLOSED, NO-FIT): S1→S3 (scalar) + S4 (vector) + Stage-χ
# Added: TT-population switch (closed overlaps vs. paper rationals)
# Author: M.Pajuhaan
# =============================================================================
# - 100% analytic/closed. NO fitting, NO use of experimental g in computations.
# - Stage χ is computed in TWO closed ways:
#   (A) Pure log-shell lemma:
#         Δδχ_pure(m) = [δ_S3]^2 * (1/3) * ln(m/m_e)
#   (B) Mass-Hierarchy–enhanced (from the paper):
#         c_eff(1) = 1/3  (electron)
#         c_eff(n) = (1/3) * [ 1 + γ_{n,k_n} * R_n ] ,  n = 2,3
#       with
#         k_n = n(2n−1),  (k_e,k_μ,k_τ)=(1,6,15)
#         η_n = 1/(nπ),  ℓ_n = 1/(nπ√π)
#         P_IR^(χ)(ℓ) = ⟨ [1 − (1/3) f_swirl(x;ℓ)] e^{-((1−x)/ℓ)^2} ⟩_w / ⟨1⟩_w
#         ΔΛ_OUT^closed(η) = −π[ ln(1−η^4)/(2η) + atanh(η) − atan(η) ]
#         X_n = (K/(2 D_C)) * c0_gauss * P_IR^(χ)(ℓ_n),  D_C = α/π
#         Γ_geom(n) = 1/2 * sinh(η_n)/η_n
#         k_curv(n) = sinh(η_n)/η_n − 1
#         Γ_map(n)  = [ w_n k_n X_n ] / [ 1 + k_curv(n) w_n k_n X_n ]
#         γ_{n,k}   = Γ_map / (Γ_geom + Γ_map)
#         R_n       = [ P_IR^(χ)(ℓ_n)/P_IR^(χ)(ℓ_1) ] * [ |ΔΛ_OUT(η_1)| / |ΔΛ_OUT(η_n)| ]
#         Δδχ_hier(n) = [δ_S3]^2 * c_eff(n) * ln(m_n/m_e)
# =============================================================================

import mpmath as mp
mp.mp.dps = 100
pi = mp.pi

# ==================== SWITCH: TT populations source ====================
# If True: compute w_n from closed overlaps (IR/OUT/toroidal norm).
# If False: use paper rational values w_μ=11/2, w_τ=41/16 (and w_e=1).
USE_CLOSED_TT_POPULATIONS = False

# ==================== GLOBAL CONSTANTS (closed-form inputs only) ====================
alpha_inv = mp.mpf('137.035999177')   # set once in the paper
alpha     = 1/alpha_inv

# Masses (kg) — used only inside ln(m/m_e) in Δδχ; do not affect geometry.
m_e   = mp.mpf('9.1093837015e-31')
m_mu  = mp.mpf('1.883531627e-28')
m_tau = mp.mpf('3.1675406591712e-27')  # ≈1776.86 MeV/c^2

# Experimental g (FOR REPORTING ONLY; never used in computations)
g_exp_e  = mp.mpf('2.002319304360')
g_exp_mu = mp.mpf('2.0023318418')
g_exp_tau= None

# Geometry constants (fixed, closed-form)
C0_uni   = (1/pi) * (mp.mpf('4')/3 + 1/(4*pi**2))
Lambda0  = mp.log(8*mp.sqrt(pi)) - 2
c0_gauss = mp.mpf('0.5')*(mp.log(2) + mp.euler)
D_C      = alpha/pi  # D_C = α/π

# Kernel / series controls (purely numerical convergence; no fitting)
S3_MAX_M      = 20
S3_SERIES_TOL = mp.mpf('1e-34')
S4_IR_KERNEL  = 'tt-a'              # 'tt-a' (recommended) | 'tophat' | 'gaussian'
S4_TTA_MODE   = 'tangential'        # 'tangential' or 'LO'
S4_OUT_MODE   = 'series'            # 'series' (multipoles) or 'dipole'
S4_OUT_LMAX   = 31
GL_NODES      = 256

# ==================== UTILITIES ====================
def g_of_delta(d):
    return mp.mpf('2')/mp.sqrt(1-d)

def ppt_error(gp, ge):
    return (gp-ge)/ge * mp.mpf('1e12')

def safe_ppt_error(g_pred, g_ref):
    return "N/A" if g_ref is None else mp.nstr(ppt_error(g_pred, g_ref), 22)

def xi_of_alpha(a):
    # ξ(a) = 2 C0_uni a
    return 2*C0_uni*a

def sep(title=None):
    print("\n" + "="*78)
    if title:
        print(f"{title}")
        print("-"*78)

def p(name, val, d=22, note=None):
    s = f"{name:<26} = {mp.nstr(val, d)}"
    if note:
        s += f"   # {note}"
    print(s)

# ==================== S1 → S3 (scalar stack, closed) ====================
def delta_S1(a=alpha):
    # δ_S1 = α/π
    return a/pi

def delta_S2(a=alpha):
    # δ_S2 = (α/π) sqrt(1−ξ),   ξ=2 C0_uni α
    xi = xi_of_alpha(a)
    return (a/pi)*mp.sqrt(1-xi)

# series for K (m=1 piece in S3)
def _In_m1(n):
    n = mp.mpf(n)
    return (-1)**(n-1)/(((n-1)*pi)**2) + (-1)**n/(((n+1)*pi)**2)

def series_K(tol=S3_SERIES_TOL):
    # K = (2/π^2) Σ_{n≥2} [(2 I_n^(m=1))^2 / (n^2−1)]
    S=mp.mpf('0'); n=2
    while True:
        t=(2*_In_m1(n))**2/(n**2-1)
        S+=t
        if abs(t)<tol and n>50: break
        n+=1
    return (2/pi**2)*S

# L_{2m} for m≥2
def _C_S(m,a):
    if a==0: return mp.mpf(1)/(m+1), mp.mpf('0')
    C=mp.sin(a)/a; S=(1-mp.cos(a))/a
    if m==0: return C,S
    for k in range(1,m+1):
        C, S = mp.sin(a)/a - (k/a)*S, (1-mp.cos(a))/a + (k/a)*C
    return C,S

def _I_nm(n,m2):
    C1,_=_C_S(m2,(n-1)*pi); C2,_=_C_S(m2,(n+1)*pi)
    return mp.mpf('0.5')*(C1-C2)

def L_2m(m, tol=S3_SERIES_TOL, nmax=1200):
    # L_{2m} = (2/π^2) Σ_{n≥2} [(2 I_nm)^2 / (n^2−1)]
    S=mp.mpf('0')
    for n in range(2, nmax+1):
        t=(2*_I_nm(n,2*m))**2/(n**2-1); S+=t
        if n>80 and abs(t)<tol: break
    return (2/pi**2)*S

def delta_S3(a=alpha, M=S3_MAX_M):
    """
    δ_S3 = (α/π)√(1−ξ) − (α/π)(ξ/2)K − (α/π) Σ_{m≥2} (ξ/2)^m L_{2m},
    with ξ=2 C0_uni α, K=K(m=1).
    """
    xi  = xi_of_alpha(a)
    K   = series_K(tol=S3_SERIES_TOL)
    d   = (a/pi)*mp.sqrt(1 - xi) - (a/pi)*(xi/2)*K    # includes m=1 via K
    rows= []
    if M >= 2:
        for m in range(2, M+1):
            Lm  = L_2m(m)
            dlt = - (a/pi) * ((xi/2)**m) * Lm
            d  += dlt
            rows.append((m, Lm, dlt))
    return d, K, rows

# ==================== S4 (vector self-magnetic: UV→IR + OUT) ====================
def _Itot():  # ∫_0^1 x^2 sin^2(πx) dx  = 1/6 − 1/(4π^2)
    return mp.mpf('1')/6 - mp.mpf('1')/(4*pi**2)

def _f_swirl(x, ell):
    return ((1-x)**2)/(((1-x)**2) + ell**2)

def P_IR_tophat(ell):
    Itot = _Itot()
    F = lambda x: (x**3)/6 - (x**2)*mp.sin(2*pi*x)/(4*pi) - x*mp.cos(2*pi*x)/(4*pi**2) + mp.sin(2*pi*x)/(8*pi**3)
    return (F(1)-F(1-ell))/Itot

def P_IR_gaussian(ell):
    Itot = _Itot()
    w    = lambda x: (x**2)*(mp.sin(pi*x)**2)
    num  = mp.quad(lambda x: w(x)*mp.e**(-((1-x)/ell)**2), [0,1])
    return num/Itot

def P_IR_tt_A(ell, mode='tangential'):
    """
    S4 / vector-channel (A): TT increases IR weight:
      P_IR^(A)(ℓ) = ⟨ [1 + c_A f_swirl(x;ℓ)] e^{-((1-x)/ℓ)^2} ⟩_w / ⟨1⟩_w
    """
    if mode.lower() in ('lo','full','strong'):
        c = mp.mpf('2')/3      # LO transverse
    else:
        c = mp.mpf('1')/32     # mild tangential
    It  = _Itot()
    w   = lambda x: (x**2)*(mp.sin(pi*x)**2)
    num = mp.quad(lambda x: w(x)*(1 + c * _f_swirl(x, ell)) * mp.e**(-((1-x)/ell)**2), [0,1])
    return num/It

def P_IR(ell, kernel, **kwargs):
    k = kernel.lower()
    if   k == 'tophat':   return P_IR_tophat(ell)
    elif k == 'gaussian': return P_IR_gaussian(ell)
    elif k in ('tt-a','qed-tt'):
        return P_IR_tt_A(ell, mode=kwargs.get('mode','tangential'))
    else:
        raise ValueError("S4 IR kernel must be 'tophat', 'gaussian', or 'tt-a/qed-tt'.")

# ---- OUT multipole energy on r=r* (R=μ0=I=1) ----
def _B_rho(rho,z):
    rc= mp.sqrt((1+rho)**2 + z**2); k2= 4*rho/((1+rho)**2 + z**2)
    if k2<=0: return mp.mpf('0')
    if k2>=1: k2 = mp.mpf('1')-mp.mpf('1e-24')
    K=mp.ellipk(k2); E=mp.ellipe(k2); denom=(1-rho)**2+z**2
    if rho==0: return mp.mpf('0')
    return (z/(2*pi*rho*rc)) * ( -K + ((1 + rho**2 + z**2)/denom) * E )

def _B_z(rho,z):
    rc= mp.sqrt((1+rho)**2 + z**2); k2= 4*rho/((1+rho)**2 + z**2)
    if k2<=0: return mp.mpf('1')/(2*rc**3)
    if k2>=1: k2 = mp.mpf('1')-mp.mpf('1e-24')
    K=mp.ellipk(k2); E=mp.ellipe(k2); denom=(1-rho)**2+z**2
    return (1/(2*pi*rc)) * ( K + ((1 - rho**2 - z**2)/denom) * E )

def _Btheta_on_sphere_x(x,rstar):
    s=mp.sqrt(1-x**2); rho=rstar*s; z=rstar*x
    return _B_rho(rho,z)*x - _B_z(rho,z)*s

def _dPdx_leg(l,x):
    if l==0: return mp.mpf('0')
    Pl=mp.legendre(l,x); Pl1=mp.legendre(l-1,x)
    return (l/(1-x**2))*(Pl1 - x*Pl)

def _gauss_legendre(n):
    xs,ws=[],[]
    for k in range(1,n+1):
        x= mp.cos(pi*(k-mp.mpf('0.25'))/(n+mp.mpf('0.5')))
        for _ in range(60):
            Pn=mp.legendre(n,x); dPn= n/(1-x**2)*(mp.legendre(n-1,x)-x*Pn)
            dx= -Pn/dPn; x+=dx
            if abs(dx)<mp.mpf('1e-40'): break
        w = 2/((1-x**2)*(dPn**2))
        xs.append(x); ws.append(w)
    return xs,ws

def DeltaLambda_OUT(eta_eff, mode, lmax):
    # ΔΛ_OUT(η) on the observation sphere r=r* with r*=1/η.
    if mode == 'dipole':
        return - (pi/6) * (eta_eff**3)
    rstar = 1/eta_eff
    xs, ws = _gauss_legendre(GL_NODES)
    Uout = mp.mpf('0')
    lstop = lmax if (lmax % 2 == 1) else (lmax-1)
    for l in range(1, lstop+1, 2):
        Il = l*(l+1)*2/(2*l+1)
        s = mp.mpf('0')
        for i in range(len(xs)):
            s += _Btheta_on_sphere_x(xs[i], rstar) * (-(1 - xs[i]**2) * _dPdx_leg(l, xs[i])) * ws[i]
        a_l = - (rstar**(l+2)) * s / Il
        Uout += ((l+1)/(2*l+1)) * (a_l**2) * (rstar**(-(2*l+1)))
    Uout *= (2*pi)
    return - 2*Uout

def stage_S4prime(delta_scalar, K, IR_kernel=S4_IR_KERNEL, OUT_mode=S4_OUT_MODE, OUT_lmax=S4_OUT_LMAX):
    """
    One-pass S4 backreaction on top of the scalar stack:
      δ_S4 = δ_scalar + δ_A^(1) + δ_A^(2),
      ζ = (K/(2π^2)) Λ_eff,  Λ_eff = Λ_ind + ΔΛ^(UV→IR) + ΔΛ_OUT.
    Geometry: η_eff = δ_scalar/α,  ℓ_eff = η_eff/√π.
    """
    eta_eff = delta_scalar/alpha
    ell_eff = (1/mp.sqrt(pi))*eta_eff
    P_ir    = P_IR(ell_eff, IR_kernel, mode=S4_TTA_MODE)
    dL_UVIR = c0_gauss * P_ir
    dL_OUT  = DeltaLambda_OUT(eta_eff, OUT_mode, OUT_lmax)
    Lambda_eff = (Lambda0 + dL_UVIR + dL_OUT)  # closed (no manual shifts)
    zeta   = Lambda_eff * (K/(2*pi**2))
    dA1    = - (alpha/pi) * zeta
    dA2    = (dA1**2) / (4*delta_scalar)
    d4     = delta_scalar + dA1 + dA2
    return (eta_eff, ell_eff, P_ir, dL_UVIR, dL_OUT, Lambda_eff, zeta, dA1, dA2, d4)

# ---------- TT–χ acceptance and closed ΔΛ_OUT for Mass-Hierarchy block ----------
def P_IR_tt_chi(ell):
    """
    TT–χ (dot-channel) IR acceptance used in the Mass-Hierarchy block:
      P_IR^(χ)(ℓ) = ⟨ [1 − (1/3) f_swirl] e^{-((1−x)/ℓ)^2} ⟩_w / ⟨1⟩_w
    """
    It  = _Itot()
    w   = lambda x: (x**2)*(mp.sin(pi*x)**2)
    num = mp.quad(lambda x: w(x)*(1 - mp.mpf('1')/3 * _f_swirl(x, ell)) * mp.e**(-((1-x)/ell)**2), [0,1])
    return num/It

def DeltaLambda_OUT_closed(eta):
    """
    Closed outer subtraction used in the Mass-Hierarchy paper:
      ΔΛ_OUT(η) = −π[ ln(1−η^4)/(2η) + atanh(η) − atan(η) ],   (0<η<1).
    """
    return -pi*( mp.log(1-eta**4)/(2*eta) + mp.atanh(eta) - mp.atan(eta) )

# ---------- NEW: TT populations from closed overlaps (no fits) ----------
def _toroidal_norm_ratio(n):
    # N_ell / N_1 with ell = 2n-1 and N_ell = ∫ (1−x^2) [P'_\ell]^2 dx = 2ℓ(ℓ+1)/(2ℓ+1)
    return ( (2*(2*n-1)*(2*n)) / (4*n - 1) ) / ( (2*1*2) / 3 )  # = (3/2)*((4n^2 − 2n)/(4n − 1))

def closed_tt_population(n):
    """
    Closed geometric TT population w_n = W_n:
      W_n = (N_{2n−1}/N_1) * [ P_IR^(χ)(ℓ_n)/P_IR^(χ)(ℓ_1) ] * [ |ΔΛ_OUT(η_1)| / |ΔΛ_OUT(η_n)| ]
      with η_n = 1/(nπ),  ℓ_n = η_n/√π.
    """
    if n == 1:
        return mp.mpf('1')
    eta1 = 1/pi;  ell1 = eta1/mp.sqrt(pi)
    PIR1 = P_IR_tt_chi(ell1);  D1 = abs(DeltaLambda_OUT_closed(eta1))
    eta_n = 1/(n*pi);  ell_n = eta_n/mp.sqrt(pi)
    PIRn = P_IR_tt_chi(ell_n); Dn = abs(DeltaLambda_OUT_closed(eta_n))
    ang  = _toroidal_norm_ratio(n)
    return ang * (PIRn/PIR1) * (D1/Dn)

# ---------- Mass-Hierarchy effective c_eff(n) (no fits; TT populations switchable) ----------
def mass_hierarchy_ceff(n, K):
    """
    Build c_eff(n) per the Mass-Hierarchy mapping:
      c_eff(1) = 1/3;
      for n=2,3 use TT population from either:
        (i) closed overlaps (USE_CLOSED_TT_POPULATIONS=True), or
        (ii) paper rationals (USE_CLOSED_TT_POPULATIONS=False).
    """
    one_third = mp.mpf('1')/3
    if n == 1:
        eta1 = 1/pi; ell1 = eta1/mp.sqrt(pi)
        return one_third, {
            'n':1,'k_n':1,'eta_n':eta1,'ell_n':ell1,
            'PIR_chi_n':P_IR_tt_chi(ell1),
            'DeltaLam_out_n':DeltaLambda_OUT_closed(eta1),
            'Gamma_geom': mp.mpf('0.5')*mp.sinh(eta1)/eta1,
            'k_curv': mp.sinh(eta1)/eta1-1,
            'X_n': (series_K(tol=S3_SERIES_TOL)/(2*D_C))*c0_gauss*P_IR_tt_chi(ell1),
            'Gamma_map': mp.mpf('0'),
            'gamma': mp.mpf('0'),
            'R_n': mp.mpf('0'),
            'w_n': mp.mpf('1'),
        }

    # Resonant law k_n = n(2n−1)
    k_n  = n*(2*n-1)
    eta1 = 1/pi;      ell1 = eta1/mp.sqrt(pi)
    PIR1 = P_IR_tt_chi(ell1); D1   = abs(DeltaLambda_OUT_closed(eta1))

    eta_n = 1/(n*pi); ell_n = eta_n/mp.sqrt(pi)
    PIRn  = P_IR_tt_chi(ell_n); Dn  = abs(DeltaLambda_OUT_closed(eta_n))

    K_here = series_K(tol=S3_SERIES_TOL)  # consistent with S3
    X_n    = (K_here/(2*D_C)) * c0_gauss * PIRn
    Gamma_geom = mp.mpf('0.5')*mp.sinh(eta_n)/eta_n
    k_curv     = mp.sinh(eta_n)/eta_n - 1

    # ---------------- SWITCHED TT POPULATIONS ----------------
    if USE_CLOSED_TT_POPULATIONS:
        # Closed overlap value (no fit, no rationals hard-coded)
        w_n = closed_tt_population(n)
    else:
        # Rational TT populations (paper): wμ=11/2, wτ=41/16
        if n == 2:
            w_n = mp.mpf('11')/2
        elif n == 3:
            w_n = mp.mpf('41')/16
        else:
            w_n = mp.mpf('1')
    # ---------------------------------------------------------

    Gamma_map = (w_n * k_n * X_n) / (1 + k_curv * w_n * k_n * X_n)
    gamma     = Gamma_map / (Gamma_geom + Gamma_map)
    R_n       = (PIRn / PIR1) * (D1 / Dn)
    c_eff     = one_third * (1 + gamma * R_n)

    details = {
        'n': n, 'k_n': k_n,
        'eta_n': eta_n, 'ell_n': ell_n,
        'PIR_chi_n': PIRn, 'PIR_chi_1': PIR1,
        'DeltaLam_out_n': DeltaLambda_OUT_closed(eta_n),
        'DeltaLam_out_1': DeltaLambda_OUT_closed(eta1),
        'X_n': X_n,
        'Gamma_geom': Gamma_geom, 'k_curv': k_curv,
        'w_n': w_n, 'Gamma_map': Gamma_map, 'gamma': gamma,
        'R_n': R_n,
    }
    return c_eff, details

# ==================== Stage χ: (A) pure log-shell, (B) Mass-Hierarchy enhanced ====================
def delta_chi_mass_shift_pure_log(mass, m_ref, delta_scalar):
    """
    PURE Stage-χ mass shift (no constants):
      Δδχ_pure(m) = [δ_scalar]^2 * (1/3) * ln(m/m_ref),  with δ_scalar ≡ δ_S3.
    """
    return (delta_scalar**2) * (mp.mpf('1')/3) * mp.log(mass/m_ref)

def delta_chi_mass_shift_hierarchy(mass, m_ref, n_band, delta_scalar, K):
    """
    Mass-Hierarchy–enhanced Stage-χ:
      Δδχ_hier(n) = [δ_scalar]^2 * c_eff(n) * ln(m_n/m_e),  c_eff from mass_hierarchy_ceff(n,K).
    """
    c_eff, _ = mass_hierarchy_ceff(n_band, K)
    return (delta_scalar**2) * c_eff * mp.log(mass/m_ref)

# ==================== MAIN ====================
def run():
    sep("Relator g — CLOSED, NO-FIT | S1→S3 + S4 + χ (e, μ, τ)")
    print(f"alpha^-1                 = {mp.nstr(alpha_inv, 16)}")
    print(f"IR kernel (S4)           = {S4_IR_KERNEL}/{S4_TTA_MODE},  OUT={S4_OUT_MODE},  Lmax={S4_OUT_LMAX}")
    print(f"TT populations           = {'closed overlaps' if USE_CLOSED_TT_POPULATIONS else 'paper rationals (wμ=11/2, wτ=41/16)'}")

    # ---- Stage S1, S2 ----
    sep("Stage S1, S2 (scalar)")
    d1 = delta_S1(); d2 = delta_S2()
    p("δ_S1", d1, note="δ_S1 = α/π")
    p("g(δ_S1)", g_of_delta(d1)); p("Δg_e (ppt)", ppt_error(g_of_delta(d1), g_exp_e))
    p("ξ(α)", xi_of_alpha(alpha), note="ξ = 2 C0_uni α")
    p("δ_S2", d2, note="δ_S2 = (α/π)√(1−ξ)")
    p("g(δ_S2)", g_of_delta(d2)); p("Δg_e (ppt)", ppt_error(g_of_delta(d2), g_exp_e))

    # ---- Stage S3 ----
    sep("Stage S3 (Coulombic stack)")
    d3, K, rows = delta_S3(alpha, S3_MAX_M)
    p("K (m=1)", K)
    p("δ_S3", d3, note="δ_S3 = (α/π)√(1−ξ) − (α/π)(ξ/2)K − Σ_{m≥2}(α/π)(ξ/2)^m L_{2m}")
    p("g(δ_S3)", g_of_delta(d3)); p("Δg_e (ppt)", ppt_error(g_of_delta(d3), g_exp_e))
    if rows:
        print("\n   m       L_{2m}                  Δδ_C^{(2m)}")
        for m,Lm,dlt in rows:
            print(f"  {m:3d}  {mp.nstr(Lm, 20):>20}  {mp.nstr(dlt, 20):>20}")

    # ---- Stage S4 (vector backreaction on scalar stack) ----
    sep("Stage S4 (vector self-magnetic)")
    (eta_eff, ell_eff, P_ir_S4, dL_UVIR_S4, dL_OUT_S4,
     Lambda_eff_S4, zeta_S4, dA1, dA2, d4) = stage_S4prime(d3, K, S4_IR_KERNEL, S4_OUT_MODE, S4_OUT_LMAX)

    p("η_eff (R/r*)", eta_eff)
    p("ℓ_eff (=εη)",  ell_eff, note="ε = 1/√π")
    p("P^(IR)[S4]",   P_ir_S4)
    p("ΔΛ^(UV→IR)[S4]", dL_UVIR_S4)
    p("ΔΛ_OUT[S4]",  dL_OUT_S4)
    p("Λ_eff[S4]",   Lambda_eff_S4, note="Λ_eff = Λ_ind + ΔΛ^(UV→IR) + ΔΛ_OUT")
    p("ζ_geom[S4]",  zeta_S4, note="ζ = (K/(2π^2)) Λ_eff")
    p("δ_A^(1)",     dA1, note="= −(α/π) ζ")
    p("δ_A^(2)",     dA2, note="= (δ_A^(1))^2 / (4 δ_scalar)")
    p("δ_S4",        d4,  note="δ_S4 = δ_S3 + δ_A^(1) + δ_A^(2)")

    # ---- Stage χ — (A) pure log-shell, CLOSED ----
    sep("Stage χ — Pure log-shell (closed)")
    ln_mu_e  = mp.log(m_mu/m_e); ln_tau_e = mp.log(m_tau/m_e)
    p("δ_scalar used", d3, note="δ_scalar ≡ δ_S3")
    p("coeff (1/3)", mp.mpf('1')/3, note="log-shell lemma")
    p("ln(m_μ/m_e)", ln_mu_e);  p("ln(m_τ/m_e)", ln_tau_e)

    ddchi_e_pure   = mp.mpf('0')
    ddchi_mu_pure  = delta_chi_mass_shift_pure_log(m_mu,  m_e, d3)
    ddchi_tau_pure = delta_chi_mass_shift_pure_log(m_tau, m_e, d3)
    p("Δδχ_pure(e)",  ddchi_e_pure)
    p("Δδχ_pure(μ)",  ddchi_mu_pure)
    p("Δδχ_pure(τ)",  ddchi_tau_pure)

    # ---- Stage χ — (B) Mass-Hierarchy enhanced, CLOSED ----
    sep("Stage χ — Mass-Hierarchy enhanced (closed)")
    c1, det1 = mass_hierarchy_ceff(1, K)
    c2, det2 = mass_hierarchy_ceff(2, K)
    c3, det3 = mass_hierarchy_ceff(3, K)

    print("\n[Mass-Hierarchy blocks] — electron band (n=1)")
    p("η_1", det1['eta_n']); p("ℓ_1", det1['ell_n'])
    p("P_IR^(χ)(ℓ_1)", det1['PIR_chi_n'])
    p("ΔΛ_OUT^closed(η_1)", det1['DeltaLam_out_n'])
    p("Γ_geom(1)", det1['Gamma_geom']); p("k_curv(1)", det1['k_curv'])
    p("X_1", det1['X_n']); p("γ_{1,k1}", det1['gamma'])
    p("R_1", det1['R_n']); p("c_eff(1) = 1/3", c1)

    print("\n[Mass-Hierarchy blocks] — muon band (n=2)")
    for key in ['eta_n','ell_n','PIR_chi_n','PIR_chi_1','DeltaLam_out_n','DeltaLam_out_1',
                'X_n','Gamma_geom','k_curv','w_n','Gamma_map','gamma','R_n','k_n']:
        p(key, det2[key])
    p("c_eff(2)", c2)

    print("\n[Mass-Hierarchy blocks] — tau band (n=3)")
    for key in ['eta_n','ell_n','PIR_chi_n','PIR_chi_1','DeltaLam_out_n','DeltaLam_out_1',
                'X_n','Gamma_geom','k_curv','w_n','Gamma_map','gamma','R_n','k_n']:
        p(key, det3[key])
    p("c_eff(3)", c3)

    ddchi_e_hier   = mp.mpf('0')
    ddchi_mu_hier  = (d3**2) * c2 * ln_mu_e
    ddchi_tau_hier = (d3**2) * c3 * ln_tau_e
    p("\nΔδχ_hier(e)",  ddchi_e_hier)
    p("Δδχ_hier(μ)",  ddchi_mu_hier)
    p("Δδχ_hier(τ)",  ddchi_tau_hier)

    # ---- Totals and final g: (A) pure log vs (B) hierarchy-enhanced ----
    sep("Final g — Pure log-shell vs Mass-Hierarchy (CLOSED, NO-FIT)")

    # Pure log-shell totals
    d_tot_e_p = d4 + ddchi_e_pure
    d_tot_mu_p= d4 + ddchi_mu_pure
    d_tot_tau_p=d4 + ddchi_tau_pure

    g_e_p = g_of_delta(d_tot_e_p)
    g_mu_p= g_of_delta(d_tot_mu_p)
    g_tau_p=g_of_delta(d_tot_tau_p)

    print("\n• Pure log-shell (1/3)")
    p("δ_tot(e)", d_tot_e_p);  p("g_e (pred)", g_e_p);   p("Δg_e (ppt)", ppt_error(g_e_p, g_exp_e))
    p("δ_tot(μ)", d_tot_mu_p); p("g_μ (pred)", g_mu_p);  p("Δg_μ (ppt)", ppt_error(g_mu_p, g_exp_mu))
    p("δ_tot(τ)", d_tot_tau_p);p("g_τ (pred)", g_tau_p); print(f"{'Δg_τ (ppt)':<26} = {safe_ppt_error(g_tau_p, g_exp_tau)}")

    # Mass-Hierarchy totals
    d_tot_e_h = d4 + ddchi_e_hier
    d_tot_mu_h= d4 + ddchi_mu_hier
    d_tot_tau_h=d4 + ddchi_tau_hier

    g_e_h = g_of_delta(d_tot_e_h)
    g_mu_h= g_of_delta(d_tot_mu_h)
    g_tau_h= g_of_delta(d_tot_tau_h)

    print("\n• Mass-Hierarchy enhanced")
    p("δ_tot(e)", d_tot_e_h);  p("g_e (pred)", g_e_h);   p("Δg_e (ppt)", ppt_error(g_e_h, g_exp_e))
    p("δ_tot(μ)", d_tot_mu_h); p("g_μ (pred)", g_mu_h);  p("Δg_μ (ppt)", ppt_error(g_mu_h, g_exp_mu))
    p("δ_tot(τ)", d_tot_tau_h);p("g_τ (pred)", g_tau_h); print(f"{'Δg_τ (ppt)':<26} = {safe_ppt_error(g_tau_h, g_exp_tau)}")

# ---------------- Driver ----------------
if __name__ == "__main__":
    run()
