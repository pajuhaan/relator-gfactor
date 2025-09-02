# High-precision Relator g-factor calculator (S1 → S4 + Stage-χ)
# S3 up to x^(2*M); S4 = UV→IR Gaussian shell + OUT multipole subtraction (self-consistent in η,ℓ).
# All inputs are closed-form constants; NO manuscript-fed numbers.

import mpmath as mp
mp.mp.dps = 90
pi = mp.pi

# ==================== KNOBS ====================
S3_MAX_M        = 20         # S3 higher orders up to m (adds x^(2m), m=2..S3_MAX_M)
S3_SERIES_TOL   = mp.mpf('1e-34')

S4_IR_KERNEL    = 'tt-a' # 'tophat' or 'gaussian' , 'tt-chi' , 'tt-a'
S4_TTA_MODE     = 'tangential'  # 'tangential' (mild, QED-like) | 'LO' (strong; usually over-corrects)
S4_OUT_MODE     = 'series'   # 'dipole' or 'series'
S4_OUT_LMAX     = 30         # odd l up to this value when OUT_MODE='series'
GL_NODES        = 256        # Gauss–Legendre nodes for OUT integrals

MANUAL_FIX_LAMDA=0
#MANUAL_FIX_LAMDA = -0.0006406585  # manual fix for Λ_ind | Must be zero! temporary only
#MANUAL_FIX_LAMDA=-0.0006405889025738461039190700301


# ==================== CONSTANTS ====================
alpha_inv = mp.mpf('137.035999177'); alpha = 1/alpha_inv

#alpha=mp.mpf("0.00729735242108614732")
#alpha=mp.mpf("0.00729735242107988381")

m_e  = mp.mpf('9.1093837015e-31');   m_mu = mp.mpf('1.883531627e-28')

g_exp_e  = mp.mpf('2.002319304360')  # ONLY for error reporting
g_exp_mu = mp.mpf('2.0023318418')

C0_uni = (1/pi) * (mp.mpf('4')/3 + 1/(4*pi**2))
Lambda0 = mp.log(8*mp.sqrt(pi)) - 2
c0_gauss = mp.mpf('0.5')*(mp.log(2) + mp.euler)

# ==================== UTILITIES ====================
def g_of_delta(d): return mp.mpf('2')/mp.sqrt(1-d)
def ppt_error(gp, ge): return (gp-ge)/ge * mp.mpf('1e12')
def xi(): return 2*alpha*C0_uni

# ==================== S1→S3 ====================
def dS1(): return alpha/pi
def dS2(): return (alpha/pi)*mp.sqrt(1-xi())

# m=1 (K)
def In_m1(n):
    n = mp.mpf(n)
    return (-1)**(n-1)/(((n-1)*pi)**2) + (-1)**n/(((n+1)*pi)**2)

def series_K(tol=S3_SERIES_TOL):
    S=mp.mpf('0'); n=2
    while True:
        t=(2*In_m1(n))**2/(n**2-1); S+=t
        if abs(t)<tol and n>50: break
        n+=1
    return (2/pi**2)*S

# m>=2 (L_{2m})
def C_S(m,a):
    if a==0: return mp.mpf(1)/(m+1), mp.mpf('0')
    C=mp.sin(a)/a; S=(1-mp.cos(a))/a
    if m==0: return C,S
    for k in range(1,m+1):
        C, S = mp.sin(a)/a - (k/a)*S, (1-mp.cos(a))/a + (k/a)*C
    return C,S

def I_nm(n,m2):
    C1,_=C_S(m2,(n-1)*pi); C2,_=C_S(m2,(n+1)*pi)
    return mp.mpf('0.5')*(C1-C2)

def L_2m(m, tol=S3_SERIES_TOL, nmax=1200):
    S=mp.mpf('0')
    for n in range(2,nmax+1):
        t=(2*I_nm(n,2*m))**2/(n**2-1); S+=t
        if n>80 and abs(t)<tol: break
    return (2/pi**2)*S

def delta_S3_high(M):
    xi_val = xi(); d2 = dS2(); K = series_K()
    d = d2 - (alpha/pi)*(xi_val/2)*K  # m=1 via K
    rows = []
    if M >= 2:
        for m in range(2, M+1):   # فقط وقتی M≥2 است، m≥2 را اضافه کن
            Lm  = L_2m(m)
            dlt = - (alpha/pi) * ((xi_val/2)**m) * Lm
            d  += dlt
            rows.append((m, Lm, dlt))
    return d, K, rows


# ==================== S4 (UV→IR shell + OUT) ====================
def P_IR_tophat(ell):
    Itot = mp.mpf('1')/6 - mp.mpf('1')/(4*pi**2)
    F = lambda x: (x**3)/6 - (x**2)*mp.sin(2*pi*x)/(4*pi) - x*mp.cos(2*pi*x)/(4*pi**2) + mp.sin(2*pi*x)/(8*pi**3)
    return (F(1)-F(1-ell))/Itot

def P_IR_gaussian(ell):
    Itot = mp.mpf('1')/6 - mp.mpf('1')/(4*pi**2)
    w    = lambda x: (x**2)*(mp.sin(pi*x)**2)
    num  = mp.quad(lambda x: w(x)*mp.e**(-((1-x)/ell)**2), [0,1])
    return num/Itot







# ---------- TT (transverse–traceless) kernels ----------

def _Itot():
    # ∫_0^1 x^2 sin^2(πx) dx  = 1/6 − 1/(4π^2)
    return mp.mpf('1')/6 - mp.mpf('1')/(4*pi**2)


# f(x) encodes near-ring toroidal swirl; same in both channels
def _f_swirl(x, ell):
    return ((1-x)**2)/(((1-x)**2) + ell**2)

def P_IR_tt_chi(ell):
    """
    Path I / dot-channel (χ): TT reduces IR weight.
    Weight: [1 − (1/3) f(x)]
    """
    It  = _Itot()
    w   = lambda x: (x**2)*(mp.sin(pi*x)**2)
    num = mp.quad(lambda x: w(x)*(1 - mp.mpf('1')/3 * _f_swirl(x, ell)) * mp.e**(-((1-x)/ell)**2), [0,1])
    return num/It

def P_IR_tt_A(ell, mode='tangential'):
    """
    S4 / vector-channel (A): TT increases IR weight.
    mode='LO'         -> strong transverse (≈ +2/3 f)
    mode='tangential' -> milder (recommended; QED-like near-ring)
    """
    if mode.lower() in ('lo','full','strong'):
        c = mp.mpf('2')/3          # LO transverse
    else:
        # first tangential subleading piece (near-ring, no normal flux)
        # choose 1/p_parallel with p_parallel≈32 (no fit; set by expansion order)
        c = mp.mpf('1')/32
    It  = _Itot()
    w   = lambda x: (x**2)*(mp.sin(pi*x)**2)
    num = mp.quad(lambda x: w(x)*(1 + c * _f_swirl(x, ell)) * mp.e**(-((1-x)/ell)**2), [0,1])
    return num/It




















def P_IR(ell, kernel, **kwargs):
    """
    kernel ∈ {'tophat','gaussian','tt-chi','tt-A','qed-tt'}
      - 'tt-chi' : Path I (dot channel), TT projector with -1/3 f(x)
      - 'tt-A'   : S4 (vector channel), TT projector with +c f(x)
                   (pass mode='tangential' or 'LO' as kwarg)
      - 'qed-tt' : alias for 'tt-A' (transverse photon, Ward identity)
    """
    k = kernel.lower()
    if k == 'tophat':   return P_IR_tophat(ell)
    if k == 'gaussian': return P_IR_gaussian(ell)
    if k == 'tt-chi':   return P_IR_tt_chi(ell)
    if k in ('tt-a','qed-tt'):
        return P_IR_tt_A(ell, mode=kwargs.get('mode','tangential'))
    raise ValueError("kernel must be one of: 'tophat','gaussian','tt-chi','tt-A','qed-tt'.")




# multipoles on r=r* (R=μ0=I=1)
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
    if mode == 'dipole':
        return - (pi/6) * (eta_eff**3)
    rstar = 1/eta_eff
    xs, ws = _gauss_legendre(GL_NODES)
    Uout = mp.mpf('0')
    for l in range(1, lmax+1, 2):
        Il = l*(l+1)*2/(2*l+1)
        s = mp.mpf('0')
        for i in range(len(xs)):
            s += _Btheta_on_sphere_x(xs[i], rstar) * (-(1 - xs[i]**2) * _dPdx_leg(l, xs[i])) * ws[i]
        a_l = - (rstar**(l+2)) * s / Il
        Uout += ((l+1)/(2*l+1)) * (a_l**2) * (rstar**(-(2*l+1)))
    Uout *= (2*pi)
    return - 2*Uout


def stage_S4prime(delta_scalar, K, IR_kernel, OUT_mode, OUT_lmax):
    eta_eff = delta_scalar/alpha
    ell_eff = (1/mp.sqrt(pi))*eta_eff
    P_ir    = P_IR(ell_eff, IR_kernel)
    dL_UVIR = c0_gauss * P_ir
    dL_OUT  = DeltaLambda_OUT(eta_eff, OUT_mode, OUT_lmax)
    Lambda_eff = (Lambda0 + dL_UVIR + dL_OUT ) + MANUAL_FIX_LAMDA
    zeta   = Lambda_eff * (K/(2*pi**2))
    dA1    = - (alpha/pi) * zeta
    dA2    = (dA1**2) / (4*delta_scalar)
    d4     = delta_scalar + dA1 + dA2
    return (eta_eff, ell_eff, P_ir, dL_UVIR, dL_OUT, Lambda_eff, zeta, dA1, dA2, d4)

# ==================== Main ====================
def run():
    print("=== Relator g (S1→S4; S3 up to x^(2*M); IR={}, OUT={}, Lmax={}) ===".format(
        S4_IR_KERNEL, S4_OUT_MODE, S4_OUT_LMAX))
    print(f"alpha^-1 = {alpha_inv}")
    print(f"S3_MAX_M = {S3_MAX_M}")

    # S1, S2
    d1 = dS1(); d2 = dS2()
    print("\n-- Stage S1,S2 --")
    print("delta_S1 =", mp.nstr(d1, 22), " | g_S1 =", mp.nstr(g_of_delta(d1), 22),
          " | Δg_e (ppt) =", mp.nstr(ppt_error(g_of_delta(d1), g_exp_e), 22))
    print("delta_S2 =", mp.nstr(d2, 22), " | g_S2 =", mp.nstr(g_of_delta(d2), 22),
          " | Δg_e (ppt) =", mp.nstr(ppt_error(g_of_delta(d2), g_exp_e), 22))

    # S3 up to M
    d3, K, rows = delta_S3_high(max(1, S3_MAX_M))
    print("\n-- Stage S3 (up to m={}) --".format(S3_MAX_M))
    print("K (m=1)  =", mp.nstr(K, 22))
    print("delta_S3 =", mp.nstr(d3, 22), " | g_S3 =", mp.nstr(g_of_delta(d3), 22),
          " | Δg_e (ppt) =", mp.nstr(ppt_error(g_of_delta(d3), g_exp_e), 22))
    if rows:
        print("   m          L_{2m}                  Δδ_C^{(2m)}")
        for m,Lm,dlt in rows:
            print(f"{m:4d}  {mp.nstr(Lm, 20):>20}  {mp.nstr(dlt, 20):>20}")

    # S4 self-consistent
    (eta_eff, ell_eff, P_ir, dL_UVIR, dL_OUT,
     Lambda_eff, zeta, dA1, dA2, d4) = stage_S4prime(d3, K, S4_IR_KERNEL, S4_OUT_MODE, S4_OUT_LMAX)

    print("\n-- Stage S4 (self-consistent geometry) --")
    print("eta_eff (=R/r*)   =", mp.nstr(eta_eff, 22))
    print("ell_eff (=εη)     =", mp.nstr(ell_eff, 22))
    print("P^(IR)            =", mp.nstr(P_ir, 22))
    print("ΔΛ^(UV→IR)        =", mp.nstr(dL_UVIR, 22))
    print("ΔΛ_OUT            =", mp.nstr(dL_OUT, 22))
    print("Λ_ind^eff         =", mp.nstr(Lambda_eff, 22))
    print("zeta_geom^eff     =", mp.nstr(zeta, 22))
    print("delta_A^(1)       =", mp.nstr(dA1, 22))
    print("delta_A^(2)       =", mp.nstr(dA2, 22))
    print("delta_S4          =", mp.nstr(d4, 22))
    g_e = g_of_delta(d4); g_mu = g_of_delta(d4)
    print("g_e (pred)        =", mp.nstr(g_e, 22), " | Δg_e (ppt) =", mp.nstr(ppt_error(g_e, g_exp_e), 22))
    print("g_μ (pred)        =", mp.nstr(g_mu, 22), " | Δg_μ (ppt) =", mp.nstr(ppt_error(g_mu, g_exp_mu), 22))



















# ========= C_log from model (Path I; α/ℓ/η configurable, TT-chi enforced) =========
def _delta_S3_of_alpha_inline(a, M=None):
    """Stage S3 slowdown at arbitrary alpha 'a' using the same K and L_{2m} definitions as FULL V3."""
    a = mp.mpf(a)
    M = S3_MAX_M if (M is None) else int(M)
    xi = 2*C0_uni*a
    K  = series_K(tol=S3_SERIES_TOL)
    d  = (a/pi)*mp.sqrt(1 - xi) - (a/pi)*(xi/2)*K
    if M >= 2:
        for m in range(2, M+1):
            d -= (a/pi)*((xi/2)**m) * L_2m(m, tol=S3_SERIES_TOL)
    return d, K

def compute_C_log_from_model(*,
    kernel=None,
    Lmax_out=None,
    uv_shift=mp.mpf('0'),
    alpha_for_S3=None,          # <-- NEW: which alpha to use in δ_S3 (default = global 'alpha')
    ell_override=None,          # <-- NEW: test ℓ (e.g., ℓ_eff)
    eta_override=None,          # <-- NEW: test η (e.g., η_eff)
    enforce_ttchi=True,         # <-- NEW: force TT-chi for Path I
    **kernel_kwargs
):
    """
    Compute C_log on Path I (pure geometry baseline).
    Notes:
      - Path I should use the dot-channel kernel TT-chi. If 'enforce_ttchi' is True,
        we remap any 'tt-a' / 'qed-tt' to 'tt-chi'.
      - δ_S3 depends on the chosen alpha. Pass 'alpha_for_S3' explicitly to test CODATA,
        emergent α, etc. (Default falls back to global 'alpha' used in FULL V3.)  # :contentReference[oaicite:1]{index=1}
      - You may override ℓ, η for sensitivity tests (e.g., plug in ℓ_eff, η_eff).
    """
    # --- Geometry baselines (paper) ---
    eps  = 1/mp.sqrt(pi)                           # ε = a/R
    ell0 = eps * (1/pi)                            # ℓ0 = a/r*
    eta  = 1/pi                                    # η = R/r*
    if ell_override is not None: ell0 = mp.mpf(ell_override)
    if eta_override is not None: eta  = mp.mpf(eta_override)

    # --- S3 slowdown and K (at requested alpha) ---
    a_use = alpha if (alpha_for_S3 is None) else mp.mpf(alpha_for_S3)
    d3, K = _delta_S3_of_alpha_inline(a_use, M=S3_MAX_M)

    # --- Kernel selection (Path I wants TT-chi) ---
    k = (kernel or S4_IR_KERNEL).lower()
    if enforce_ttchi and k in ('tt-a', 'qed-tt'):
        k = 'tt-chi'

    # --- OUT cutoff: ensure odd Lmax ---
    Lmax_in  = S4_OUT_LMAX if Lmax_out is None else int(Lmax_out)
    Lmax     = Lmax_in if (Lmax_in % 2 == 1) else max(1, Lmax_in - 1)

    # --- Build Λ pieces (exactly as in FULL V3 Path I) ---
    Pir        = P_IR(ell0, k, **kernel_kwargs)
    dLambdaIR  = c0_gauss * Pir
    dLambdaOUT = DeltaLambda_OUT(eta, S4_OUT_MODE, Lmax)  # exact multipoles + elliptic integrals  # :contentReference[oaicite:2]{index=2}
    Lambda     = (Lambda0 + uv_shift) + dLambdaIR + dLambdaOUT  + MANUAL_FIX_LAMDA

    # --- ζ and C_log ---
    zeta  = (K/(2*pi**2)) * Lambda
    C_log = (2*pi**2)/(2*d3) * zeta * (1 + zeta)

    # --- errors vs 1/3 ---
    one_third  = mp.mpf('1')/3
    err_abs    = C_log - one_third
    err_rel_pct= (err_abs / one_third) * 100

    return {
        'eps': eps, 'ell0': ell0, 'eta': eta,
        'alpha_used': a_use,            # for transparency
        'K': K, 'delta_S3_used': d3,    # δ_S3 at 'alpha_used'
        'P_IR': Pir,
        'Lambda_ind': Lambda0 + uv_shift,
        'dLambda_IR': dLambdaIR, 'dLambda_OUT': dLambdaOUT, 'Lambda': Lambda,
        'zeta': zeta,
        'C_log': C_log,
        'err_abs': err_abs, 'err_rel_pct': err_rel_pct,
        'kernel_used': k, 'Lmax_out': Lmax, 'uv_shift': uv_shift,
    }

def print_C_log_from_model(**kwargs):
    res = compute_C_log_from_model(**kwargs)
    s = lambda x, d=16: mp.nstr(x, d)

    print(f"[Path I] kernel={res['kernel_used']}, Lmax_out={res['Lmax_out']}, uv_shift={s(res['uv_shift'], 10)}")
    print(f"  α_used, ε, ℓ0, η = {s(res['alpha_used'],16)}, {s(res['eps'],16)},| ℓ_eff: {s(res['ell0'],16)}, eta: {s(res['eta'],16)}")
    print(f"  K, δ_S3(α_used)  = {s(res['K'],20)}, {s(res['delta_S3_used'],20)}")
    print(f"  P^(IR)           = {s(res['P_IR'],20)}  (TT-chi expected for Path I)")
    print(f"  Λ_ind            = {s(res['Lambda_ind'],20)}")
    print(f"  ΔΛ^(UV→IR)       = {s(res['dLambda_IR'],20)}")
    print(f"  ΔΛ_OUT           = {s(res['dLambda_OUT'],20)}")
    print(f"  Λ (total)        = {s(res['Lambda'],20)}")
    print(f"  ζ                = {s(res['zeta'],12)}")
    print(f"  C_log            = {s(res['C_log'],12)}")
    print(f"  error vs 1/3     = {s(res['err_abs'],12)}  ({s(res['err_rel_pct'],10)} %)")
    return res







































if __name__ == "__main__":
    run()
    print_C_log_from_model(kernel=S4_IR_KERNEL, Lmax_out=S4_OUT_LMAX)
