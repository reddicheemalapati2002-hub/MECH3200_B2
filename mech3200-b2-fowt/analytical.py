"""
MECH3200 B2 -- analytical tower design (SN*=91).
B2.1 diameter from f1 >= 1.2 x 3P (tip speed 90.7 m/s), fixed base, no nacelle.
B2.2 thickness from tip amplitude < 0.7% Lt under P = k U^2 at 1P and 3P.
B2.3 nacelle tip-mass effect: Rayleigh estimate vs exact transcendental solution.
"""
import numpy as np
from scipy.optimize import brentq
import json

SN = 91
L  = 100*(1+SN/200)          # blade length, m
Lt = 1.25*L                  # tower length, m
Mr = 3*L*1e3                 # rotor+hub mass, kg
Mn = 1.2*Mr                  # nacelle mass, kg
E, nu, rho = 200e9, 0.25, 7850.0   # AS4100 structural steel
G = E/(2*(1+nu))
Vtip = 90.7                  # limited blade tip speed, m/s
nb = 3

# --- excitation ---
Omega_b = Vtip/L
f1P = Omega_b/(2*np.pi); f3P = nb*f1P
f_req = 1.2*f3P

# --- cantilever eigenvalues (fixed-free, no tip mass) ---
def charfn(bl): return 1+np.cos(bl)*np.cosh(bl)
bL = [brentq(charfn, a, b) for a,b in [(1.5,2.2),(4.4,5.0),(7.5,8.2)]]

# thin tube: I = pi d^3 t /8, A = pi d t  -> I/A = d^2/8, so f_n depends on d only
def f_n_tube(d, n):   # n index 0..2
    return bL[n]**2/(2*np.pi*Lt**2)*np.sqrt(E/rho)*d/np.sqrt(8)

# B2.1: solve d such that f1 = f_req
d_t = f_req/(bL[0]**2/(2*np.pi*Lt**2)*np.sqrt(E/rho)/np.sqrt(8))
f_bare = [f_n_tube(d_t,n) for n in range(3)]

# cantilever mode shape, tip-normalised
def phi(x, n, Lx=Lt):
    b = bL[n]/Lx
    sig = (np.sinh(bL[n])-np.sin(bL[n]))/(np.cosh(bL[n])+np.cos(bL[n]))
    y = np.cosh(b*x)-np.cos(b*x)-sig*(np.sinh(b*x)-np.sin(b*x))
    ytip = np.cosh(bL[n])-np.cos(bL[n])-sig*(np.sinh(bL[n])-np.sin(bL[n]))
    return y/ytip
xs = np.linspace(0,Lt,4001)
modal_int = [np.trapezoid(phi(xs,n)**2, xs)/Lt for n in range(3)]  # ~0.25 for all

# B2.2: wind force
P10 = 46.8e3*(1+SN/100)
kP  = P10/10**2
Umax = 40.0
P    = kP*Umax**2
a_load = 0.95*Lt
zeta = 0.0  # undamped (conservative) for the design; 1% checked in FE
def tip_amp(t, d=d_t, modes=3, zeta=0.0, freq_list=None):
    """steady-state tip amplitude, modal summation, point load at a_load"""
    mu = rho*np.pi*d*t
    tot = 0.0
    out = {}
    for lab, fex in (('1P',f1P),('3P',f3P)):
        Om = 2*np.pi*fex
        X = 0.0
        for n in range(modes):
            wn = 2*np.pi*f_n_tube(d,n)
            mn = mu*Lt*modal_int[n]
            r = Om/wn
            X += P*phi(a_load,n)/(mn*wn**2)/np.sqrt((1-r**2)**2+(2*zeta*r)**2)
        out[lab]=X
        tot += X
    out['sum']=tot
    return out
X_allow = 0.007*Lt
t_req = brentq(lambda t: tip_amp(t)['sum']-X_allow, 0.005, 0.5)
amp_at_t = tip_amp(t_req)
# static deflection check (1st-order): delta = P a^2 (3L-a)/(6EI)
I_t = np.pi*d_t**3*t_req/8
delta_st = P*a_load**2*(3*Lt-a_load)/(6*E*I_t)
DAF3 = 1/(1-(f3P/f_bare[0])**2); DAF1 = 1/(1-(f1P/f_bare[0])**2)
tower_mass = rho*np.pi*d_t*t_req*Lt

# B2.3: nacelle tip mass
alpha = Mn/tower_mass
# Rayleigh (static-shape) estimate: f' = f sqrt(0.2357 mL/(M+0.2357 mL)); 33/140=0.2357
rat_R = np.sqrt((33/140)/(alpha+33/140))
# exact transcendental: 1+cos cosh - alpha*bL*(sin cosh - cos sinh)=0
def charM(bl, al=alpha):
    return 1+np.cos(bl)*np.cosh(bl) - al*bl*(np.sin(bl)*np.cosh(bl)-np.cos(bl)*np.sinh(bl))
bLM = [brentq(charM, a, b) for a,b in [(0.5,1.875),(3.5,4.694),(6.5,7.855)]]
rat_exact = (bLM[0]/bL[0])**2
f_with_M = [f_bare[n]*(bLM[n]/bL[n])**2 for n in range(3)]

# redesign: enlarge d so that f1 (with nacelle) >= f_req; t scaled to hold amplitude
def f1_with_nacelle(d, t):
    mL = rho*np.pi*d*t*Lt
    al = Mn/mL
    bl1 = brentq(lambda b: charM(b, al), 0.3, 1.875)
    return f_n_tube(d,0)*(bl1/bL[0])**2
d_new = brentq(lambda d: f1_with_nacelle(d, t_req)-f_req, d_t, 3*d_t)
# recheck amplitude with nacelle: use modal approach with tip mass (first mode dominant)
def tip_amp_nac(d, t, zeta=0.0):
    mu = rho*np.pi*d*t
    mL = mu*Lt; al = Mn/mL
    tot=0; out={}
    bls = [brentq(lambda b: charM(b, al), lo, hi) for lo,hi in [(0.3,1.875),(3.5,4.694),(6.5,7.855)]]
    for lab,fex in (('1P',f1P),('3P',f3P)):
        Om=2*np.pi*fex; X=0
        for n in range(3):
            blm = bls[n]
            b = blm/Lt
            sig=(np.sinh(blm)-np.sin(blm))/(np.cosh(blm)+np.cos(blm))
            # with tip mass the shape constant differs; use exact tip-mass shape
            # shape: cosh-cos - s*(sinh-sin) with s from BC EI y''' = -M w^2 y at tip -> use ratio from y''(L)=0
            s = (np.cosh(blm)+np.cos(blm))/(np.sinh(blm)+np.sin(blm))
            ph = lambda x: (np.cosh(b*x)-np.cos(b*x)-s*(np.sinh(b*x)-np.sin(b*x)))
            ptip = ph(Lt)
            phn = lambda x: ph(x)/ptip
            mn = mu*np.trapezoid(phn(xs)**2,xs) + Mn*1.0
            wn = 2*np.pi*f_n_tube(d,n)*(blm/bL[n])**2
            r=Om/wn
            X += P*phn(a_load)/(mn*wn**2)/np.sqrt((1-r**2)**2+(2*zeta*r)**2)
        out[lab]=X; tot+=X
    out['sum']=tot; return out
t_new = brentq(lambda t: tip_amp_nac(d_new,t)['sum']-X_allow, 0.005, 0.5)
f_new = [f1_with_nacelle(d_new,t_new)]
mL_new = rho*np.pi*d_new*t_new*Lt
al_new = Mn/mL_new
bls_new = [brentq(lambda b: charM(b, al_new), lo, hi) for lo,hi in [(0.3,1.875),(3.5,4.694),(6.5,7.855)]]
f_new = [f_n_tube(d_new,n)*(bls_new[n]/bL[n])**2 for n in range(3)]

res = dict(L=L,Lt=Lt,Mr=Mr,Mn=Mn,E=E,G=G,rho=rho,Vtip=Vtip,Omega_b=Omega_b,f1P=f1P,f3P=f3P,f_req=f_req,
  bL=bL,d_t=d_t,f_bare=f_bare,modal_int=modal_int,P10=P10,kP=kP,P=P,a_load=a_load,
  phi_a=[phi(a_load,n) for n in range(3)],X_allow=X_allow,t_req=t_req,amp_at_t=amp_at_t,
  I_t=I_t,delta_st=delta_st,DAF1=DAF1,DAF3=DAF3,tower_mass=tower_mass,alpha=alpha,
  rat_R=rat_R,rat_exact=rat_exact,bLM=bLM,f_with_M=f_with_M,
  d_new=d_new,t_new=t_new,f_new=f_new,al_new=al_new,mL_new=mL_new,
  amp_new=tip_amp_nac(d_new,t_new), amp_new_1pc=tip_amp_nac(d_new,t_new,0.01),
  amp_bare_1pc=tip_amp(t_req,zeta=0.01))
for k,v in res.items(): print(f"{k:12s} {v}")
json.dump(res, open('analytical.json','w'), indent=1, default=float)

# ---- joint iteration for redesign ----
d_i, t_i = d_new, t_new
for it in range(30):
    d_i = brentq(lambda d: f1_with_nacelle(d, t_i)-f_req, 0.5*d_i, 3*d_i)
    t_i = brentq(lambda t: tip_amp_nac(d_i,t)['sum']-X_allow, 0.005, 0.5)
mL_i = rho*np.pi*d_i*t_i*Lt; al_i = Mn/mL_i
bls_i = [brentq(lambda b: charM(b, al_i), lo, hi) for lo,hi in [(0.3,1.875),(3.5,4.694),(6.5,7.855)]]
f_i = [f_n_tube(d_i,n)*(bls_i[n]/bL[n])**2 for n in range(3)]
res.update(d_final=d_i, t_final=t_i, mL_final=mL_i, al_final=al_i, f_final=f_i,
           amp_final=tip_amp_nac(d_i,t_i), amp_final_1pc=tip_amp_nac(d_i,t_i,0.01),
           bL_final=bls_i, dt_ratio=d_i/t_i, dt_ratio_b21=d_t/t_req)
print("FINAL d,t,f:", d_i, t_i, f_i, "alpha", al_i, "amp", res['amp_final'])
json.dump(res, open('analytical.json','w'), indent=1, default=float)

# ---- design action: keep t = t_req, raise d until f1(with nacelle) = f_req ----
d_act = brentq(lambda d: f1_with_nacelle(d, t_req)-f_req, d_t, 3*d_t)
mL_a = rho*np.pi*d_act*t_req*Lt; al_a = Mn/mL_a
bls_a = [brentq(lambda b: charM(b, al_a), lo, hi) for lo,hi in [(0.3,1.875),(3.5,4.694),(6.5,7.855)]]
f_a = [f_n_tube(d_act,n)*(bls_a[n]/bL[n])**2 for n in range(3)]
amp_a = tip_amp_nac(d_act, t_req); amp_a1 = tip_amp_nac(d_act,t_req,0.01)
res.update(d_act=d_act, t_act=t_req, mL_act=mL_a, al_act=al_a, f_act=f_a, bL_act=bls_a,
           amp_act=amp_a, amp_act_1pc=amp_a1, dt_act=d_act/t_req, dt_b21=d_t/t_req,
           tower_mass_act=mL_a)
print("ACTION d=%.3f t=%.4f d/t=%.0f f=%s alpha=%.4f amp=%s massT=%.0f t" % (d_act,t_req,d_act/t_req,f_a,al_a,amp_a,mL_a/1e3))
json.dump(res, open('analytical.json','w'), indent=1, default=float)
