"""
MECH3200 B2.4 -- planar finite-element model of the floating offshore wind turbine (SN*=91).

Idealisation (x = surge/horizontal, y = vertical, theta = pitch about z):
  * Platform   : rigid body, DOF (X, Y, Th) at its centre of mass (COM).
                 Supported on the four A2 mounts at (+-b, -a) -> 4kx surge, 4ky heave,
                 coupling 4kx a, rotational 4kx a^2 + 4ky b^2  (A2.1 derivation).
  * Tower      : 2-node Timoshenko beam elements (bending + shear + rotary inertia),
                 base node rigidly tied to the platform at y = +10 m above the COM
                 (project table: COM 10 m below tower base).  Axial DOF suppressed.
  * Nacelle    : point mass Mn at the tower tip.
  * Power cable: tensioned Euler/Timoshenko beam elements with geometric stiffness T K_g,
                 top end pinned at the platform COM (translation tied, rotation free),
                 bottom end pinned to the seabed.  Modelled hanging vertically.
  * Damping    : 1 % modal (applies to forced response only).
Outputs: natural frequencies, mode shapes (plots), forced tip response, comparison table.
"""
import numpy as np, json
from scipy.linalg import eigh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open('analytical.json'))
SN = 91
# ---------------- parameters ----------------
Lt   = A['Lt']; E = A['E']; G = A['G']; rho = A['rho']
Mn   = A['Mn']; Mr = A['Mr']
d_t  = A['d_act']; t_w = A['t_act']          # tower design after B2.3 action
I_t  = np.pi*d_t**3*t_w/8;  A_t = np.pi*d_t*t_w
ks_t = 0.5                                  # shear coefficient, thin circular tube
# platform (A2)
M_tot = 16e3*(1+SN/300)*1e3                 # 2.0853e7 kg (turbine+tower+platform)
kxy   = 41.0                                # radius of gyration, pitch about z
a_m, b_m = 7.0, 40.0
rho_sw = 1025.0; g = 9.81
kx = 0.5e6; ky = rho_sw*g*1000.0
y_base = 10.0                               # tower base above COM
# cable (B1)
Lpc = 1.1*A['L']; dpc = 0.24; rhopc = 3000.0; Epc = 10e9; Tpc = 400e3; nupc = 0.25
Gpc = Epc/(2*(1+nupc)); Ipc = np.pi*dpc**4/64; Apc = np.pi*dpc**2/4; ks_c = 0.9
mu_c = rhopc*Apc
# excitation
f1P, f3P, P, a_load = A['f1P'], A['f3P'], A['P'], A['a_load']
zeta_m = 0.01

# ---------------- element matrices ----------------
def timo_elem(EI, kGA, mu, rI, Le, T=0.0):
    """2-node Timoshenko beam (v1,th1,v2,th2); consistent mass incl. rotary inertia;
       geometric stiffness for axial tension T."""
    phi = 12*EI/(kGA*Le**2) if kGA>0 else 0.0
    c = EI/((1+phi)*Le**3)
    K = c*np.array([[12, 6*Le, -12, 6*Le],
                    [6*Le, (4+phi)*Le**2, -6*Le, (2-phi)*Le**2],
                    [-12, -6*Le, 12, -6*Le],
                    [6*Le, (2-phi)*Le**2, -6*Le, (4+phi)*Le**2]])
    # translational consistent mass (Timoshenko, Przemieniecki)
    m = mu*Le/(1+phi)**2
    Mt = m*np.array([
        [13/35+7/10*phi+1/3*phi**2, (11/210+11/120*phi+1/24*phi**2)*Le, 9/70+3/10*phi+1/6*phi**2, -(13/420+3/40*phi+1/24*phi**2)*Le],
        [0, (1/105+1/60*phi+1/120*phi**2)*Le**2, (13/420+3/40*phi+1/24*phi**2)*Le, -(1/140+1/60*phi+1/120*phi**2)*Le**2],
        [0, 0, 13/35+7/10*phi+1/3*phi**2, -(11/210+11/120*phi+1/24*phi**2)*Le],
        [0, 0, 0, (1/105+1/60*phi+1/120*phi**2)*Le**2]])
    Mt = Mt + Mt.T - np.diag(np.diag(Mt))
    # rotary inertia
    mr = rI/(Le*(1+phi)**2)
    Mr_ = mr*np.array([
        [6/5, (1/10-1/2*phi)*Le, -6/5, (1/10-1/2*phi)*Le],
        [0, (2/15+1/6*phi+1/3*phi**2)*Le**2, -(1/10-1/2*phi)*Le, -(1/30+1/6*phi-1/6*phi**2)*Le**2],
        [0, 0, 6/5, -(1/10-1/2*phi)*Le],
        [0, 0, 0, (2/15+1/6*phi+1/3*phi**2)*Le**2]])
    Mr_ = Mr_ + Mr_.T - np.diag(np.diag(Mr_))
    Kg = T/(30*Le)*np.array([[36, 3*Le, -36, 3*Le],
                              [3*Le, 4*Le**2, -3*Le, -Le**2],
                              [-36, -3*Le, 36, -3*Le],
                              [3*Le, -Le**2, -3*Le, 4*Le**2]])
    return K+Kg, Mt+Mr_

def assemble(n_t=40, n_c=40, timoshenko=True, flexible_base=True, cable=True, nacelle=True,
             d=d_t, t=t_w, cable_T=Tpc, kx_=kx):
    I = np.pi*d**3*t/8; Ar = np.pi*d*t; mu_t = rho*Ar
    EI, kGA, rI = E*I, (ks_t*G*Ar if timoshenko else 1e300), (rho*I if timoshenko else 0.0)
    # DOF numbering: platform 0,1,2 (X,Y,Th); tower nodes i: v=3+2i, th=4+2i (i=0..n_t)
    nd_t = n_t+1; off_c = 3+2*nd_t; nd_c = n_c+1
    ndof = off_c + 2*nd_c
    K = np.zeros((ndof,ndof)); M = np.zeros((ndof,ndof))
    # platform
    Mp = M_tot - (mu_t*Lt + (Mn if nacelle else 0.0))    # avoid double-counting tower+nacelle in M_tot
    # NOTE: the table's total radius of gyration (41 m) is smaller than the tower alone would give
    # about the COM, so J cannot be split consistently; the table value is assigned to the platform
    # hull and the tower/nacelle inertia is added explicitly by the FE elements.
    Jp = M_tot*kxy**2
    J_tower_about_COM = mu_t*Lt*((Lt/2+y_base)**2 + Lt**2/12) + (Mn if nacelle else 0.0)*(Lt+y_base)**2
    M[0,0] += Mp; M[1,1] += M_tot; M[2,2] += Jp   # heave: tower axially rigid, whole mass heaves together
    Kp = np.array([[4*kx_, 0, 4*kx_*a_m],[0, 4*ky, 0],[4*kx_*a_m, 0, 4*kx_*a_m**2+4*ky*b_m**2]])
    K[:3,:3] += Kp
    # tower
    Le = Lt/n_t
    for e in range(n_t):
        ke, me = timo_elem(EI, kGA, mu_t, rI, Le)
        idx = [3+2*e, 4+2*e, 5+2*e, 6+2*e]
        K[np.ix_(idx,idx)] += ke; M[np.ix_(idx,idx)] += me
    if nacelle:
        M[3+2*n_t, 3+2*n_t] += Mn
    # cable (hangs vertically from COM; transverse DOF = x direction)
    if cable:
        Lc = Lpc/n_c
        EIc = Epc*Ipc; kGAc = ks_c*Gpc*Apc if timoshenko else 1e300; rIc = rhopc*Ipc if timoshenko else 0.0
        for e in range(n_c):
            ke, me = timo_elem(EIc, kGAc, mu_c, rIc, Lc, T=cable_T)
            idx = [off_c+2*e, off_c+2*e+1, off_c+2*e+2, off_c+2*e+3]
            K[np.ix_(idx,idx)] += ke; M[np.ix_(idx,idx)] += me
    # ---- constraints (master-slave transformation) ----
    # tower base: v_base = X - y_base*Th, th_base = Th
    # cable top : v_top = X (pinned at COM), rotation free; cable bottom: v=0 (pinned)
    keep = list(range(ndof))
    Tm = np.eye(ndof)
    slaves = []
    if flexible_base:
        Tm[3,:] = 0; Tm[3,0] = 1; Tm[3,2] = -y_base;  slaves.append(3)
        Tm[4,:] = 0; Tm[4,2] = 1;                     slaves.append(4)
        if cable:
            Tm[off_c,:] = 0; Tm[off_c,0] = 1; slaves.append(off_c)
            slaves.append(off_c+2*n_c)              # bottom pinned: v=0
    else:
        slaves += [0,1,2,3,4]                       # fixed base, platform removed
        if cable:
            slaves += [off_c, off_c+2*n_c]          # cable pinned at a fixed point
    if not cable:
        slaves += list(range(off_c, ndof))
    masters = [i for i in keep if i not in slaves]
    Tr = Tm[:, masters]
    Kr = Tr.T@K@Tr; Mr_ = Tr.T@M@Tr
    return Kr, Mr_, Tr, dict(J_tower=J_tower_about_COM, ndof=ndof, n_t=n_t, n_c=n_c, off_c=off_c, masters=masters, Le=Le,
                              tip_v=3+2*n_t, cable_Lc=Lpc/n_c)

def modes(Kr, Mr_, nmodes=12):
    w2, V = eigh(Kr, Mr_)
    w2 = np.clip(w2, 0, None)
    f = np.sqrt(w2)/(2*np.pi)
    return f[:nmodes], V[:, :nmodes]

def describe(f, V, Tr, info, label):
    """classify each mode by energy share: platform surge/heave/pitch, tower bending, cable"""
    out=[]
    full = Tr@V
    n_t, off_c, n_c = info['n_t'], info['off_c'], info['n_c']
    for i in range(len(f)):
        u = full[:,i]
        plat = u[:3]; tow_v = u[3:3+2*(n_t+1):2]; cab_v = u[off_c:off_c+2*(n_c+1):2] if n_c else np.zeros(1)
        # tower elastic deflection relative to rigid platform motion
        ys = np.linspace(0,Lt,n_t+1)+ (0)  # tower local coordinate
        rigid = plat[0] - (ys+10.0)*plat[2]
        tow_el = tow_v - rigid
        cab_rel = cab_v - plat[0]*np.ones_like(cab_v)
        scale = max(abs(u).max(),1e-30)
        comps = dict(X=abs(plat[0]), Y=abs(plat[1]), Th=abs(plat[2])*Lt,
                     tower=abs(tow_el).max(), cable=abs(cab_rel).max())
        dom = max(comps, key=comps.get)
        out.append((i+1, f[i], dom, comps))
    return out

# =============== runs ===============
results = {}
# (a) fixed-base tower only, Euler (validation against B2.1/B2.3 analytical)
Kr,Mr_,Tr,info = assemble(timoshenko=False, flexible_base=False, cable=False, nacelle=False)
f,V = modes(Kr,Mr_,3); results['fixed_EB_noNacelle'] = f.tolist()
Kr,Mr_,Tr,info = assemble(timoshenko=False, flexible_base=False, cable=False, nacelle=True)
f,V = modes(Kr,Mr_,3); results['fixed_EB_nacelle'] = f.tolist()
Kr,Mr_,Tr,info = assemble(timoshenko=True, flexible_base=False, cable=False, nacelle=True)
f,V = modes(Kr,Mr_,3); results['fixed_Timo_nacelle'] = f.tolist()
# (b) B2.1 geometry (d_t, t_req) fixed base, no nacelle -> check vs analytical 0.357
Kr,Mr_,Tr,info = assemble(timoshenko=False, flexible_base=False, cable=False, nacelle=False, d=A['d_t'], t=A['t_req'])
f,V = modes(Kr,Mr_,3); results['B21_geom_fixed_EB'] = f.tolist()
Kr,Mr_,Tr,info = assemble(timoshenko=False, flexible_base=False, cable=False, nacelle=True, d=A['d_t'], t=A['t_req'])
f,V = modes(Kr,Mr_,3); results['B21_geom_fixed_EB_nacelle'] = f.tolist()
# (c) cable only, pinned-pinned, tensioned (validation vs B1.3)
Kr,Mr_,Tr,info = assemble(timoshenko=False, flexible_base=False, cable=True, nacelle=False)
f,V = modes(Kr,Mr_,12)
desc = describe(f,V,Tr,info,'cab')
results['cable_pp_tensioned'] = [ff for (i,ff,dm,c) in desc if dm=='cable'][:8]
# (d) full flexible-base model, Timoshenko, tension, nacelle, cable
Kr,Mr_,Tr,info = assemble(timoshenko=True, flexible_base=True, cable=True, nacelle=True)
f,V = modes(Kr,Mr_,16)
full_desc = describe(f,V,Tr,info,'full')
results['full'] = [(i,float(ff),dm,{k:float(v) for k,v in c.items()}) for (i,ff,dm,c) in full_desc]
# (e) flexible base, Euler (no shear/rotary) for the Timoshenko comparison
Kr_e,Mr_e,Tr_e,info_e = assemble(timoshenko=False, flexible_base=True, cable=True, nacelle=True)
f_e,V_e = modes(Kr_e,Mr_e,16)
results['full_EB'] = [(i,float(ff),dm) for (i,ff,dm,c) in describe(f_e,V_e,Tr_e,info_e,'eb')]
# (f) flexible base, no cable tension (to show tension effect)
Kr0,Mr0,Tr0,info0 = assemble(timoshenko=True, flexible_base=True, cable=True, nacelle=True, cable_T=0.0)
f0,V0 = modes(Kr0,Mr0,16)
results['full_noT'] = [(i,float(ff),dm) for (i,ff,dm,c) in describe(f0,V0,Tr0,info0,'noT')]
# (g) A2.3 repeat: mooring stiffness scaled to 17.2 % (surge/sway cluster)
Kr3,Mr3,Tr3,info3 = assemble(timoshenko=True, flexible_base=True, cable=True, nacelle=True, kx_=0.314*kx)
f3,V3 = modes(Kr3,Mr3,16)
results['full_kx017'] = [(i,float(ff),dm) for (i,ff,dm,c) in describe(f3,V3,Tr3,info3,'kx')]

# ---------- forced response: tower tip under P at a_load, 1 % modal damping ----------
def forced_tip(Kr,Mr_,Tr,info, fex, zeta=zeta_m, nm=16):
    w2,V = eigh(Kr,Mr_)
    wn = np.sqrt(np.clip(w2,0,None))
    full_V = Tr@V
    # load vector: point force at a_load on tower -> distribute to nearest node via shape? use nodal lumping to node
    n_t = info['n_t']; Le = info['Le']
    node = int(round(a_load/Le)); F = np.zeros(info['ndof']); F[3+2*node] = P
    Fr = Tr.T@F
    Om = 2*np.pi*fex
    q = np.zeros(len(wn), dtype=complex)
    for i in range(len(wn)):
        mi = V[:,i]@Mr_@V[:,i]; fi = V[:,i]@Fr
        q[i] = fi/(mi*(wn[i]**2 - Om**2 + 2j*zeta*wn[i]*Om))
    u = full_V@q
    tip = u[info['tip_v']]
    # tip relative to platform rigid motion
    rel = tip - (u[0] - (Lt+10.0)*u[2]) if info['masters'][0]==0 else tip
    return abs(tip), abs(rel)
Kr,Mr_,Tr,info = assemble(timoshenko=True, flexible_base=True, cable=True, nacelle=True)
tip1,rel1 = forced_tip(Kr,Mr_,Tr,info,f1P); tip3,rel3 = forced_tip(Kr,Mr_,Tr,info,f3P)
results['forced_flex'] = dict(tip_1P=tip1, tip_3P=tip3, rel_1P=rel1, rel_3P=rel3, sum_abs=tip1+tip3, sum_rel=rel1+rel3)
Kf,Mf,Tf,inff = assemble(timoshenko=True, flexible_base=False, cable=False, nacelle=True)
tip1f,_ = forced_tip(Kf,Mf,Tf,inff,f1P); tip3f,_ = forced_tip(Kf,Mf,Tf,inff,f3P)
results['forced_fixed'] = dict(tip_1P=tip1f, tip_3P=tip3f, sum=tip1f+tip3f)
# transmissibility of surge-dominated mode at 12 s wave (A2.2/A2.3 repeat) using FE surge freq
def surge_freq(res_list):
    for (i,ff,dm,*rest) in res_list:
        if dm=='X': return ff
fs = surge_freq(results['full']); fs3 = surge_freq(results['full_kx017'])
def T_tr(r,z=0.04): return np.sqrt((1+(2*z*r)**2)/((1-r**2)**2+(2*z*r)**2))
fw = 1/12
results['surge_T'] = dict(f_surge=fs, r=fw/fs, T=T_tr(fw/fs), f_surge_017=fs3, r_017=fw/fs3, T_017=T_tr(fw/fs3))
# ---- effective modal mass at tower tip for the first tower-dominated coupled mode, TMD sizing (A2.5 rule) ----
w2,Vall = eigh(Kr,Mr_); fullV=Tr@Vall
kt_ = [k for k,(i,ff,dm,c) in enumerate(full_desc) if dm=='tower'][0]
phi_ = Vall[:,kt_]; m_mod = phi_@Mr_@phi_; phitip = (fullV[:,kt_])[info['tip_v']] - ( (fullV[:,kt_])[0] - (Lt+10)*(fullV[:,kt_])[2] )
m_eff_tip = m_mod/phitip**2
f_c = full_desc[kt_][1]; mu_tmd = 0.789
results['tmd'] = dict(f_coupled=f_c, m_eff_tip=float(m_eff_tip), mu=mu_tmd, m_a=float(mu_tmd*m_eff_tip),
                      k_a=float(mu_tmd*m_eff_tip*(2*np.pi*f_c)**2), J_tower_about_COM=float(info['J_tower']),
                      Jp=float(M_tot*kxy**2), Mp=float(M_tot-(rho*np.pi*d_t*t_w*Lt+Mn)))
json.dump(results, open('fe_results.json','w'), indent=1)
for k,v in results.items(): print(k, v)

# ---------- mode shape plots ----------
full = Tr@V
fig, axes = plt.subplots(2,4, figsize=(14,8)); axes=axes.ravel()
ys = np.linspace(0,Lt,info['n_t']+1)+10.0
yc = -np.linspace(0,Lpc,info['n_c']+1)
sel = 0
for k,(i,ff,dm,c) in enumerate(full_desc[:8]):
    u = full[:,k]; ax=axes[k]
    X,Y,Th = u[:3]
    tow = u[3:3+2*(info['n_t']+1):2]; cab = u[info['off_c']:info['off_c']+2*(info['n_c']+1):2]
    s = 25/max(abs(u).max(),1e-12)
    # platform box
    bx = np.array([-b_m,b_m,b_m,-b_m,-b_m]); by=np.array([-a_m,-a_m,10,10,-a_m])
    bxd = bx + s*(X - by*Th); byd = by + s*(Y + bx*Th)
    ax.plot(bx,by,'k:',lw=0.8); ax.plot(bxd,byd,'k-',lw=1.2)
    ax.plot(np.zeros_like(ys),ys,'b:',lw=0.8); ax.plot(s*tow,ys,'b-',lw=1.5)
    ax.plot(np.zeros_like(yc),yc,'r:',lw=0.8); ax.plot(s*cab,yc,'r-',lw=1.2)
    ax.plot(s*tow[-1], ys[-1],'bo',ms=5)
    ax.set_title(f"Mode {i}: {ff:.3f} Hz\n({dm}-dominated)", fontsize=9)
    ax.set_aspect('equal'); ax.set_xlim(-70,70); ax.set_ylim(-175,200); ax.tick_params(labelsize=7)
fig.suptitle("FE mode shapes, flexible-base model (platform black, tower blue, cable red; exaggerated)", fontsize=11)
plt.tight_layout(); plt.savefig('fe_modes.png', dpi=160); plt.close()

# fixed vs flexible tower first-mode comparison plot
Kf,Mf,Tf,inff = assemble(timoshenko=True, flexible_base=False, cable=False, nacelle=True)
ff_,Vf_ = modes(Kf,Mf,3); uf = (Tf@Vf_)[:,0]; towf = uf[3:3+2*(inff['n_t']+1):2]
# flexible: pick first tower-dominated mode
kt = [k for k,(i,ff,dm,c) in enumerate(full_desc) if dm=='tower'][0]
ut = full[:,kt]; towt = ut[3:3+2*(info['n_t']+1):2]; rigid = ut[0]-(ys)*ut[2]
fig,ax = plt.subplots(figsize=(5,6))
ax.plot(towf/abs(towf).max(), ys-10, 'b-', label=f"fixed base, f={ff_[0]:.3f} Hz")
ax.plot(towt/abs(towt).max(), ys-10, 'r-', label=f"flexible base (total), f={full_desc[kt][1]:.3f} Hz")
ax.plot(rigid/abs(towt).max(), ys-10, 'r--', lw=0.8, label="flexible base: rigid-body part")
ax.set_xlabel("normalised lateral displacement"); ax.set_ylabel("height above tower base (m)")
ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("First tower bending mode")
plt.tight_layout(); plt.savefig('fe_tower_mode1.png', dpi=160); plt.close()

# FRF of tower tip, fixed vs flexible
fr = np.linspace(0.02,1.2,600)
def frf(Kr,Mr_,Tr,info):
    w2,V = eigh(Kr,Mr_); wn=np.sqrt(np.clip(w2,0,None)); fullV=Tr@V
    node=int(round(a_load/info['Le'])); F=np.zeros(info['ndof']); F[3+2*node]=P; Fr=Tr.T@F
    mi=np.einsum('ij,ij->j',V,Mr_@V); fi=V.T@Fr
    out=[]
    for fx in fr:
        Om=2*np.pi*fx; q=fi/(mi*(wn**2-Om**2+2j*zeta_m*wn*Om)); u=fullV@q; out.append(abs(u[info['tip_v']]))
    return np.array(out)
H_flex = frf(Kr,Mr_,Tr,info); H_fix = frf(Kf,Mf,Tf,inff)
fig,ax=plt.subplots(figsize=(8,4.2))
ax.semilogy(fr,H_flex,'r-',label='flexible base (platform + cable), 1 % damping')
ax.semilogy(fr,H_fix,'b-',label='fixed base tower + nacelle, 1 % damping')
for fx,lab in ((f1P,'1P'),(f3P,'3P'),(1/12,'12 s wave')): ax.axvline(fx,color='gray',ls=':',lw=1); ax.text(fx,ax.get_ylim()[1] if False else H_flex.max()*1.5,lab,rotation=90,fontsize=8,va='top')
ax.axhline(0.007*Lt,color='k',ls='--',lw=0.8,label='0.7 % $L_t$ limit')
ax.set_xlabel('forcing frequency (Hz)'); ax.set_ylabel('tower-tip amplitude (m)'); ax.legend(fontsize=8); ax.grid(alpha=.3,which='both')
ax.set_title(f'Tower-tip FRF, P = {P/1e6:.2f} MN at 0.95$L_t$')
plt.tight_layout(); plt.savefig('fe_frf.png',dpi=160); plt.close()
print("plots done")
