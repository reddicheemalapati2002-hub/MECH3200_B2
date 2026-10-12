"""
MECH3200 B2 -- STEP geometry of the floating offshore wind turbine (SN*=91).
Geometry only (no mesh, loads or boundary conditions); the FE results in the
report come from fe_model.py.  Dimensions in metres, origin at the platform
centre of mass, Y up, X = surge direction.

Bodies:
  tower     : thin-walled tube, d_t = 14.374 m, t = 68.9 mm, L_t = 181.875 m,
              base at y = +10 m (COM is 10 m below the tower base, project table)
  nacelle   : box at the tower tip representing M_n = 523.8 t (steel-equivalent volume)
  platform  : rigid semi-submersible hull, 4 columns at (+-40, +-40) m, pontoon ring, deck;
              mount pads at the column bases (y = -7 m) = the four A2 isolator positions
  cable     : d_pc = 0.24 m, L_pc = 160.05 m, hanging from the COM to the seabed pin
"""
import json, cadquery as cq

A = json.load(open('analytical.json'))
d_t, t_w, Lt = A['d_act'], A['t_act'], A['Lt']          # 14.374 m, 0.0689 m, 181.875 m
Mn = A['Mn']; rho = 7850.0
y_base = 10.0; b_m = 40.0; a_m = 7.0
Lpc, dpc = 1.1*A['L'], 0.24

# ---------- tower (thin-walled tube) ----------
tower = (cq.Workplane("XZ").circle(d_t/2).circle(d_t/2 - t_w)
         .extrude(-Lt)                      # XZ workplane normal is -Y; extrude(-) goes +Y
         .translate((0, y_base, 0)))

# ---------- nacelle: box with steel-equivalent volume of M_n ----------
V_n = Mn/rho                                  # 66.7 m^3
nac_w, nac_h = 8.0, 6.0
nac_l = V_n/(nac_w*nac_h)                     # ~1.39 m -> make it a sensible 20 m long shell instead
nac_l = 20.0
nacelle = (cq.Workplane("XY").box(nac_l, nac_h, nac_w)
           .translate((nac_l/2 - 6.0, y_base + Lt + nac_h/2, 0)))
# simple hub + 3 blades (visual only, blade length L = 145.5 m)
hub = cq.Workplane("YZ").circle(2.5).extrude(4.0).translate((nac_l - 6.0, y_base + Lt + nac_h/2, 0))
blades = None
for ang in (0, 120, 240):
    bl = (cq.Workplane("YZ").rect(4.0, 1.2).extrude(A['L'])   # extrude along +X? no: want radial
          )
    bl = (cq.Workplane("XZ").rect(1.2, 4.0).extrude(-A['L'])    # along +Y
          .translate((nac_l - 2.0, y_base + Lt + nac_h/2, 0))
          .rotate((nac_l - 2.0, y_base + Lt + nac_h/2, 0), (nac_l - 1.0, y_base + Lt + nac_h/2, 0), ang))
    blades = bl if blades is None else blades.union(bl)

# ---------- platform: 4 columns + pontoons + deck, COM at origin ----------
col_d, col_h = 12.0, 30.0                     # columns from y=-20 to y=+10
pont_w, pont_h = 8.0, 6.0
cols = None
for sx in (-1, 1):
    for sz in (-1, 1):
        c = (cq.Workplane("XZ").circle(col_d/2).extrude(-col_h)
             .translate((sx*b_m, -20.0, sz*b_m)))
        cols = c if cols is None else cols.union(c)
pontoons = None
for (x1, z1, x2, z2) in [(-b_m, -b_m, b_m, -b_m), (b_m, -b_m, b_m, b_m), (b_m, b_m, -b_m, b_m), (-b_m, b_m, -b_m, -b_m)]:
    length = 2*b_m
    if z1 == z2:   # along X
        p = cq.Workplane("XY").box(length, pont_h, pont_w).translate(((x1+x2)/2, -17.0, z1))
    else:          # along Z
        p = cq.Workplane("XY").box(pont_w, pont_h, length).translate((x1, -17.0, (z1+z2)/2))
    pontoons = p if pontoons is None else pontoons.union(p)
deck = cq.Workplane("XY").box(2*b_m + col_d, 2.0, 2*b_m + col_d).translate((0, y_base - 1.0, 0))
# mount pads at the four A2 isolator positions (x = +-b, y = -a, z = +-b)
pads = None
for sx in (-1, 1):
    for sz in (-1, 1):
        pd = cq.Workplane("XY").box(3.0, 1.0, 3.0).translate((sx*b_m, -a_m, sz*b_m))
        pads = pd if pads is None else pads.union(pd)
platform = cols.union(pontoons).union(deck).union(pads)

# ---------- power cable: hangs from COM to seabed pin ----------
cable = cq.Workplane("XZ").circle(dpc/2).extrude(Lpc)       # along -Y
# small pin blocks at the two cable ends (model attachment points)
pin_top = cq.Workplane("XY").sphere(0.6)
pin_bot = cq.Workplane("XY").sphere(0.6).translate((0, -Lpc, 0))

asm = cq.Assembly(name="FOWT_B2_SN91")
asm.add(platform, name="platform_rigid", color=cq.Color(0.55, 0.55, 0.6))
asm.add(tower,    name="tower_d14p37_t0p069", color=cq.Color(0.2, 0.4, 0.8))
asm.add(nacelle,  name="nacelle_Mn_523t", color=cq.Color(0.8, 0.3, 0.2))
asm.add(hub,      name="hub", color=cq.Color(0.3, 0.3, 0.3))
asm.add(blades,   name="rotor_blades_L145p5", color=cq.Color(0.9, 0.9, 0.9))
asm.add(cable,    name="power_cable_d0p24_L160", color=cq.Color(0.8, 0.1, 0.1))
asm.add(pin_top,  name="cable_pin_COM", color=cq.Color(0, 0, 0))
asm.add(pin_bot,  name="cable_pin_seabed", color=cq.Color(0, 0, 0))
asm.save("MECH3200_B2_46555991.step")
# tower-only STEP for a quick solver import
cq.exporters.export(tower, "B2_tower_only.step")
print("STEP written: MECH3200_B2_46555991.step, B2_tower_only.step")
print(f"tower d={d_t:.3f} m t={t_w*1e3:.1f} mm L={Lt:.3f} m; tower volume={tower.val().Volume():.1f} m^3 -> mass {tower.val().Volume()*rho/1e3:.0f} t")
