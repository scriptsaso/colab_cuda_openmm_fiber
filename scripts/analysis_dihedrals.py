import matplotlib.pyplot as plt
import numpy as np
from analysis_common import (
    load_traj, FRAME_TEMP_MAP, compute_dihedral, wrap180,
    MONOMER_ATOM_COUNT, FIBER_MONOMER_COUNT,
    ATOM_1_REL, ATOM_2_REL, ATOM_3_REL, ATOM_4_REL, ATOM_SINGLE
)

traj = load_traj()

# (A) Segmental
a1, a2, a3, a4 = ATOM_SINGLE
temps = []
dihed = []
for frame, temp in FRAME_TEMP_MAP.items():
    p = traj.xyz[frame]
    d = wrap180(compute_dihedral(p[a1], p[a2], p[a3], p[a4]))
    temps.append(temp)
    dihed.append(d)

plt.figure(figsize=(3.5, 2.5))
plt.plot(temps, dihed, marker="o", label="Segmental torsion angle (N–N+1)")
plt.axhline(0, linestyle="--")
plt.axvline(16, linestyle="--", label="16°C threshold")
plt.xlim(55, 2)
plt.ylim(-10, 100)
plt.xlabel("Temperature (°C)", fontsize=10)
plt.ylabel("Torsion Angle (°)", fontsize=10)
plt.xticks(fontsize=9); plt.yticks(fontsize=9)
plt.legend(fontsize=9, loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/dihedral_vs_temperature_map.png", dpi=300)
plt.close()

# (B) Fiber-average (20 interfaces), descending temp order
temps_sorted = sorted(FRAME_TEMP_MAP.items(), key=lambda x: -x[1])
temps_avg = []
avg_vals = []
for frame, temp in temps_sorted:
    p = traj.xyz[frame]
    vals = []
    for i in range(FIBER_MONOMER_COUNT - 1):
        base = i * MONOMER_ATOM_COUNT
        nxt  = (i + 1) * MONOMER_ATOM_COUNT
        d = compute_dihedral(
            p[base + ATOM_1_REL],
            p[base + ATOM_2_REL],
            p[nxt  + ATOM_3_REL],
            p[nxt  + ATOM_4_REL],
        )
        vals.append(wrap180(d))
    temps_avg.append(temp)
    avg_vals.append(float(np.mean(vals)))

plt.figure(figsize=(3.5, 2.5))
plt.plot(temps_avg, avg_vals, marker="o", label="Avg. torsion angle")
plt.axhline(0, linestyle="--")
plt.axvline(16, linestyle="--", label="16°C threshold")
plt.xlim(55, 5)
plt.ylim(-10, 100)
plt.xlabel("Temperature (°C)", fontsize=10)
plt.ylabel("Avg. Torsion Angle (°)", fontsize=10)
plt.xticks(fontsize=9); plt.yticks(fontsize=9)
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/average_dihedral_vs_temperature_map.png", dpi=300)
plt.close()

print("Wrote dihedral plots to /content.")
