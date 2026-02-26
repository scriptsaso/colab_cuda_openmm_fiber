import matplotlib.pyplot as plt
import numpy as np
from analysis_common import (
    load_traj, FRAME_TEMP_MAP, compute_dihedral, wrap180,
    MONOMER_ATOM_COUNT, FIBER_MONOMER_COUNT,
    ATOM_1_REL, ATOM_2_REL, ATOM_3_REL, ATOM_4_REL
)

traj = load_traj()
temps_sorted = sorted(FRAME_TEMP_MAP.items(), key=lambda x: -x[1])

heat = []
for frame, temp in temps_sorted:
    p = traj.xyz[frame]
    row = []
    for i in range(FIBER_MONOMER_COUNT - 1):
        base = i * MONOMER_ATOM_COUNT
        nxt  = (i + 1) * MONOMER_ATOM_COUNT
        d = compute_dihedral(
            p[base + ATOM_1_REL],
            p[base + ATOM_2_REL],
            p[nxt  + ATOM_3_REL],
            p[nxt  + ATOM_4_REL],
        )
        row.append(wrap180(d))
    heat.append(row)

heat = np.array(heat)

plt.figure(figsize=(8, 4))
im = plt.imshow(
    heat, aspect="auto", cmap="coolwarm", vmin=-120, vmax=120,
    extent=[0, FIBER_MONOMER_COUNT - 1, temps_sorted[-1][1], temps_sorted[0][1]]
)
cbar = plt.colorbar(im)
cbar.set_label("Dihedral angle (°) signed", fontsize=14)
cbar.ax.tick_params(labelsize=13)
cbar.set_ticks(np.arange(-120, 121, 40))

plt.axhline(y=16, color="black", linestyle="--", linewidth=1.5, label="16 °C threshold")
plt.xlabel("Fiber segment index (Monomer N to N+1)", fontsize=14)
plt.ylabel("Temperature (°C)", fontsize=14)
plt.xticks(ticks=np.arange(0, FIBER_MONOMER_COUNT, 1),
           labels=[str(i) for i in range(FIBER_MONOMER_COUNT)], fontsize=12)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("/content/fiber_segment_dihedral_heatmap.png", dpi=300)
plt.close()

print("Wrote: /content/fiber_segment_dihedral_heatmap.png")
