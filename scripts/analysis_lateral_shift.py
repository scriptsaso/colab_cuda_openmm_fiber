import matplotlib.pyplot as plt
import numpy as np
from analysis_common import (
    load_traj, FRAME_TEMP_MAP, pca_first_axis,
    MONOMER_ATOM_COUNT, FIBER_MONOMER_COUNT, ATOM_N_IN_MONOMER
)

traj = load_traj()
all_lat = []
temps = []

for frame, temp in FRAME_TEMP_MAP.items():
    Npos = []
    for i in range(FIBER_MONOMER_COUNT):
        base = i * MONOMER_ATOM_COUNT
        Npos.append(traj.xyz[frame, base + ATOM_N_IN_MONOMER, :])
    Npos = np.array(Npos)

    axis = pca_first_axis(Npos)

    shifts = []
    for i in range(FIBER_MONOMER_COUNT - 1):
        delta = Npos[i+1] - Npos[i]
        delta_proj = np.dot(delta, axis) * axis
        lateral_vec = delta - delta_proj
        shifts.append(np.linalg.norm(lateral_vec))
    all_lat.append(shifts)
    temps.append(temp)

data = np.array(all_lat)
median = np.median(data, axis=1)
q1 = np.percentile(data, 25, axis=1)
q3 = np.percentile(data, 75, axis=1)

plt.figure(figsize=(3.5, 2.5))
plt.boxplot(data.T, positions=temps, widths=1.2, patch_artist=True,
            medianprops=dict(color="black"), whiskerprops=dict(color="grey"))
plt.plot(temps, median, marker="D", label="Median lateral shift")
plt.fill_between(temps, q1, q3, alpha=0.3, label="Interquartile range")

plt.xlim(55, 5)
plt.ylim(0, 0.8)
plt.xticks([50, 40, 30, 20, 16, 10], ["50", "40", "30", "20", "16", "10"], fontsize=9)
plt.xlabel("Temperature (°C)", fontsize=10)
plt.ylabel("Lateral shift (nm)", fontsize=10)
plt.yticks(fontsize=9)
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/lateral_shift_vs_temperature.png", dpi=300)
plt.close()

print("Wrote: /content/lateral_shift_vs_temperature.png")
