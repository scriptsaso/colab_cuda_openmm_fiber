import matplotlib.pyplot as plt
import numpy as np
from analysis_common import (
    load_traj, FRAME_TEMP_MAP, pca_first_axis,
    MONOMER_ATOM_COUNT, FIBER_MONOMER_COUNT, ATOM_N_IN_MONOMER
)

traj = load_traj()
all_dist = []
temps = []

for frame, temp in FRAME_TEMP_MAP.items():
    Npos = []
    for i in range(FIBER_MONOMER_COUNT):
        base = i * MONOMER_ATOM_COUNT
        Npos.append(traj.xyz[frame, base + ATOM_N_IN_MONOMER, :])
    Npos = np.array(Npos)

    axis = pca_first_axis(Npos)
    proj = np.dot(Npos, axis)
    dist = np.abs(proj[1:] - proj[:-1])

    all_dist.append(dist)
    temps.append(temp)

data = np.array(all_dist)
median = np.median(data, axis=1)
q1 = np.percentile(data, 25, axis=1)
q3 = np.percentile(data, 75, axis=1)

plt.figure(figsize=(3.5, 2.5))
plt.plot(temps, median, marker="D", label="Median stacking distance")
plt.fill_between(temps, q1, q3, alpha=0.3, label="Interquartile range")
plt.xlim(55, 5)
plt.ylim(0.275, 0.425)
plt.xlabel("Temperature (°C)", fontsize=10)
plt.ylabel("π–π stacking distance (nm)", fontsize=10)
plt.xticks(fontsize=9); plt.yticks(fontsize=9)
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/pi_pi_stacking_distance_vs_temperature.png", dpi=300)
plt.close()

print("Wrote: /content/pi_pi_stacking_distance_vs_temperature.png")
