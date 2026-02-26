import os
os.environ.pop("MPLBACKEND", None)
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import mdtraj as md

TRAJ_PATH = "/content/fiber_ramp_50C_to_10C.dcd"
TOP_PATH  = "/content/fiber_21mol_TOL.prmtop"

FRAME_TEMP_MAP = {
    0: 50,
    125: 40,
    249: 30,
    374: 20,
    399: 18,
    412: 17,
    424: 16,
    437: 15,
    449: 14,
    474: 12,
    499: 10
}

MONOMER_ATOM_COUNT = 52
FIBER_MONOMER_COUNT = 21

ATOM_1_REL = 20
ATOM_2_REL = 19
ATOM_3_REL = 71 - MONOMER_ATOM_COUNT
ATOM_4_REL = 72 - MONOMER_ATOM_COUNT

ATOM_SINGLE = (20, 19, 71, 72)
ATOM_N_IN_MONOMER = 3

def load_traj():
    return md.load(TRAJ_PATH, top=TOP_PATH)

def compute_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def wrap180(angle_deg):
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360
    return angle_deg

def pca_first_axis(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    axis = vt[0]
    axis /= np.linalg.norm(axis)
    return axis
