import mdtraj as md
import numpy as np

traj = md.load("/content/fiber_ramp_50C_to_10C.dcd", top="/content/fiber_21mol_TOL.prmtop")

fiber_atoms = 21 * 52
fiber = traj.atom_slice(np.arange(fiber_atoms))

fiber.save_dcd("/content/fiber_only.dcd")
fiber[0].save_pdb("/content/fiber_only.pdb")

print("Wrote /content/fiber_only.dcd and /content/fiber_only.pdb")
print("Frames:", fiber.n_frames, "Atoms:", fiber.n_atoms)
