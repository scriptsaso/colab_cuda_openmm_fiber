# colab_cuda_openmm_fiber

CUDA-accelerated OpenMM workflow (Google Colab) for a GAFF-parametrized supramolecular fiber (21 monomers) in explicit solvent.  
This repository supports thesis-grade simulations and analysis by providing a reproducible GPU pipeline and a compact set of morphology-relevant descriptors.

## Scope
This project focuses on:
- Running molecular dynamics (MD) of a pre-assembled 21-monomer fiber in explicit toluene using OpenMM (CUDA).
- Extracting structural descriptors relevant for supramolecular packing (torsion/dihedral, axial stacking distance, lateral displacement).
- Producing visualization-ready trajectories for external viewers (ChimeraX/VMD).

## Inputs (generated upstream)
The MD simulations in this repo assume that the following Amber/GAFF files already exist:
- `fiber_21mol_TOL.prmtop`
- `fiber_21mol_TOL.inpcrd`

These files are generated in a separate upstream preparation workflow (AmberTools/GAFF + tleap), where:
- the 21-monomer fiber geometry is constructed,
- GAFF parameters are assigned,
- the system is solvated in explicit toluene,
- and the final solvated topology/coordinates are exported as `prmtop/inpcrd`.

Large inputs/outputs are intentionally excluded from version control via `.gitignore`.

## Reproducible environment
The Colab environment is pinned to OpenMM 8.3.* to avoid CUDA/PTX driver mismatch commonly observed on Colab T4 runtimes:
- `environment/env.yml`

## What is included
### Simulation
- `scripts/smoke_test_cuda.py`  
  Minimal CUDA sanity check (minimize + short run) to verify the CUDA platform is active.

- `scripts/run_ramp_cuda.py`  
  Production MD protocol on CUDA:
  - energy minimization  
  - equilibration: 50 ps @ 323.15 K (50 °C)  
  - temperature ramp: 323.15 → 283.15 K (50 → 10 °C) over 500 ps (block stepping for GPU efficiency)  
  Outputs (written to `/content` in Colab):
  - `fiber_ramp_50C_to_10C.dcd`
  - `fiber_ramp_50C_to_10C.log`
  - `fiber_10C_final.pdb`

### Analysis (thesis-relevant descriptors)
- `scripts/analysis_common.py`  
  Shared utilities and constants (trajectory loading, signed dihedral, PCA axis via SVD).

- `scripts/analysis_dihedrals.py`  
  Segmental and fiber-averaged signed dihedral vs temperature using a fixed frame–temperature map.

- `scripts/analysis_dihedral_heatmap.py`  
  Segment-resolved dihedral heatmap along the fiber axis (local torsional heterogeneity).

- `scripts/analysis_stacking_pca.py`  
  PCA-axis (SVD) projected axial stacking distances vs temperature (median + IQR).

- `scripts/analysis_lateral_shift.py`  
  PCA-axis projected lateral displacement vs temperature (median + IQR).

### Visualization export
- `scripts/make_fiber_dcd_top.py`  
  Creates fiber-only visualization files:
  - `fiber_only.pdb` (topology)
  - `fiber_only.dcd` (trajectory)
  Recommended workflow for ChimeraX/VMD: open PDB first, then load DCD onto the same model.

## Performance benchmark (Colab Tesla T4)
Measured on the same solvated system (PME, 2 fs timestep, HBonds constraints):
- CPU: 33.16 ns/day  
- CUDA (mixed precision): 1348.09 ns/day  
- Speedup: 40.66×

Script:
- `scripts/benchmark_openmm.py`

## Notes on version control
Trajectory/topology outputs (e.g., `.dcd`, `.prmtop`, `.inpcrd`, `.pdb`) are excluded via `.gitignore` to keep the repository lightweight.  
For sharing large artifacts, use external storage (Drive/Zenodo/OSF) and link them from the README if needed.
