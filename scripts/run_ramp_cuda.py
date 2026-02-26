from openmm.app import (
    AmberPrmtopFile, AmberInpcrdFile, Simulation,
    PME, HBonds, DCDReporter, StateDataReporter, PDBFile
)
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picosecond, nanometer

prmtop = AmberPrmtopFile("/content/fiber_21mol_TOL.prmtop")
inpcrd = AmberInpcrdFile("/content/fiber_21mol_TOL.inpcrd")

system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(323.15*kelvin, 0.3/picosecond, 0.002*picosecond)

platform = Platform.getPlatformByName("CUDA")
props = {"Precision": "mixed"}

sim = Simulation(prmtop.topology, system, integrator, platform, props)
sim.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

print("Using platform:", sim.context.getPlatform().getName())

sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(323.15*kelvin)

print("Equilibration: 50 ps @ 323.15 K")
sim.step(25000)

sim.reporters.append(DCDReporter("/content/fiber_ramp_50C_to_10C.dcd", 500))
sim.reporters.append(StateDataReporter(
    "/content/fiber_ramp_50C_to_10C.log", 500,
    step=True, temperature=True, potentialEnergy=True, density=True, speed=True
))

n_steps = 250000
start_temp = 323.15
end_temp = 283.15

block_size = 250
n_blocks = n_steps // block_size
assert n_blocks * block_size == n_steps

print(f"Ramp: {start_temp:.2f} K -> {end_temp:.2f} K | steps={n_steps} | blocks={n_blocks} | block={block_size}")

for b in range(n_blocks):
    frac = b/(n_blocks-1) if n_blocks > 1 else 1.0
    T = start_temp + frac*(end_temp-start_temp)
    integrator.setTemperature(T*kelvin)
    sim.step(block_size)

state = sim.context.getState(getPositions=True)
pos = state.getPositions()

with open("/content/fiber_10C_final.pdb", "w") as f:
    PDBFile.writeFile(prmtop.topology, pos, f)

print("DONE. Outputs written to /content.")
