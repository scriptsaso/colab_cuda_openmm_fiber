from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, PME, HBonds, StateDataReporter
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picosecond, nanometer
import sys

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
sim.reporters.append(StateDataReporter(sys.stdout, 500, step=True, temperature=True, potentialEnergy=True))
sim.step(2000)
print("OK: 2000 steps completed.")
