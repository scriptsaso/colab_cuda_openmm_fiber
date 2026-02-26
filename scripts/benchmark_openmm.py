import time
import platform as py_platform
from dataclasses import dataclass

from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Simulation, PME, HBonds
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picosecond, nanometer

@dataclass
class BenchResult:
    platform: str
    precision: str
    steps: int
    dt_ps: float
    wall_s: float
    ns_per_day: float

def build_sim(prmtop_path: str, inpcrd_path: str, platform_name: str, precision: str):
    prmtop = AmberPrmtopFile(prmtop_path)
    inpcrd = AmberInpcrdFile(inpcrd_path)

    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
    )

    integrator = LangevinIntegrator(323.15 * kelvin, 0.3 / picosecond, 0.002 * picosecond)

    if platform_name == "CUDA":
        platform = Platform.getPlatformByName("CUDA")
        props = {"Precision": precision}
        sim = Simulation(prmtop.topology, system, integrator, platform, props)
    else:
        platform = Platform.getPlatformByName(platform_name)
        sim = Simulation(prmtop.topology, system, integrator, platform)

    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    return sim

def run_bench(sim: Simulation, steps_warmup: int, steps_bench: int, dt_ps: float, platform_name: str, precision: str):
    # Warmup: JIT compile / kernel init / cache fill
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(323.15 * kelvin)
    sim.step(steps_warmup)

    t0 = time.perf_counter()
    sim.step(steps_bench)
    t1 = time.perf_counter()

    wall = t1 - t0
    sim_ns = (steps_bench * dt_ps) / 1000.0  # ps -> ns
    ns_per_day = sim_ns / (wall / 86400.0)

    return BenchResult(
        platform=platform_name,
        precision=precision,
        steps=steps_bench,
        dt_ps=dt_ps,
        wall_s=wall,
        ns_per_day=ns_per_day,
    )

def main():
    prmtop_path = "/content/fiber_21mol_TOL.prmtop"
    inpcrd_path = "/content/fiber_21mol_TOL.inpcrd"

    dt_ps = 0.002  # 2 fs
    steps_warmup = 2000
    steps_bench  = 20000  # keep small enough for Colab, large enough for stable timing

    print("System:", prmtop_path, inpcrd_path)
    print("Python:", py_platform.python_version())
    print("OS:", py_platform.platform())

    # CPU baseline
    sim_cpu = build_sim(prmtop_path, inpcrd_path, platform_name="CPU", precision="n/a")
    r_cpu = run_bench(sim_cpu, steps_warmup, steps_bench, dt_ps, platform_name="CPU", precision="n/a")
    print(f"[CPU]  wall={r_cpu.wall_s:.2f} s | ns/day={r_cpu.ns_per_day:.2f}")

    # CUDA mixed precision
    sim_cuda = build_sim(prmtop_path, inpcrd_path, platform_name="CUDA", precision="mixed")
    r_cuda = run_bench(sim_cuda, steps_warmup, steps_bench, dt_ps, platform_name="CUDA", precision="mixed")
    print(f"[CUDA mixed] wall={r_cuda.wall_s:.2f} s | ns/day={r_cuda.ns_per_day:.2f}")

    speedup = r_cuda.ns_per_day / r_cpu.ns_per_day if r_cpu.ns_per_day > 0 else float("inf")
    print(f"Speedup (CUDA/CPU): {speedup:.2f}x")

if __name__ == '__main__':
    main()
