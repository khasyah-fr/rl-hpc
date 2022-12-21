import csv
import sys
import pathlib
from time import time

from batsim_py import SimulatorHandler
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor

from easy_backfilling import EASYScheduler
from timeout_policy import TimeoutPolicy
from config import define_args_parser


def run_simulation(shutdown_policy, workload_filename, platform_path, is_baseline):
    simulator = SimulatorHandler()
    if not is_baseline:
        policy = shutdown_policy(simulator)
    scheduler = EASYScheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform=platform_path, workload=workload_filename, verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        simulator.proceed_time()  # proceed directly to the next event.
    simulator.close()
    
    # 4) Return/Dump statistics
    return sim_mon, host_mon, e_mon, simulator

def F(mean_slowdown , consumed_joules, max_consumed_joules, alpha, beta, is_normalized):
    if is_normalized:
        consumed_joules = consumed_joules/max_consumed_joules
    return alpha * mean_slowdown + beta * consumed_joules

def compute_score(sim_mon, sim_handler, alpha=0.5, beta=0.5, is_normalized=True):
  platform = sim_handler.platform
  hosts = platform.hosts
  total_max_watt_per_min = 0
  for host in hosts:
    max_watt_per_min = 0
    for pstate in host.pstates:
      max_watt_per_min = max(max_watt_per_min, pstate.watt_full)
    total_max_watt_per_min += max_watt_per_min

  total_time = sim_handler.current_time
  max_consumed_joules = total_time*total_max_watt_per_min
  consumed_joules = sim_mon.info["consumed_joules"]
  mean_slowdown = sim_mon.info["mean_slowdown"]
  score = F(mean_slowdown, consumed_joules, max_consumed_joules, alpha, beta, is_normalized)
  return score

if __name__ == "__main__":
    
    parser = define_args_parser()
    args = parser.parse_args(sys.argv[1:])
    to_policy = lambda s: TimeoutPolicy(args.timeout, s)
    start = time()
    sim_mon, host_mon, e_mon, simulator = run_simulation(to_policy, args.workload_path, args.platform_path, args.is_baseline) # Simulation 1: Timeout (30 minute) Dataset (Gaia)
    end = time()
    walltime = end-start
    header = ['dataset', 'timeout', 'f(1,0)=slowdown', 'f(0,1)=energy', 'f(0.5,0.5)=balance', "experiment walltime"]

    data = []
    row = []
    row.append(args.workload_path)
    row.append(args.timeout)
    row.append(compute_score(sim_mon, simulator, 1, 0, is_normalized=False))
    row.append(compute_score(sim_mon, simulator, 0, 1, is_normalized=False))
    row.append(compute_score(sim_mon, simulator, 0.5, 0.5, is_normalized=False))
    row.append(walltime)
    data.append(row)

    result_dir = pathlib.Path('.')/"results"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir/(args.title+".csv")
    with open(result_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)