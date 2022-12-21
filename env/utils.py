from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor
from batsim_py.resources import Host, HostState
import numpy as np


def get_feasible_mask(hosts: 'list[Host]'):
    #fm[:, 0] = dibiarkan, dummy
    #fm[:, 1] = boleh matikan/tidak
    #fm[:, 2] = boleh hidupkan/tidak
    feasible_mask = np.ones((len(hosts), 3), dtype=np.float32)
    is_switching_off = np.asarray([host.is_switching_off for host in hosts])
    is_switching_on = np.asarray([host.is_switching_on for host in hosts])
    is_switching = np.logical_or(is_switching_off, is_switching_on)
    is_idle = np.asarray([host.is_idle for host in hosts])
    is_sleeping = np.asarray([host.is_sleeping for host in hosts])
    is_allocated = np.asarray([host.is_allocated for host in hosts])

    # can it be switched off
    is_really_idle = np.logical_and(is_idle, np.logical_not(is_allocated))
    feasible_mask[:, 1] = np.logical_and(np.logical_not(is_switching), is_really_idle)

    # can it be switched on
    feasible_mask[:, 2] = np.logical_and(np.logical_not(is_switching), is_sleeping)
    # return cuma 2 action, update 15-09-2022
    return feasible_mask[:, 1:]


def get_arrival_rate(submission_times, is_normalized=True):
    if len(submission_times) == 0:
        return 0
    if len(submission_times) == 1:
        return submission_times[0]
    submission_times = np.asarray(submission_times)
    submission_times -= submission_times[0].item()
    max_time = max(submission_times[-1],1e-8)
    submission_times_r = np.roll(submission_times, 1)  
    submission_times -= submission_times_r
    arrival_rate = np.mean(submission_times[1:])
    if is_normalized:
        arrival_rate /= max_time
    return arrival_rate

def get_mean_walltime_in_queue(queue, is_normalized):
    walltime_in_queue = [job.walltime for job in queue]
    walltime_in_queue = np.asarray(walltime_in_queue, dtype=np.float32)
    if len(walltime_in_queue) == 0:
        return 0
    mean_walltime = np.mean(walltime_in_queue)
    if is_normalized:
        mean_walltime /= np.max(walltime_in_queue)
    return mean_walltime

def get_mean_waittime_queue(queue, current_time, is_normalized):
    subtimes = [job.subtime for job in queue]
    if len(subtimes)==0:
        return 0
    
    subtimes = np.asarray(subtimes, dtype=np.float32)
    wait_times = current_time-subtimes
    mean_wait_times = np.mean(wait_times)
    if is_normalized:
        mean_wait_times /= np.max(wait_times)
    return mean_wait_times

def get_wasted_energy(energy_mon, host_mon, is_normalized):
    wasted_energy = host_mon.info["energy_waste"]
    all_energy = host_mon.info["consumed_joules"]

    if is_normalized and all_energy>0:
        wasted_energy = wasted_energy/all_energy

    return wasted_energy

def get_host_on_off(platform):
    host_states = [[0 if state is HostState.SLEEPING else 1 for state in platform.state]]
    host_states = np.asarray(host_states)

    return host_states

def get_host_active_idle(platform):
    host_states = [[0 if (state is HostState.IDLE) else 1 for state in platform.state]]
    host_states = np.asarray(host_states)

    return host_states

def get_current_idle_time(host_monitor):
    host_info = host_monitor.host_info
    current_idle_time = [[0. for _ in range(len(host_info))]]
    for id in host_info.keys():
        current_idle_time[0][int(id)] = host_info[int(id)]["time_idle_current"]
    current_idle_time = np.asarray(current_idle_time)

    return current_idle_time

def get_host_wasted_energy(host_monitor, is_normalized=False):
    host_info = host_monitor.host_info
    wasted_energy = [[0. for _ in range(len(host_info))]]
    for id in host_info.keys():
        wasted_energy[0][int(id)] = host_info[int(id)]["energy_waste"]
        if is_normalized and host_info[int(id)]["consumed_joules"]:
            wasted_energy[0][int(id)] /= host_info[int(id)]["consumed_joules"]
    wasted_energy = np.asarray(wasted_energy)

    return wasted_energy

def get_remaining_runtime_percent(hosts, job_infos, current_time):
    remaining_runtime = [0 for _ in range(len(hosts))]
    for i,h in enumerate(hosts):
        if len(h.jobs) > 0 and h.state == HostState.COMPUTING:
            job = job_infos[h.jobs[0]]
            remaining_runtime[i] = current_time-job.start_time
            remaining_runtime[i] /= job.walltime
    remaining_runtime = np.asarray([remaining_runtime])

    return remaining_runtime

def get_switching_time(host_monitor, is_normalized=False, current_time=1):
    host_info = host_monitor.host_info
    switching_time = [[0. for _ in range(len(host_info))]]
    for id in host_info.keys():
        h_info = host_info[int(id)]
        switching_time[0][int(id)] = h_info["time_switching_off"] + h_info["time_switching_on"]
        if is_normalized and current_time:
            switching_time[0][int(id)] /= current_time
    switching_time = np.asarray(switching_time)

    return switching_time    


def compute_objective(sim_mon:SimulationMonitor, sim_handler:SimulatorHandler, alpha=0.5, beta=0.5, is_normalized=True):
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
  
  return consumed_joules, mean_slowdown, score
