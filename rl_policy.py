import enum
from platform import platform
from batsim_py import SimulatorHandler
from batsim_py.events import HostEvent, SimulatorEvent, JobEvent
from batsim_py.resources import Host, HostState
from batsim_py.monitors import SimulationMonitor, HostMonitor, ConsumedEnergyMonitor, JobMonitor

import torch as T
"""
kita tiru shutdown policy
si simulator akan subcribe ke sebuah callback function setiap dtime
callback function ini isinya:
1. extract features dari monitor
2. RL decides to switch on/off
3. RL executes on or off
4. 
"""

class RLPolicy:
    def __init__(self, 
                agent, 
                dtime, 
                simulation_monitor:SimulationMonitor, 
                energy_monitor:ConsumedEnergyMonitor,
                host_monitor:HostMonitor,
                job_monitor: JobMonitor,
                simulator: SimulatorHandler,
                alpha=0.5,
                beta=0.5) -> None:
        self.simulator = simulator
        self.dtime = dtime
        self.hosts_idle = {}
        # Subscribe to some events.
        self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self.on_simulation_begins)
        self.simulator.subscribe(JobEvent.SUBMITTED, self.add_to_job_infos)
    
        # agent and related variables
        self.agent = agent
        self.previous_action = None
        self.previous_simulator_features = None
        self.previous_node_features = None
        self.alpha = alpha
        self.beta = beta
        # self.er = ExperienceReplay(max_size=500?)
        
        # monitors for extracting features
        self.simulation_monitor = simulation_monitor
        self.energy_monitor = energy_monitor
        self.host_monitor = host_monitor
        self.job_monitor = job_monitor
        self.job_infos = {}

        self.previous_wasted_energy = None

    def add_to_job_infos(self, job):
        self.job_infos[job.id] = job        

    def on_simulation_begins(self, s: SimulatorHandler) -> None:
        self.n_host = len(list(self.simulator.platform.hosts))
        self.setup_callback()

    def setup_callback(self) -> None:
        t_next_call = self.simulator.current_time + self.dtime
        self.simulator.set_callback(t_next_call, self.callback)

    def callback(self, current_time: float) -> None:
        self.host_monitor.update_info_all()
        simulator_features, node_features = self.get_features(current_time)

        # 0. get rewards for previous action
        reward = self.get_rewards(current_time)
        print(reward)
        # 1. cek state of host, boleh disleep atau tidak (get feasible mask) size = (n_host, 2)
        hosts = list(self.simulator.platform.hosts)
        feasible_mask = get_feasible_mask(hosts)
        has_possible_action = feasible_mask.sum(dim=1) > 0
        # print(self.simulator.platform.state)
        # print(feasible_mask)
        # 2. generate probability (agent(simulator_features, node_features))
        # probs = agent(simulator_features, node_features)
        probs = T.rand(size=feasible_mask.shape)*feasible_mask # dari 0->1 jika feasible, 0 jika enggak, ini dummy
        # print(probs)
        # 3. select feasible action
        actions, logprobs = self.select(probs[has_possible_action, :])
        # print(actions)
        # 4. set node to sleep or active
        node_with_possible_action = has_possible_action.nonzero()
        for idx, action in enumerate(actions):
            host_id = node_with_possible_action[idx]
            if action == 0:
                if hosts[host_id].state == HostState.IDLE:
                    self.simulator.switch_off([host_id])
            else:
                if hosts[host_id].state == HostState.SLEEPING:
                    self.simulator.switch_on([host_id])

        # 5. add ke ER
        # 6. update berkala si agent atau langsung update setiap n_step aksi diambil?


        self.setup_callback()
        # for host_id, t_idle_start in list(self.hosts_idle.items()):
        #     if  current_time - t_idle_start >= self.t_timeout:
        #         self.simulator.switch_off([host_id])

    def get_rewards(self, current_time):
        wasted_energy = self.host_monitor.info["energy_waste"] 
        wasted_energy_ = wasted_energy
        if self.previous_wasted_energy is not None:
            wasted_energy -= self.previous_wasted_energy
        wasted_energy /= (190.*self.n_host*self.dtime)
        self.previous_wasted_energy = wasted_energy_

        waiting_time_since_last_dt = 0.
        n_job_waitting = 0
        for job in self.job_infos.values():
            if job.start_time is not None and (job.start_time < current_time-self.dtime):
                continue
            n_job_waitting += 1
            last_dt = current_time-self.dtime
            waittime = 0
            if job.start_time is not None:
                waittime = job.start_time - last_dt
            else:
                waittime = self.dtime # atau current_time-last_dt
            waiting_time_since_last_dt += (waittime/self.dtime)
        waiting_time_since_last_dt /= max(n_job_waitting,1)
        reward = -self.alpha*wasted_energy -self.beta*waiting_time_since_last_dt
        return reward


        
        


    def select(self, probs, is_training=True):
        '''
        ### Select next to be executed.
        -----
        Parameter:
            probs: probabilities of each operation

        Return: index of operations, log of probabilities
        '''
        if is_training:
            dist = T.distributions.Categorical(probs)
            op = dist.sample()
            logprob = dist.log_prob(op)
        else:
            prob, op = T.max(probs, dim=1)
            logprob = T.log(prob)
        return op, logprob


    def get_features(self, current_time):
        # print2 feature2 dulu sementara
        # print("CURRENT TIME", current_time)
        # print("PLATFORM Features")
        # print("switch on dan switch off time langsung dikirim ke batsim, ga ada di batsimpy")
        # print("1. switch on: 1 second")
        # print("2. switch off: 10 seconds")
        
        # print("SIMULATOR FEATURES")
        # print("1. jumlah jobs di queue:", len(self.simulator.queue))
        # print("2. arrival rate:")
        # submission_time = self.job_monitor.info["submission_time"]
        # print("---overall arrival rate (not&normalized):",get_arrival_rate(submission_time, False), get_arrival_rate(submission_time, True))
        # print("---last 100 jobs arrival rate (not&normalized):",get_arrival_rate(submission_time[-100:], False), get_arrival_rate(submission_time[-100:], True) )
        # hosts = self.simulator.platform.hosts
        # print("3. mean runtime jobs di queue:", get_mean_runtime_nodes_in_queue(self.simulator, normalized=False), get_mean_runtime_nodes_in_queue(self.simulator, normalized=False))
        # exit()
        # queue = self.simulator.queue
        # print("3. current mean waiting time:", get_mean_waittime_queue(queue, current_time, False), get_mean_waittime_queue(queue, current_time, True))
        # print("4. Wasted energy (Joule):", get_wasted_energy(self.energy_monitor, self.host_monitor, False), get_wasted_energy(self.energy_monitor, self.host_monitor, True))
        # print("5. mean requested walltime jobs in queue:", get_mean_walltime_in_queue(queue, False), get_mean_walltime_in_queue(queue,True))
        num_sim_features = 5
        simulator_features = T.zeros((num_sim_features,), dtype=T.float32)
        simulator_features[0] = len(self.simulator.queue)
        submission_time = self.job_monitor.info["submission_time"]
        simulator_features[1] = get_arrival_rate(submission_time, is_normalized=True)
        hosts = self.simulator.platform.hosts
        queue = self.simulator.queue
        simulator_features[2] = get_mean_waittime_queue(queue, current_time, is_normalized=True)
        simulator_features[3] = get_wasted_energy(self.energy_monitor, self.host_monitor, is_normalized=True)
        simulator_features[4] = get_mean_walltime_in_queue(queue,is_normalized=True)
        simulator_features = simulator_features.unsqueeze(0)

        num_node_features = 6
        hosts = list(self.simulator.platform.hosts)
        node_features = T.zeros((len(hosts), num_node_features), dtype=T.float32)
        # print("NODE FEATURES")
        # print("1. ON/OFF")
        host_on_off = get_host_on_off(self.simulator.platform)
        node_features[:, 0] = host_on_off
        # print("2. ACTIVE/IDLE")
        host_active_idle = get_host_active_idle(self.simulator.platform)
        node_features[:, 1] = host_active_idle
        # print("3. Running Idle TIME")
        current_idle_time = get_current_idle_time(self.host_monitor)
        node_features[:, 2] = current_idle_time
        # print("4. remaining time (percent) of job in nodes")
        remaining_runtime_percent = get_remaining_runtime_percent(list(self.simulator.platform.hosts), self.job_infos, self.simulator.current_time)
        node_features[:, 3] = remaining_runtime_percent
        # print("5. wasted energy / consumed joules")
        wasted_energy, normalized_wasted_energy = get_host_wasted_energy(self.host_monitor, False), get_host_wasted_energy(self.host_monitor, True)
        node_features[:, 4] = normalized_wasted_energy
        # print("6. time switching state/ time computing? or total time perhaps?")
        switching_time, normalized_switching_time = get_switching_time(self.host_monitor, False), get_switching_time(self.host_monitor, True, self.simulator.current_time) 
        node_features[:, 5] = normalized_switching_time
        return simulator_features, node_features

def get_feasible_mask(hosts):
    #fm[:, 0] = boleh matikan/tidak
    #fm[:, 1] = boleh hidupkan/tidak
    feasible_mask = T.ones((len(hosts), 2), dtype=T.float32)
    for i, h in enumerate(hosts):
        # cannot be switched off
        if h.is_allocated or h.state == HostState.SWITCHING_ON or h.state == HostState.COMPUTING:
            feasible_mask[i][0] = 0
        # cannot be switched on
        if h.state == HostState.SWITCHING_OFF:
            feasible_mask[i][1] = 0

    return feasible_mask


def get_arrival_rate(submission_times, is_normalized=True):
    if len(submission_times) == 0:
        return 0
    if len(submission_times) == 1:
        return submission_times[0]
    submission_times = T.tensor(submission_times)
    submission_times -= submission_times[0].item()
    max_time = submission_times[-1]
    submission_times_r = T.roll(submission_times, 1)  
    submission_times -= submission_times_r
    arrival_rate = T.mean(submission_times[1:])
    if is_normalized:
        arrival_rate /= max_time
    return arrival_rate

def get_mean_walltime_in_queue(queue, is_normalized):
    walltime_in_queue = [job.walltime for job in queue]
    walltime_in_queue = T.tensor(walltime_in_queue, dtype=T.float32)
    if len(walltime_in_queue) == 0:
        return 0
    mean_walltime = T.mean(walltime_in_queue)
    if is_normalized:
        mean_walltime /= T.max(walltime_in_queue)
    return mean_walltime

def get_mean_waittime_queue(queue, current_time, is_normalized):
    subtimes = [job.subtime for job in queue]
    if len(subtimes)==0:
        return 0
    
    subtimes = T.tensor(subtimes, dtype=T.float32)
    wait_times = current_time-subtimes
    mean_wait_times = T.mean(wait_times)
    if is_normalized:
        mean_wait_times /= T.max(wait_times)
    return mean_wait_times

def get_wasted_energy(energy_mon, host_mon, is_normalized):
    wasted_energy = host_mon.info["energy_waste"]
    if is_normalized:
        all_energy = energy_mon.info["energy"]
        total_energy = T.sum(T.tensor(all_energy, dtype=T.float32))
        wasted_energy = wasted_energy/total_energy
    return wasted_energy

def get_host_on_off(platform):
    host_states = [[0 if state is HostState.SLEEPING else 1 for state in platform.state]]
    host_states = T.tensor(host_states)
    return host_states

def get_host_active_idle(platform):
    host_states = [[0 if (state is HostState.IDLE) else 1 for state in platform.state]]
    host_states = T.tensor(host_states)
    return host_states

def get_current_idle_time(host_monitor):
    host_info = host_monitor.host_info
    current_idle_time = [[0. for _ in range(len(host_info))]]
    for id in host_info.keys():
        current_idle_time[0][int(id)] = host_info[int(id)]["time_idle_current"]
    current_idle_time = T.tensor(current_idle_time)
    return current_idle_time

def get_host_wasted_energy(host_monitor, is_normalized=False):
    host_info = host_monitor.host_info
    wasted_energy = [[0. for _ in range(len(host_info))]]
    for id in host_info.keys():
        wasted_energy[0][int(id)] = host_info[int(id)]["energy_waste"]
        if is_normalized:
            wasted_energy[0][int(id)] /= host_info[int(id)]["consumed_joules"]
    wasted_energy = T.tensor(wasted_energy)
    return wasted_energy

def get_remaining_runtime_percent(hosts, job_infos, current_time):
    remaining_runtime = [0 for _ in range(len(hosts))]
    for i,h in enumerate(hosts):
        if len(h.jobs) > 0 and h.state == HostState.COMPUTING:
            job = job_infos[h.jobs[0]]
            remaining_runtime[i] = current_time-job.start_time
            remaining_runtime[i] /= job.walltime
    remaining_runtime = T.tensor([remaining_runtime])
    return remaining_runtime

def get_switching_time(host_monitor, is_normalized=False, current_time=1):
    host_info = host_monitor.host_info
    switching_time = [[0. for _ in range(len(host_info))]]
    for id in host_info.keys():
        h_info = host_info[int(id)]
        switching_time[0][int(id)] = h_info["time_switching_off"] + h_info["time_switching_on"]
        if is_normalized:
            switching_time[0][int(id)] /= current_time
    switching_time = T.tensor(switching_time)
    return switching_time    

# we are here now
# def get_mean_runtime_nodes_in_queue(hosts, normalized):
#     runtimes = []
#     for h in hosts:
#         if h.is_computing:


        
