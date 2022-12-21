import csv
import queue
from random import randint
import sys
import pathlib
from this import s
from time import time
from typing import Tuple

import gym
from gym import Env, spaces
import numpy as np

from batsim_py import SimulatorHandler
from batsim_py import SimulatorHandler
from batsim_py.monitors import HostStateSwitchMonitor, SimulationMonitor, HostMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_py.events import HostEvent, JobEvent

from env.easy_backfilling import EASYScheduler
from env.utils import *

# NO_OP = 0
# SWITCH_OFF = 1
# SWITCH_ON = 2
SWITCH_OFF = 0
SWITCH_ON = 1

class SDS_ENV(Env):
    def __init__(self, 
                 dataset_name, 
                 batsim_verbosity="quiet", 
                 alpha=0.5, 
                 beta=0.5, 
                 num_host=128, 
                 is_test=False) -> None:
        super(SDS_ENV, self).__init__()
        self.batsim_verbosity = batsim_verbosity
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.platform_path = pathlib.Path(".")/"platform"/("platform-"+str(num_host)+".xml")
        self.dataset_dir = pathlib.Path(".")/"dataset"
        if self.is_test:
            self.dataset_dir= self.dataset_dir/"test"
        else:
            self.dataset_dir= self.dataset_dir/"training"
        self.dataset_filepath = self.dataset_dir/self.dataset_name
        self.alpha = alpha
        self.beta = beta
        self.simulator = SimulatorHandler()
        self.scheduler = EASYScheduler(self.simulator)
        self.num_sim_features = 5
        self.num_node_features = 6
        self.n_host = num_host
        self.observation_shape = (self.n_host, self.num_sim_features+self.num_node_features)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float32)
        # Define an action space for each host, ranging from 0 to 3
        # 0. No Action
        # 1. Turn OFF
        # 2. Turn ON
        self.action_space = spaces.MultiDiscrete([3 for _ in range(self.n_host)])


    def add_to_job_infos(self, job):
        self.job_infos[job.id] = job  
    
    def close(self):
        self.simulator.close()

    def reset(self):
        self.simulator.close()
        self.last_host_info = None
        # 1) Instantiate monitors to collect simulation statistics
        self.simulation_monitor = SimulationMonitor(self.simulator)
        self.host_monitor = HostMonitor(self.simulator)
        self.host_state_switch_monitor = HostStateSwitchMonitor(self.simulator)
        self.energy_monitor = ConsumedEnergyMonitor(self.simulator)
        self.job_monitor = JobMonitor(self.simulator)

        # job infos dict, manually compile 
        self.job_infos = {}
        self.simulator.subscribe(JobEvent.SUBMITTED, self.add_to_job_infos)

        # trigger for scheduler
        self.simulator.subscribe(JobEvent.COMPLETED, self.scheduler.schedule)
        self.simulator.subscribe(JobEvent.SUBMITTED, self.scheduler.schedule)
        self.simulator.subscribe(HostEvent.ON, self.scheduler.schedule)
        
        # self.simulator.subscribe(HostEvent.STATE_CHANGED, self.scheduler.schedule)
        # self.simulator.subscribe(HostEvent.COMPUTATION_POWER_STATE_CHANGED, self.scheduler.schedule)
                
        self.previous_wasted_energy = None

        self.simulator.start(platform=self.platform_path, workload=self.dataset_filepath.absolute(), verbosity=self.batsim_verbosity)
        self.hosts = list(self.simulator.platform.hosts)
        self.host_monitor.update_info_all()
        
        features = self.get_features(self.simulator.current_time)
        mask = get_feasible_mask(list(self.simulator.platform.hosts))

        return features

    def get_mask(self):
        return get_feasible_mask(list(self.simulator.platform.hosts))

    def step(self, node_idx_list, actions):
        self.apply(node_idx_list, actions)
        dt = 1800
        # proceed time, and schedule
        # get next features
        current_time = self.simulator.current_time
        rewards = 0
        # rewards, _, _ = self.get_rewards(node_idx_list, current_time, dt) 
        
        if not self.simulator.is_running:
            done=True
            # self.simulator.proceed_time()
            self.host_monitor.update_info_all()
            self.simulator.close()
            return None, rewards, done, None

        # self.scheduler.schedule()
        self.simulator.proceed_time(time=dt)  # proceed directly to the next event.
        done = False
        self.host_monitor.update_info_all()
        # print("A---------------------A")
        # print(is_really_running)
        # print(self.simulator.jobs, len(self.simulator.jobs))
        # print(self.simulator.is_running, self.simulator.is_submitter_finished)
        # print(len(self.simulator.queue))
        # print(self.host_state_switch_monitor.info["nb_computing"])
        current_time = self.simulator.current_time
        features = self.get_features(current_time)
        mask = get_feasible_mask(list(self.simulator.platform.hosts))
        rewards, wasted_energy, wasting_time_since_last_dt = self.get_rewards(node_idx_list, current_time, dt) 
        return features, rewards, done, (mask, wasted_energy, wasting_time_since_last_dt)

    @property
    def is_really_running(self):
        # return self.simulator.is_running
        is_really_running_ = not self.simulator.is_submitter_finished
        is_really_running_ = is_really_running_ or len(self.simulator.jobs) > 0
        is_really_running_ = is_really_running_ or self.host_state_switch_monitor.info["nb_computing"][-1] > 0
        return is_really_running_

    def get_mask(self):
        return get_feasible_mask(list(self.simulator.platform.hosts))

    def apply(self, node_idx_list, actions):
        feasible_mask = get_feasible_mask(list(self.simulator.platform.hosts))
        actions = actions.squeeze(0)
        host_id_to_switch_on = []
        host_id_to_switch_off = []
        for i in range(len(node_idx_list)):
            node_idx = node_idx_list[i]
            action = actions[i]
            mask = feasible_mask[node_idx].astype(bool)
            if mask[action]:
                if action == SWITCH_OFF:
                    host_id_to_switch_off += [node_idx.item()]
                else:
                    host_id_to_switch_on += [node_idx.item()]
        if len(host_id_to_switch_off)>0:
            self.simulator.switch_off(host_id_to_switch_off)
        if len(host_id_to_switch_on)>0:
            self.simulator.switch_on(host_id_to_switch_on)

    def get_rewards(self, node_idx_list, current_time, dt):
        wasted_energy = 0
        host_info = self.host_monitor.host_info
        last_host_info = self.last_host_info
        for i in range(len(node_idx_list)):
            node_idx = node_idx_list[i].item()
            host_wasted_energy = host_info[node_idx]["energy_waste"]-last_host_info[node_idx]["energy_waste"]
            host_wasted_energy /= (190.*dt)
            wasted_energy += host_wasted_energy
        wasted_energy /= max(len(node_idx_list),1)
        if not self.is_really_running:
            return -self.alpha*wasted_energy, wasted_energy, 0

        waiting_time_since_last_dt = 0.
        n_job_waitting = 0
        for job in self.job_infos.values():
            if job.start_time is not None and (job.start_time < current_time-dt):
                continue
            n_job_waitting += 1
            last_dt = current_time-dt
            waittime = 0
            if job.start_time is not None:
                waittime = job.start_time - last_dt
            else:
                waittime = dt
            waiting_time_since_last_dt += (waittime/dt)
        waiting_time_since_last_dt /= max(n_job_waitting,1)
        reward = -self.alpha*wasted_energy -self.beta*waiting_time_since_last_dt
        # print(reward, wasted_energy, waiting_time_since_last_dt)
        return reward, wasted_energy, waiting_time_since_last_dt

    def get_features(self, current_time)-> Tuple[np.array, np.array]:
        simulator_features = np.zeros((self.num_sim_features,), dtype=np.float32)
        simulator_features[0] = len(self.simulator.queue)
        submission_time = self.job_monitor.info["submission_time"]
        simulator_features[1] = get_arrival_rate(submission_time, is_normalized=True)
        hosts = self.simulator.platform.hosts
        queue = self.simulator.queue
        simulator_features[2] = get_mean_waittime_queue(queue, current_time, is_normalized=True)
        simulator_features[3] = get_wasted_energy(self.energy_monitor, self.host_monitor, is_normalized=True)
        simulator_features[4] = get_mean_walltime_in_queue(queue,is_normalized=True)
        simulator_features = simulator_features[np.newaxis, ...]

        hosts = list(self.simulator.platform.hosts)
        node_features = np.zeros((len(hosts), self.num_node_features), dtype=np.float32)
        host_on_off = get_host_on_off(self.simulator.platform)
        node_features[:, 0] = host_on_off
        host_active_idle = get_host_active_idle(self.simulator.platform)
        node_features[:, 1] = host_active_idle
        current_idle_time = get_current_idle_time(self.host_monitor)
        node_features[:, 2] = current_idle_time

        remaining_runtime_percent = get_remaining_runtime_percent(list(self.simulator.platform.hosts), self.job_infos, self.simulator.current_time)
        node_features[:, 3] = remaining_runtime_percent
        normalized_wasted_energy = get_host_wasted_energy(self.host_monitor, True)
        node_features[:, 4] = normalized_wasted_energy
        normalized_switching_time = get_switching_time(self.host_monitor, True, self.simulator.current_time) 
        node_features[:, 5] = normalized_switching_time
        simulator_features = np.broadcast_to(simulator_features, (node_features.shape[0], simulator_features.shape[1]))
        features = np.concatenate((simulator_features, node_features), axis=1)

        return features
