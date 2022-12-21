from copy import deepcopy
import torch as T
import numpy as np
import pathlib

from env.sds_env import SDS_ENV
from config import get_args
from utils import select, get_success_jobs_info, compute_objective, run_partly_with_baseline, ResultInfo
from setup import setup_test

NUM_DATASETS = 1000

def save_checkpoint(agent_state_dict, 
                    agent_opt_state_dict, 
                    critic_state_dict,
                    critic_opt_state_dict, 
                    epoch,
                    step,
                    checkpoint_path:pathlib.Path):
    checkpoint = {
                    "agent_state_dict": agent_state_dict,
                    "agent_opt_state_dict": agent_opt_state_dict,
                    "critic_state_dict": critic_state_dict,   
                    "critic_opt_state_dict":critic_opt_state_dict,
                    "epoch":epoch,
                    "step":step
                }
    T.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    args = get_args()
    agent, epoch, checkpoint_path, writer = setup_test(args)

    #run with baseline first
    args.num_envs = 1
    # mulai generate experience dari training environments
    env = SDS_ENV(dataset_name=args.dataset_name, batsim_verbosity="information", is_test=True, alpha=args.alpha, beta=args.beta)
    env.reset()
    run_partly_with_baseline(env, completed_percentage_target=0.8)
    result_prerun = ResultInfo(
        env.simulation_monitor.info["total_slowdown"],
        env.simulation_monitor.info["total_waiting_time"],
        env.simulation_monitor.info["nb_jobs_finished"],
        env.simulator.current_time,
        env.simulation_monitor.info["consumed_joules"],
        env.simulation_monitor.info["time_idle"],
        env.simulation_monitor.info["time_computing"],
        env.simulation_monitor.info["time_switching_off"],
        env.simulation_monitor.info["time_switching_on"],
        env.simulation_monitor.info["time_sleeping"],
        env.simulation_monitor.info["energy_waste"]
    )
    last_waste_energy = env.simulation_monitor.info["energy_waste"]

    # start testing
    # 1 epoch = 1 full training data,, not the epoch commonly understood (?)
    # init training environment
    agent.eval()
    env.host_monitor.update_info_all()
    env.last_host_info = deepcopy(env.host_monitor.host_info)
    done = not env.is_really_running
    features = env.get_features(env.simulator.current_time)
    features = np.concatenate(features)
    features = features.reshape(args.num_envs, -1, 11)
    mask = env.get_mask()
    mask = np.asanyarray(mask)
    mask = mask.reshape(args.num_envs, -1, 2)      
    while not done:
        with T.no_grad():
            features_ = T.from_numpy(features).to(agent.device).float()
            mask_ = T.from_numpy(mask).to(agent.device).float()
            # print(mask_)
            if not T.any(mask_):
                env.simulator.proceed_time(time=1800)
                env.host_monitor.update_info_all()
                env.last_host_info = deepcopy(env.host_monitor.host_info)
                done = not env.is_really_running
                features = env.get_features(env.simulator.current_time)
                features = np.concatenate(features)
                features = features.reshape(args.num_envs, -1, 11)
                mask = env.get_mask()
                mask = np.asanyarray(mask)
                mask = mask.reshape(args.num_envs, -1, 2)
                continue
            # print(mask_)
            probs, entropy = agent(features_, mask_)
            need_decision_idx = T.any(mask_, dim=2).nonzero()[:,1]
            probs = probs[:, need_decision_idx, :]
            print(probs)
            # probs_new = T.zeros_like(probs)
            # probs_new[:,:,1] = 1
            actions, logprobs = select(probs, is_training=False)
            new_features, rewards, done, info = env.step(need_decision_idx, actions)
            if not done:
                features = new_features
                features = np.concatenate(features)
                features = features.reshape(args.num_envs, -1, 11)
                new_mask, wasted_energy, waiting_time_since_last_dt = info
                mask = new_mask
                mask = np.asanyarray(mask)
                mask = mask.reshape(args.num_envs, -1, 2)
            next_state = features

            env.last_host_info = deepcopy(env.host_monitor.host_info)

    result_current = ResultInfo(
        env.simulation_monitor.info["total_slowdown"],
        env.simulation_monitor.info["total_waiting_time"],
        env.simulation_monitor.info["nb_jobs_finished"],
        env.simulator.current_time,
        env.simulation_monitor.info["consumed_joules"],
        env.simulation_monitor.info["time_idle"],
        env.simulation_monitor.info["time_computing"],
        env.simulation_monitor.info["time_switching_off"],
        env.simulation_monitor.info["time_switching_on"],
        env.simulation_monitor.info["time_sleeping"],
        env.simulation_monitor.info["energy_waste"]
    )
    current_waste_energy = env.simulation_monitor.info["energy_waste"]
    env.simulator.close()
    alpha=0.5
    beta=0.5
    consumed_joules, mean_slowdown, score, time_idle, time_computing, time_switching_off, time_switching_on, time_sleeping, energy_waste, mean_waiting_time = compute_objective(env.simulator, result_current, result_prerun, alpha, beta)
    print("OBJECTIVE:", score)
    print("CONSUMED JOULES:", consumed_joules)
    print("MEAN SLOWDOWN:", mean_slowdown)
    print("TIME IDLE:", time_idle)
    print("TIME COMPUTING:", time_computing)
    print("TIME SWITCHING OFF:", time_switching_off)
    print("TIME SWITCHING ON:", time_switching_on)
    print("TIME SLEEPING:", time_sleeping)
    print("WASTE ENERGY:", current_waste_energy-last_waste_energy)
    print("AVERAGE JOBS WAITING TIME:", mean_waiting_time)
    print("ELAPSED TIME:", env.simulation_monitor.info["simulation_time"])
    print("Execution time:", env.simulator.current_time)

    s_job_info_list = get_success_jobs_info(env.job_monitor)
    filename = "Test_job_info_"+args.title+".csv"
    with open(filename, 'a+') as f:
        header="job_id,submission_time,num_nodes,requested_time,starting_time,execution_time,finish_time,waiting_time,turnaround_time,stretch\n"
        f.write(header)
        for s_job_info in s_job_info_list:
            row = str(s_job_info.job_id) + "," + str(s_job_info.submission_time) + "," + str(s_job_info.num_nodes) + "," + str(s_job_info.requested_time) + "," + str(s_job_info.starting_time) + "," + str(s_job_info.execution_time) + "," + str(s_job_info.finish_time) + "," + str(s_job_info.waiting_time) + "," + str(s_job_info.turnaround_time) + "," + str(s_job_info.stretch) + "\n"
            f.write(row)