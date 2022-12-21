from copy import deepcopy
import torch as T
import numpy as np
import pathlib

from env.sds_env import SDS_ENV
from config import get_args
from utils import select, learn, ResultInfo, compute_objective
from setup import setup

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
    T.save(checkpoint, checkpoint_path.absolute())
    epoch_checkpoint_path = str(checkpoint_path.absolute())+"_"+str(epoch)
    T.save(checkpoint, epoch_checkpoint_path)


class AgentRunner():
    def __init__(self,
                 agent, 
                 critic,
                 agent_opt,
                 writer,
                 dt=1800) -> None:
        self.agent = agent
        self.critic = critic
        self.agent_opt = agent_opt
        self.writer = writer
        self.mask = np.ones((args.num_envs, 128, 2))
        self.mask[:,:,1] = 0
        self.features = env.reset() 
        self.features = features.reshape(args.num_envs, -1, 11)
        self.done = False
        self.saved_logprobs = []
        self.saved_rewards = []
        self.saved_states = []
        self.next_state = None

    def new_epoch(self):
        self.mask = np.ones((args.num_envs, 128, 2))
        self.mask[:,:,1] = 0
        self.features = env.reset() 
        self.features = features.reshape(args.num_envs, -1, 11)
        self.done = False
        self.saved_logprobs = []
        self.saved_rewards = []
        self.saved_states = []
        self.next_state = None

    def act(self, step, epoch, env:SDS_ENV, current_time):
        if not env.is_really_running:
            return
        # set callback for next step
        next_step_at = current_time + self.dt
        next_act_func = lambda ct, step_=step, epoch_=epoch, env_=env: self.act(step_,epoch_,env_,ct) 
        env.simulator.set_callback(next_step_at, next_act_func)

        #do stuffss
        features_ = T.from_numpy(self.features).to(self.agent.device).float()
        mask_ = T.from_numpy(self.mask).to(self.agent.device).float()
        if not T.any(mask_):
            env.host_monitor.update_info_all()
            env.last_host_info = deepcopy(env.host_monitor.host_info)
            self.done = not env.simulator.is_running
            features = env.get_features(env.simulator.current_time)
            features = np.concatenate(features)
            self.features = features.reshape(args.num_envs, -1, 11)
            mask = env.get_mask()
            mask = np.asanyarray(mask)
            self.mask = mask.reshape(args.num_envs, -1, 2)
            return

        # print(mask_)
        probs, entropy = agent(features_, mask_)
        need_decision_idx = T.any(mask_, dim=2).nonzero()[:,1]
        probs = probs[:, need_decision_idx, :]
        actions, logprobs = select(probs)
        new_features, rewards, done, info = env.step(need_decision_idx, actions)
        
        # save the experiences
        self.saved_logprobs += [logprobs.sum()]
        self.saved_rewards += [rewards]
        self.saved_states += [features_]
        
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


if __name__ == "__main__":
    args = get_args()
    agent, critic, agent_opt, critic_opt, memory, last_epoch, last_step, checkpoint_path, writer = setup(args)
    
    # start training
    # 1 epoch = 1 full training data,, not the epoch commonly understood (?)
    # init training environment
    step = last_step
    args.num_envs = 1
    for epoch in range(last_epoch, args.max_epoch):
        # mulai generate experience dari training environments
        env = SDS_ENV(dataset_name=args.dataset_name, batsim_verbosity="quiet", is_test=False, alpha=args.alpha, beta=args.beta)
        env.last_host_info = deepcopy(env.host_monitor.host_info)
        
        while env.simulator.is_running:
            env.simulator.set_callback(at=env.simulator.current_time+args.dt, step_agent)
            env.simulator.proceed_time()
            env.scheduler.schedule()
        
        while not done:
            
            

            #log important values
            writer.add_scalar("Entropy", entropy.sum().item(), step)
            writer.add_scalar("Reward", rewards, step)
            writer.add_scalar("Wasted Energy Reward", wasted_energy, step)
            writer.add_scalar("Waitim Time Reward", waiting_time_since_last_dt, step)
            writer.add_scalar("Consume Joules", env.host_monitor.info["consumed_joules"], step)
            writer.add_scalar("Wasted Energy", env.host_monitor.info["energy_waste"], step)
            writer.add_scalar("Time Computing", env.host_monitor.info["time_computing"], step)
            writer.add_scalar("Time Idle", env.host_monitor.info["time_idle"], step)
            writer.add_scalar("Time Switching On", env.host_monitor.info["time_switching_on"], step)
            writer.add_scalar("Time Switching Off", env.host_monitor.info["time_switching_off"], step)
            writer.add_scalar("Time Sleeping", env.host_monitor.info["time_sleeping"], step)
            writer.add_scalar("Number of Switching State", env.host_monitor.info["nb_switches"], step)
                
            if step > 0 and len(saved_logprobs) >= args.training_steps:
                saved_experiences = (saved_logprobs, saved_states, saved_rewards, next_state)
                learn(args, agent, agent_opt, critic, critic_opt, done, saved_experiences)
                #clean experiences
                saved_logprobs = []
                saved_rewards = []
                saved_states = []
                next_state = None
                save_checkpoint(agent.state_dict(), agent_opt.state_dict(), critic.state_dict(), critic_opt.state_dict(), epoch, step, checkpoint_path)
            step+=1
            print(step)

        # if done, log the objective
        # compute objective
        result = ResultInfo(
            env.simulation_monitor.info["total_slowdown"],
            env.simulation_monitor.info["nb_jobs_finished"],
            env.simulator.current_time,
            env.simulation_monitor.info["consumed_joules"],
            env.simulation_monitor.info["time_idle"],
            env.simulation_monitor.info["time_computing"],
            env.simulation_monitor.info["time_switching_off"],
            env.simulation_monitor.info["time_switching_on"],
            env.simulation_monitor.info["time_sleeping"]
        )
        consumed_joules, mean_slowdown, score, time_idle, time_computing, time_switching_off, time_switching_on, time_sleeping, energy_waste = compute_objective(env.simulator, result, None, args.alpha, args.beta)
    
        writer.add_scalar("Consumed Joules Epoch", consumed_joules, epoch)
        writer.add_scalar("Mean Slowdown Epoch", mean_slowdown, epoch)
        writer.add_scalar("Score Epoch", score, epoch)
        writer.add_scalar("Time Idle Epoch", time_idle, epoch)
        writer.add_scalar("Time Computing Epoch", time_computing, epoch)
        writer.add_scalar("Time Switching Off Epoch", time_switching_off, epoch)
        writer.add_scalar("Time Switching On Epoch", time_switching_on, epoch)
        writer.add_scalar("Time Sleeping Epoch", time_switching_off, epoch)
        
        