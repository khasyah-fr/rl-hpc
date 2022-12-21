import os
import pathlib

import gym
import torch as T
from torch.utils.tensorboard import SummaryWriter

from agent.agent import Agent
from agent.critic import Critic
from agent.memory import PPOMemory
from env.sds_env import SDS_ENV

def setup(args):
    agent = Agent(n_heads=args.n_heads,
                 n_gae_layers=args.n_gae_layers,
                 input_dim=11,
                 embed_dim=args.embed_dim,
                 gae_ff_hidden=args.gae_ff_hidden,
                 tanh_clip=args.tanh_clip,
                 device=args.device)    
    agent_opt = T.optim.Adam(agent.parameters(), lr=args.lr)
    critic = Critic(n_heads=args.n_heads,
                 n_gae_layers=args.n_gae_layers,
                 input_dim=11,
                 embed_dim=args.embed_dim,
                 gae_ff_hidden=args.gae_ff_hidden,
                 device=args.device)
    critic_opt = T.optim.Adam(critic.parameters(), lr=args.lr)

    memory = None
    # load checkpoint if exists
    checkpoint_root = pathlib.Path(".")/"checkpoints"
    checkpoint_dir = checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/"checkpoint.pt"

    last_epoch = 0
    last_step = 0
    if os.path.isfile(checkpoint_path.absolute()):
        print(checkpoint_path.absolute())
        checkpoint = T.load(checkpoint_path.absolute(), map_location=agent.device)
        agent_state_dict = checkpoint["agent_state_dict"]
        agent_opt_state_dict = checkpoint["agent_opt_state_dict"]
        critic_state_dict = checkpoint["critic_state_dict"]
        critic_opt_state_dict = checkpoint["critic_opt_state_dict"]
        last_epoch = checkpoint["epoch"]
        last_step = checkpoint["step"]
        agent.load_state_dict(agent_state_dict)
        agent_opt.load_state_dict(agent_opt_state_dict)
        critic.load_state_dict(critic_state_dict)
        critic_opt.load_state_dict(critic_opt_state_dict)
    else:
        print("CHECKPOINT NOT FOUND, new run?")
    memory = PPOMemory(args.mini_batch_size)
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    return agent, critic, agent_opt, critic_opt, memory, last_epoch, last_step, checkpoint_path, writer 

def setup_test(args):
    agent = Agent(n_heads=args.n_heads,
                 n_gae_layers=args.n_gae_layers,
                 input_dim=11,
                 embed_dim=args.embed_dim,
                 gae_ff_hidden=args.gae_ff_hidden,
                 tanh_clip=args.tanh_clip,
                 device=args.device)    
    # load checkpoint if exists
    checkpoint_root = pathlib.Path(".")/"checkpoints"
    checkpoint_dir = checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/"checkpoint.pt"
    if args.epoch_test > -1:
        epoch_checkpoint_path = str(checkpoint_path.absolute())+"_"+str(args.epoch_test)

    if os.path.isfile(checkpoint_path.absolute()):
        print(checkpoint_path.absolute())
        checkpoint = T.load(checkpoint_path.absolute(), map_location=agent.device)
        if args.epoch_test > -1:
            epoch_checkpoint_path = str(checkpoint_path.absolute())+"_"+str(args.epoch_test)
            checkpoint = T.load(epoch_checkpoint_path, map_location=agent.device)
        agent_state_dict = checkpoint["agent_state_dict"]
        agent.load_state_dict(agent_state_dict)
    else:
        print("CHECKPOINT NOT FOUND, new run?")
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    return agent, args.epoch_test, checkpoint_path, writer 