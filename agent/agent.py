import math
from typing import Tuple
import torch as T
import torch.nn as nn

from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = T.device("cpu")

class Agent(T.jit.ScriptModule):
# class Agent(nn.Module):
    def __init__(self,
            n_heads: int = 8,
            n_gae_layers: int = 3,
            input_dim: int = 11,
            embed_dim: int = 128,
            gae_ff_hidden: int = 512,
            tanh_clip: float = 10,
            device=CPU_DEVICE):
        super(Agent, self).__init__()
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads
        # embedder
        self.gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=input_dim,
                                         feed_forward_hidden=gae_ff_hidden)
        self.prob_head = nn.Sequential(nn.Linear(embed_dim, 64),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(64,32),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(32,2))
        self.to(device)

    @T.jit.script_method
    def forward(self, features:T.Tensor, mask:T.Tensor)->Tuple[T.Tensor, T.Tensor]:
        batch_size, num_hosts, _ = features.shape
        embeddings, mean_embeddings = self.gae(features)
        
        logits = self.prob_head(embeddings)
        # logits = logits + mask.log()
            
        probs = T.softmax(logits, dim=2)
        entropy = -T.sum(probs*probs.log())
        return probs, entropy


# Documentation:
# Fitur2 objek itu diubah representasi ke sebuah space baru dan jarak antar representasi objek di space baru ini bisa dianggap jarak/perbedaan/relasi antar di objek di dunia nyata.
# Dengan membawa ke space baru, itu bisa menghasilkan fitur2 yg lebih representatif. 
# Harapannya di space baru ini, objek yang beda akan menjadi jauh, sebaliknya objek yang mirip akan menjadi dekat.

# logits bisa dianggap predicted Q value? Q value return dari aksi
# unscaled, semakin besar nanti saat dinormalize probsnya akan besar juga
# milih logits terbesar -> itu aksinya
# agent nanti ditrain agar logitsnya mirip dengan reward asli yang didapatkan

    
# Offline vs Online 
# agent update dengan experience yang lama ataupun experience dari luar -> offline
# agent menghasilkan experience yang "fresh", lalu dipakai update -> online