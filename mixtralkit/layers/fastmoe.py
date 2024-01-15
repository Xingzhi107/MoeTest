from fmoe import FMoE
from fmoe.gates import GShardGate,NaiveGate
from fmoe.gates.faster_gate import FasterGate
from fmoe import functions
import torch
import torch.nn as nn

import torch.nn.functional as F
import os
from .utils import ModelArgs
from .attention import TorchAttention
from .transformer import TorchTransformerBlock, TorchTransformer
import pdb
# 普通线性层
class Expert(nn.Module):
    def __init__(
        self,
        d_model, d_hidden,
        rank = 0,
    ):
        super().__init__()

        self.w1 = nn.Linear(
            d_model, d_hidden, bias=False
        )
        self.w2 = nn.Linear(
            d_hidden, d_model, bias=False
        )
        self.w3 = nn.Linear(
            d_model, d_hidden, bias=False
        )

    def forward(self, x, fec=None):
        # device = x.device
        # x = x.to(self.w1.weight.device)
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        # print(out.shape)
        return out

class FastMoe(FMoE):
    def __init__(self,
                 num_expert=4,
                 d_model = 1024,
                 d_hidden=4096,
                 activation=torch.nn.SiLU(),
                 world_size =1,
                 top_k = 2,
                 # moe_group = 1,
        ):
        def one_expert(d_model):
            return Expert( d_model,d_hidden)
        expert = one_expert
        super().__init__(num_expert, d_model, world_size,
                         top_k=top_k,expert=expert)
        self.mark_parallel_comm()
    def forward(self, inp: torch.tensor):
        original_shape = inp.shape
        #print("original_shape:",original_shape) #[bsz,seq,d]
        inp = inp.reshape(-1, self.d_model) #[bsz*seq,d]


        # pdb.set_trace()
        output = super().forward(inp)

        return output.reshape(original_shape)

class MoETorchTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)

        self.attention = TorchAttention(args)
        assert args.moe["num_experts"] % args.num_gpus == 0, "num_experts must be divisible by num_gpus"
        # print(int(os.environ['WORLD_SIZE']))
        self.feed_forward = FastMoe (
                num_expert=args.num_gpus,
                 d_model = args.dim,
                 d_hidden=args.hidden_dim,
                 activation=torch.nn.SiLU(),
                 world_size =args.world_size,
                 top_k = args.moe["num_experts_per_tok"],
        )



class MoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoETorchTransformerBlock(layer_id, params))
