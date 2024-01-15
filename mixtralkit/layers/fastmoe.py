from fmoe import FMoe
import torch
import torch.nn as nn
from fmoe import FMoELinear
import torch.nn.functional as F

from .utils import ModelArgs
from .attention import TorchAttention
from .transformer import TorchTransformerBlock, TorchTransformer
class Expert(nn.Module):
    def __init__(
        self,
        num_expert: int,
        d_model, d_hidden, activation,
        rank = 0,
    ):
        super().__init__()

        self.w1 = FMoELinear(
            num_expert, d_model, d_hidden, bias=True, rank=rank
        )
        self.w2 = nn.Linear(
            num_expert, d_hidden, d_model, bias=True, rank=rank
        )
        self.w3 = nn.Linear(
            num_expert, d_model, d_hidden, bias=True, rank=rank
        )
        self.activation = activation

    def forward(self, x):
        device = x.device
        x = x.to(self.w1.weight.device)
        return self.w2(F.silu(self.w1(x)) * self.w3(x)).to(device)

class FastMoe(FMoe):
    def __init__(self,
                 num_expert=8,
                 d_model = 1024,
                 d_hidden=4096,
                 activation=torch.nn.GELU(),
                 world_size =1,
                 top_k = 2,
        ):
        expert = Expert(1,d_model,d_hidden,activation,rank=0)
        super().__init__(num_expert, d_model, world_size,
                         top_k=top_k,expert=expert)
        self.mark_parallel_comm()
    
    def forward(self, inp: torch.tensor):
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
    
class MoETorchTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        
        self.attention = TorchAttention(args)
        assert args.moe["num_experts"] % args.num_gpus == 0, "num_experts must be divisible by num_gpus"
        self.feed_forward = FastMoe (
             num_expert=args.moe["num_experts"],
                 d_model = args.smoe["dim"],
                 d_hidden=args.smoe["hidden_dim"],
                 activation=torch.nn.GELU(),
                 world_size =1,
                 top_k = args.moe["num_experts_per_tok"],
                 gate = None,
                 expert = None
        )
        # self.feed_forward = MoETorchFFN(
        #     dim=args.dim,
        #     hidden_dim=args.hidden_dim,
        #     num_shards=args.moe["num_experts"] // args.num_gpus,
        #     **args.moe,
        # )


class MoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoETorchTransformerBlock(layer_id, params))