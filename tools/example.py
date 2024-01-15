# Copyright (c) OpenMMLab. and affiliates.

import argparse
from mixtralkit.mixtral import Mixtral
import torch.distributed as dist
import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity
def parse_args():
    parser = argparse.ArgumentParser(description='Run an inference of mixtral-8x7b model')
    parser.add_argument('-m',
                        '--model-weights',
                        help='Model weights.',
                        default=None,
                        type=str)
    parser.add_argument('-t',
                        '--tokenizer',
                        help='path of tokenizer file.',
                        default=None,
                        type=str)
    # parser.add_argument("--local_rank", default=-1)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    world_size = int(os.environ["WORLD_SIZE"])

    rank = int(os.environ["RANK"])

    nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    nnodes = world_size//nproc_per_node
    dist.init_process_group(backend="nccl",world_size=world_size,rank=rank)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    max_batch_size = 32
    max_seq_len = 1024
    max_gen_len = 64
    results = [None]

    prompts = [
        "Who are you?",
        "1 + 1 -> 3\n"
        "2 + 2 -> 5\n"
        "3 + 3 -> 7\n"
        "4 + 4 -> ",
        "请问你是什么模型？",
        "format(dist.get_rank())model_inference?format(dist.get_rank())model_inference?",
    ]

    temperature = 1.0 # for greedy decoding
    top_p = 0.9
    num_gpus = 8
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True) as prof:
        with record_function("model_inference"):
            generator = Mixtral.build(
                ckpt_dir=args.model_weights,
                tokenizer_path=args.tokenizer,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                num_gpus=1,
                world_size=world_size,
            )
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
    print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=20))
    prof.export_stacks("torch_cuda_stack.json", metric="self_cuda_time_total")
    for prompt, result in zip(prompts, results):
        print("="*30 + "Example START" + "="*30 + '\n')
        print("[Prompt]:\n{}\n".format(prompt))
        print("[Response]:\n{}\n".format(result['generation']))
        print("="*30 + "Example END" + "="*30 + '\n')

    # prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    main()
