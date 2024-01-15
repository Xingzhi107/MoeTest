# 1、编译fastmoe
https://github.com/laekov/fastmoe/blob/master/doc/installation-guide.md

## （1）环境：docker（跑不了太大和太多卡的）或者conda
```
docker pull nvcr.io/nvidia/pytorch:21.10-py3
docker run -dt --name xxx --restart=always --gpus all \
--network=host \
--shm-size 12G \
-v /xxx/:/xxx \
-w /xxx \
nvcr.io/nvidia/pytorch:21.10-py3 \
/bin/bash
docker exec -it xxx bash
```
```
conda create --name xxx python=3.8
conda activate xxx
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
## （2）git fastmoe
```
git clone https://github.com/laekov/fastmoe.git
cd fastmoe
USE_NCCL=1 python setup.py install
pip install dm-tree
```

# 2、Download via HF
## Download the Hugging Face
```
git lfs install
git clone https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen
```

## Merge Files(Only for HF)
cd mixtral-8x7b-32kseqlen/

## Merge the checkpoints
```
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
```


# 3、编译MixtralKit
```
git clone https://github.com/Xingzhi107/MoeTest.git
cd MoeTest
pip install -r requirements.txt
pip install -e .
ln -s path/to/checkpoints_folder/ ckpts
```

# 4、运行
### 不开策略，单机：
```markdown
torchrun --standalone --nproc_per_node=4  tools/example.py -m ./ckpts -t ckpts/tokenizer.model
```
### 开策略，无用：
```markdown
FMOE_FASTER_SCHEDULE_ENABLE=1 FMOE_FASTER_SHADOW_ENABLE=1 torchrun --standalone --nproc_per_node=4  tools/example.py -m ./ckpts -t ckpts/tokenizer.model
```

### 单机多卡：
```markdown
torchrun --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29540 \
    tools/example.py -m ./ckpts -t ckpts/tokenizer.model
```
### 多机多卡：
```markdown
torchrun --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr 172.20.20.140 \
    --master_port 29540 \
    tools/example.py -m ./ckpts -t ckpts/tokenizer.model

torchrun --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr 172.20.20.140 \
    --master_port 29540 \
    tools/example.py -m ./ckpts -t ckpts/tokenizer.model
```
# 问题：
不确定torch.load是否会报错
mixtralkit/mixtral/generation.py

修改专家数量在tools/example.py
    num_gpus=1, //忘记改名为expert_num了
    world_size=world_size,
world_size设置为nnode*nproc_per_node
多机的时候不确定是nnode*nproc_per_node还是nproc_per_node