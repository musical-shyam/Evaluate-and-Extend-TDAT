#!/bin/bash
#SBATCH --job-name=tdat-multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpucompute-a40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gmanikandan@binghamton.edu

module load cuda/12.4
source /data/home/gmanikandan/dust3r_env/bin/activate

mkdir -p logs

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501   # (different port from dust3r if running parallel)
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=0

echo "Running on node: $(hostname)"

# Launch using torchrun
srun torchrun \
     --nnodes=1 \
     --nproc_per_node=4 \
     --rdzv_id=$RANDOM \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     TDAT.py --batch-m 0.75 --delta-init "random" --out-dir "CIFAR100" --log "DeiT.log" --model "DeiT-Small" --lamda 0.6 --inner-gamma 0.15 --outer-gamma 0.15 --save-epoch 1 --dataset CIFAR100
