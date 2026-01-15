export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
torchrun \
  --nproc_per_node 7 \
  --master_port 29400 \
  train.py