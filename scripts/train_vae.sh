export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PYTHONPATH:$(pwd)

time=$(date +%m%d_%H%M)
EXP_NAME="vae_${time}"

mkdir -p exps/${EXP_NAME}/logs

nohup python scripts/train_vae.py --exp_name ${EXP_NAME} > exps/${EXP_NAME}/logs/run.log 2>&1 &

sleep 1

tail -f exps/${EXP_NAME}/logs/run.log
