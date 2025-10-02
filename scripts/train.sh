export PYTHONPATH=$PYTHONPATH:$(pwd)

time=$(date +%m%d_%H%M)
EXP_NAME="test_${time}"

mkdir -p exps/${EXP_NAME}/logs

# python scripts/train.py --exp_name ${EXP_NAME}

nohup python scripts/train.py --exp_name ${EXP_NAME} > exps/${EXP_NAME}/logs/run.log 2>&1 &

sleep 1

tail -f exps/${EXP_NAME}/logs/run.log
