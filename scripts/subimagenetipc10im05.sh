export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
python ./main.py --subset "imagenetfruitLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 300 --imbalance_rate  0.5 > imagenetfruit2.log
# python ./main.py --subset "imagenetmeowLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetmeow2.log
python ./main.py --subset "imagenetnetteLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 300 --imbalance_rate  0.5 > imagenetnette2.log
# python ./main.py --subset "imagenetsquawkLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetsquawk2.log
# python ./main.py --subset "imagenetwoofLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetwoof2.log
# python ./main.py --subset "imagenetyellowLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetyellow2.log
