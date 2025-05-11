export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
python ./main.py --subset "imagenetfruitLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 300 --imbalance_rate  0.1 > imagenetfruit1.log
# python ./main.py --subset "imagenetmeowLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.1 > imagenetmeow1.log
# python ./main.py --subset "imagenetnetteLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.1 > imagenetnette1.log
# python ./main.py --subset "imagenetsquawkLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.1 > imagenetsquawk1.log
# python ./main.py --subset "imagenetwoofLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.1 > imagenetwoof1.log
# python ./main.py --subset "imagenetyellowLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 10 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.1 > imagenetyellow1.log
