export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
python ./main.py --subset "imagenetfruitLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 1 --stud-name "conv5" --re-epochs 300 --imbalance_rate  0.5 > imagenetfruit4.log
# python ./main.py --subset "imagenetmeowLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 1 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetmeow4.log
# python ./main.py --subset "imagenetnetteLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 1 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetnette4.log
# python ./main.py --subset "imagenetsquawkLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 1 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetsquawk4.log
# python ./main.py --subset "imagenetwoofLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 1 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetwoof4.log
# python ./main.py --subset "imagenetyellowLT" --arch-name "conv5" --factor 2 --num-crop 5 --mipc 300 --ipc 1 --stud-name "conv5" --re-epochs 200 --imbalance_rate  0.5 > imagenetyellow4.log
