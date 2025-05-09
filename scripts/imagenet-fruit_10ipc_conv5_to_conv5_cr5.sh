export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
python ./main.py \
--subset "imagenet-fruit" \
--arch-name "conv5" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "conv5" \
--re-epochs 300
