# CUDA_VISIBLE_DEVICES=0 python main.py --run_name AL_trail1_s \
#                 --dataset cifar100\
#                 --model MLP \
#                 --batch_size 4000 \
#                 --learning_rate 1e-4 \
#                 --learning_rate_s 5e-2\
#                 --method AL_FPOR \
#                 --subprob_max_epoch 200 \
#                 --rounds 100 \
#                 --alpha 0.99 \
#                 --t 0.5 \
#                 --solver AdamW \
#                 --warm_start 0 \
#                 --rho 1 \
#                 --delta 1 \
#                 --sto 0

                

CUDA_VISIBLE_DEVICES=0 python main.py --run_name WCE_trail1_s \
               --dataset cifar100 \
               --model MLP \
               --learning_rate 1e-4 \
               --method WCE \
               --rounds 300 \
               --batch_size 1000