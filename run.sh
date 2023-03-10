# CUDA_VISIBLE_DEVICES=0 python main.py --run_name AL_trail1_s \
#                 --dataset $2\
#                 --model MLP \
#                 --batch_size 1000 \
#                 --learning_rate 5e-4 \
#                 --learning_rate_s 5e-2\
#                 --method $1 \
#                 --subprob_max_epoch 100 \
#                 --rounds $3 \
#                 --alpha 0.99 \
#                 --t 0.5 \
#                 --solver AdamW \
#                 --warm_start 20 \
#                 --rho 1 \
#                 --delta 1 \
#                 --sto 0


CUDA_VISIBLE_DEVICES=0 python main.py --run_name constrained_IL \
                --dataset cifar100\
                --model MLP \
                --batch_size 8000 \
                --learning_rate 5e-4 \
                --learning_rate_s 5e-2\
                --method AL_FROP \
                --subprob_max_epoch 50 \
                --rounds 20 \
                --alpha 0.99 \
                --t 0.5 \
                --solver AdamW \
                --warm_start 0 \
                --rho 1 \
                --delta 1 \
                --sto 1

                

# CUDA_VISIBLE_DEVICES=0 python main.py --run_name WCE_trail1_s \
#                --dataset cifar100 \
#                --model MLP \
#                --learning_rate 1e-4 \
#                --method WCE \
#                --rounds 500 \
#                --batch_size 1024