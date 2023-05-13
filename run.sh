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

# breast-cancer-wisc
# wilt

python main.py --run_name constrained_IL_NIPS \
                --dataset cifar100 \
                --model MLP \
                --batch_size 1024 \
                --learning_rate 5e-4 \
                --learning_rate_s 1e-2 \
                --method AL_OFBS \
                --subprob_max_epoch 1 \
                --rounds 20 \
                --alpha 1 \
                --t 0.5 \
                --solver AdamW \
                --warm_start 1 \
                --rho 100 \
                --delta 1.2 \
                --sto 1 \
                --reg 0.1 \
                --saved_fig_name 01_ws \
                --random_seed $1


# python main.py --run_name constrained_IL_NIPS \
#                --dataset cifar100\
#                --model MLP \
#                --learning_rate 1e-4 \
#                --method WCE \
#                --rounds 200 \
#                --batch_size 1024