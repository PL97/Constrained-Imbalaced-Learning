CUDA_VISIBLE_DEVICES=1 python main.py --run_name AL_trail1_s \
                --dataset diabetic \
                --model MLP \
                --batch_size 1024 \
                --learning_rate 1e-4 \
                --learning_rate_s 0.1 \
                --method AL_FROP \
                --subprob_max_epoch 500 \
                --rounds 100 \
                --alpha 0.99 \
                --t 0.5 \
                --solver AdamW \
                --warm_start 0 \
                --rho 1 \
                --delta 1

                

# CUDA_VISIBLE_DEVICES=3 python main.py --run_name WCE_trail1_s \
#                --dataset wilt \
#                --model MLP \
#                --learning_rate 1e-4 \
#                --method WCE \
#                --rounds 300 \
#                --batch_size 256         