# python main.py --run_name AL_trail-1 \
#                 --dataset blogger \
#                 --model MLP \
#                 --learning_rate 1e-4 \
#                 --learning_rate_s 2 \
#                 --method AL \
#                 --subprob_max_epoch 100 \
#                 --rounds 0 \
#                 --alpha 0.99 \
#                 --t 0.5 \
#                 --solver AdamW \
#                 --warm_start 500 \
#                 --rho 1 \
#                 --delta 1
                

python main.py --run_name WCE_trail1 \
               --dataset blogger \
               --model MLP \
               --learning_rate 1e-4 \
               --method WCE \
               --rounds 500
            