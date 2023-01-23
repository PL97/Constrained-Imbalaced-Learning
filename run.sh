python main.py --run_name AL_trail1_s \
                --dataset diabetic \
                --model MLP \
                --batch_size 100 \
                --learning_rate 1e-5 \
                --learning_rate_s 0.1 \
                --method AL \
                --subprob_max_epoch 50 \
                --rounds 100 \
                --alpha 0.90 \
                --t 0.5 \
                --solver AdamW \
                --warm_start 0 \
                --rho 1 \
                --delta 1 

                

# python main.py --run_name WCE_trail1_s \
#                --dataset diabetic \
#                --model MLP \
#                --learning_rate 1e-4 \
#                --method WCE \
#                --rounds 10
#                --batch_size 200          