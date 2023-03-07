CUDA_VISIBLE_DEVICES=0 python main.py --run_name WCE_trail1_s \
               --dataset $1 \
               --model MLP \
               --learning_rate 1e-4 \
               --method WCE \
               --rounds $2 \
               --batch_size 256