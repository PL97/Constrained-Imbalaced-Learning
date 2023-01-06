# Constrained-Imbalaced-Learning

# Dependency
```
wandb, PyTorch
```
For wandb, please create an account and set up the environment following the [guide](https://wandb.ai/site).

# Get Started
```bash
chmod +x run.sh
./run.sh
```


# what is in run.sh
solve the constrained optimization problem using the Augmented Lagrangian method
```
python main.py --run_name AL_trail10 \
                --dataset diabetic \
                --model MLP \
                --batch_size 100 \
                --learning_rate 1e-4 \
                --learning_rate_s 2 \
                --method AL \
                --subprob_max_epoch 100 \
                --rounds 100 \
                --alpha 0.99 \
                --t 0.5 \
                --solver AdamW \
                --warm_start 10 \
                --rho 1 \
                --delta 1
```

solve the unconstrained problem with weighted cross-entropy loss               
```
python main.py --run_name WCE_trail1 \
               --dataset magic \
               --model MLP \
               --learning_rate 1e-4 \
               --method WCE \
               --rounds 300
```