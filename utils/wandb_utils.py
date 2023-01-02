import wandb

class progress_tracker:
    def __init__(self):
        self.wandb_run = wandb.init(project='FPOR', \
                   name=self.args.run_name, \
                   group=self.args.dataset, \
                   config={
                    'subprob_max_epoch': self.subprob_max_epoch, \
                    'rounds': self.rounds, \
                    'lr': self.lr, \
                    'alpha': self.alpha, \
                    't': self.t, \
                    'solver': self.solver \
                   })
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/precision", summary="max")
        wandb.define_metric("val/recall", summary="max")