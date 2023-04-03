import wandb
import  torch

def log(constraints, obj, train_metrics, val_metrics, test_metrics, verbose=True, **args):
    
    r = args['r']
    rounds = args['rounds']
    
    if verbose:
        print(f"========== Round {r}/{rounds} ===========")
        print("Precision: {:3f} \t Recall {:3f} \t F_beta {:.3f} \t AP {:.3f}".format(\
                train_metrics['precision'], train_metrics['recall'], train_metrics['F_beta'], train_metrics['AP']))
        print("Obj: {}\tIEQ: {}\tEQ: {}".format(obj, constraints[0].item(), torch.sum(constraints[1:]).item()))
        print("(val)Precision: {:3f} \t Recall {:3f} F_beta {:.3f} AP:{:.3f}".format(\
                val_metrics['precision'], val_metrics['recall'], val_metrics['F_beta'], val_metrics['AP']))
            
        print("(test)Precision: {:3f} \t Recall {:3f} F_beta {:.3f} AP:{:.3f}".format(\
                test_metrics['precision'], test_metrics['recall'], test_metrics['F_beta'], test_metrics['AP']))
    
    
    wandb.log({ "trainer/global_step": r, \
            "train/Obj": obj, \
            "train/IEQ": constraints[0].item(), \
            "train/EQ": torch.sum(constraints[1:]).item(), \
            "train/Precision": train_metrics['precision'], \
            "train/Recall": train_metrics['recall'], \
            "train/F_beta": train_metrics['F_beta' ], \
            "train/AP": train_metrics['AP'], \
            "val/Precision": val_metrics['precision'], \
            "val/Recall": val_metrics['recall'], \
            "val/F_beta": val_metrics['F_beta'], \
            "val/AP": val_metrics['AP'], \
            "test/Precision": test_metrics['precision'], \
            "test/Recall": test_metrics['recall'], \
            "test/F_beta": test_metrics['F_beta'], \
            "test/AP": test_metrics['AP']
            })