import pandas as pd 
import wandb
from tqdm import tqdm

api = wandb.Api()
sweep = api.sweep("shamus-team/multiplexed-pixels/w2wbo923")

# Project is specified by <entity/project-name>
runs = sweep.runs

summary_list, config_list, name_list = [], [], []
for run in tqdm(runs): 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv(f"project_{sweep.id}.csv")