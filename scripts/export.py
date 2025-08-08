import pandas as pd 
import wandb
import argparse
from tqdm import tqdm

def main(sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"shamus-team/multiplexed-pixels/{sweep_id}")
    # Project is specified by <entity/project-name>
    runs = sweep.runs

    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs): 
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })

    runs_df.to_csv(f"project_{sweep.id}.csv")
    print(f"Saved runs data to project_{sweep.id}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id", type=str, help="wandb sweep ID")
    args = parser.parse_args()

    main(args.sweep_id)
