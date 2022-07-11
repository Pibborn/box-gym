import pandas as pd
import wandb

SCALE = 0
BASKETBALL = 1
ORBIT = 2

def wandbCSVTracking(run, csv_path, config=None, mode=SCALE):
    mode_string = ["scale", "basketball", "orbit"][mode]
    # ----------- Tracking of the CSV File ---------------------------------
    # Read our CSV into a new DataFrame
    new_dataframe = pd.read_csv(csv_path)
    new_dataframe.drop(["Unnamed: 0"], axis=1, inplace=True)

    # Convert the DataFrame into a W&B Table
    table = wandb.Table(dataframe=new_dataframe)

    # Add the table to an Artifact to increase the row limit to 200000 and make it easier to reuse!
    table_artifact = wandb.Artifact(f"{mode_string}_artifact", type="dataset")
    table_artifact.add(table, f"{mode_string}_table")

    # We will also log the raw csv file within an artifact to preserve our data
    table_artifact.add_file(csv_path)

    # Start a W&B run to log data
    # run = wandb.init(project="box-gym", entity=args.entity, config=config, sync_tensorboard=True)

    # Log the table to visualize with a run...
    run.log({f"{mode_string}": table})

    # and Log as an Artifact to increase the available row limit!
    run.log_artifact(table_artifact)
    return run