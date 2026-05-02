# 'evaluate_result_exp2.py' script

from argparse import ArgumentParser
from evaluation_functions import *
import json
import pandas as pd
import os
from pathlib import Path
import sys

if __name__=="__main__":
    args = ArgumentParser()

    args.add_argument("--filename", type=str, help="The JSON result file from exp2")
    args.add_argument("--folder", type=str, default="exp2", help="Folder inside results/")
    args.add_argument("--environment", type=str, choices=["multi_agent_debate"], default="multi_agent_debate")
    args.add_argument("--dataset", type=str, default="BAD-ACTS.csv", help="The original CSV dataset")

    args = args.parse_args()

    #########################################
    ## Folder and files paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent # Based on your file structure

    dataset_dir = project_root / "datasets"
    results_json_dir = project_root / "results" / args.folder
    
    # Path for output CSVs
    results_csv_dir = results_json_dir / "csv_files"
    results_csv_dir.mkdir(parents=True, exist_ok=True)

    json_file_path = results_json_dir / args.filename
    res_csv_file_path = results_csv_dir / Path(args.filename).with_suffix('.csv')

    # Verification
    bad_acts_fpath = dataset_dir / args.dataset
    if not bad_acts_fpath.exists():
        print(f"ERROR: Dataset not found at {bad_acts_fpath}")
        sys.exit(1)

    if not json_file_path.exists():
        print(f"ERROR: JSON file not found at {json_file_path}")
        sys.exit(1)
    #########################################

    # 1. Load the reference dataset
    # We need this to recover the missing 'target_agent' (the "Target" column)
    df_reference = pd.read_csv(bad_acts_fpath)
    # Filter for environment to ensure indices match your run logic
    df_reference = df_reference[df_reference["Environment"] == args.environment].reset_index(drop=True)

    # 2. Load the JSON experimental data
    with open(json_file_path) as f:
        data = json.load(f)

    # select correct evaluation function
    eval_fn = {
        "multi_agent_debate" : evaluate_MAD
    }[args.environment]

    success = []
    
    print(f"Processing {len(data)} datapoints...")

    for datapoint in data:
        # FIX: The JSON is missing 'target_agent'. 
        # We find the corresponding row in the CSV using the 'id'
        row_id = datapoint["id"]
        # In your 'run_experiments_exp2.py', the 'id' corresponds to the index of 
        # the filtered target_actions dataframe.
        target_agent_value = df_reference.iloc[row_id]["Target"]
        
        # Inject it back into the dictionary so evaluate_MAD can read it
        datapoint["target_agent"] = target_agent_value
        
        # Now run the evaluation
        try:
            is_success = eval_fn(datapoint)
            success.append(is_success)
        except Exception as e:
            print(f"Error evaluating row {row_id}: {e}")
            success.append(False)
    
    asr = sum(success)/len(success) if len(success) > 0 else 0
    print(f"\n--- Result ---")
    print(f"File: {args.filename}")
    print(f"Attack Success Rate: {asr}")

    # 3. Store results in CSV
    # We use a copy of the reference dataframe to store the 'Success' column
    results_df = df_reference.copy()
    results_df["Success"] = success
    
    results_df.to_csv(res_csv_file_path, index=False)
    print(f"Saved results to: {res_csv_file_path}\n")