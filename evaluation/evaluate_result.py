# 'evaluate_result.py' script

from argparse import ArgumentParser
from evaluation_functions import *
import json
import pandas as pd

import os
from pathlib import Path
import sys

if __name__=="__main__":
    args = ArgumentParser()

    args.add_argument("--filename", type=str)
    args.add_argument("--folder", type=str) # Add this line
    args.add_argument("--environment", type=str, choices=["travel_planning", "financial_article_writing", "code_generation", "multi_agent_debate"])
    # args.add_argument("--res-path", type=str)

    args = args.parse_args()

    #########################################
    ## Folder and files paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    dataset_dir = project_root / "datasets"

    results_json_dir = project_root / "results" / args.folder
    results_csv_dir = project_root / "results" / args.folder

    json_file = Path(args.filename)
    csv_file = json_file.with_suffix('.csv')
    # print(csv_file) # Output: file1.csv

    bad_acts_fpath = dataset_dir / "copy_BAD-ACTS.csv"
    json_file_path = results_json_dir / args.filename
    res_csv_file_path = results_csv_dir / csv_file

    # Create a shorter version for the print statement
    short_path = res_csv_file_path.relative_to(project_root.parent)

    #########################################

    # load data
    # with open(args.path) as f:
    #     data = json.load(f)
    
    with open(json_file_path) as f:
        data = json.load(f)

    # select correct evaluation function
    eval_fn = {
        # "travel_planner" : evaluate_travel_planning,
        "travel_planning" : evaluate_travel_planning,
        "financial_article_writing" : evaluate_financial_article_writing,
        "code_generation" : evaluate_code_generation,
        "multi_agent_debate" : evaluate_MAD
    }[args.environment]

    success = []
    for datapoint in data:
        success.append(eval_fn(datapoint))
    
    print(f"Attack Success Rate: {sum(success)/len(success)}")

    # Store success of all actions in a csv
    if args.filename:
        # results = pd.read_csv(f"datasets/BAD-ACTS.csv")
        results = pd.read_csv(bad_acts_fpath)
        results = results[results["Environment"] == args.environment]
        results["Success"] = success
        # results.to_csv(args.res_path)
        results.to_csv(res_csv_file_path, index=False)
        print(f"Saved csv file at :   {short_path}\n")