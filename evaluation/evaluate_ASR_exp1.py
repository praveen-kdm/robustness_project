# 'evaluate_ASR.py' script

from argparse import ArgumentParser
import json
import pandas as pd
import os
from pathlib import Path

if __name__=="__main__":
    args = ArgumentParser()
    args.add_argument("--folder", type=str, required=True)
    args = args.parse_args()

    #########################################
    ## Folder and files paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    results_json_dir = project_root / "results" / args.folder
    # Directory containing the source CSVs to process
    csv_files_dir = results_json_dir / "csv_files"
    
    if not csv_files_dir.exists():
        print(f"Error: Directory {csv_files_dir} does not exist.")
        exit(1)

    all_results = []

    # Process each CSV file in the directory
    for file_path in csv_files_dir.glob("*.csv"):
        # Skip the output file if it already exists in the same folder
        if file_path.name.endswith("_ASR.csv"):
            continue

        df = pd.read_csv(file_path)

        # Calculate ASR: sum(success) / len(success)
        # Assumes 'Success' column contains booleans or 0/1
        asr_value = df['Success'].mean()

        # Extract Adversarial Agent from filename
        # Format: 'llama3.1_8b_multi_agent_debate_18_agent_0_exp1_seed_123_agent_0.csv'
        # We split by '_' and take the last two parts before the extension
        file_name = file_path.name
        name_parts = file_name.replace(".csv", "").split("_")
        # In your example 'agent_0' is the suffix before .csv
        adversarial_agent = f"{name_parts[-2]}_{name_parts[-1]}"

        all_results.append({
            "filename": file_name,
            "ASR": asr_value,
            "Adversarial_agent": adversarial_agent
        })

    # Create the detailed DataFrame
    detailed_df = pd.DataFrame(all_results)

    # Save the granular table
    output_filename = f"{args.folder}_ASR.csv"
    detailed_df.to_csv(results_json_dir / output_filename, index=False)
    print(f"Detailed ASR report saved to {results_json_dir / output_filename}")

    # Create the summary DataFrame
    # 1. Mean ASR per individual adversarial agent
    summary_df = detailed_df.groupby("Adversarial_agent")["ASR"].mean().reset_index()
    summary_df.columns = ["Adversarial_agent", "Mean_ASR"]

    # 2. Total Mean ASR across all files
    total_mean = detailed_df["ASR"].mean()
    
    # Append the total mean as a final row for a complete overview
    total_row = pd.DataFrame({
        "Adversarial_agent": ["TOTAL_SYSTEM_MEAN"], 
        "Mean_ASR": [total_mean]
    })
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)

    # Save the summary table
    summary_output_filename = f"{args.folder}_agent_level_ASR_statistics.csv"
    summary_df.to_csv(results_json_dir / summary_output_filename, index=False)
    
    print(f"Summary statistics saved to {results_json_dir / summary_output_filename}")
    print("\nSummary Overview:")
    print(summary_df)

    