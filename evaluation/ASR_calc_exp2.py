# 'evaluate_ASR_exp2.py' script

from argparse import ArgumentParser
import pandas as pd
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
    csv_files_dir = results_json_dir / "csv_files"
    
    if not csv_files_dir.exists():
        print(f"Error: Directory {csv_files_dir} does not exist.")
        exit(1)

    all_results = []

    # 1. Process Files and Calculate Pair-Level ASR
    for file_path in csv_files_dir.glob("*.csv"):
        if "_ASR" in file_path.name or "statistics" in file_path.name:
            continue

        df = pd.read_csv(file_path)
        asr_value = df['Success'].mean()

        # Filename format: ..._agent_0_agent_1_seed42_collusion_run.csv
        # We need to extract the "agent_X_agent_Y" part
        file_name = file_path.name
        parts = file_name.split('_')
        
        # Finding indices for agent names; based on your format, 
        # they appear before 'seed'
        try:
            seed_index = parts.index('seed42') # or find by prefix 'seed'
        except ValueError:
            # Fallback logic if seed varies: look for 'agent' markers
            seed_index = [i for i, s in enumerate(parts) if 'seed' in s][0]
            
        # The two agents are the two segments immediately preceding the seed
        agent_a = f"{parts[seed_index-4]}_{parts[seed_index-3]}" # agent_0
        agent_b = f"{parts[seed_index-2]}_{parts[seed_index-1]}" # agent_1
        colluded_pair = f"{agent_a}_{agent_b}"

        all_results.append({
            "filename": file_name,
            "ASR": asr_value,
            "Adversarial_agent": colluded_pair,
            "agent_list": [agent_a, agent_b] # Helper list for agent-level stats
        })

    detailed_df = pd.DataFrame(all_results)

    # Save detailed ASR report
    detailed_output = results_json_dir / f"{args.folder}_ASR.csv"
    detailed_df.drop(columns=['agent_list']).to_csv(detailed_output, index=False)

    # 2. Group-Level Statistics (The Pairs)
    group_stats = detailed_df.groupby("Adversarial_agent")["ASR"].mean().reset_index()
    group_stats.columns = ["Adversarial_group", "Mean_ASR"]
    
    # Add Total System Mean
    total_mean = detailed_df["ASR"].mean()
    total_row = pd.DataFrame({"Adversarial_group": ["TOTAL_SYSTEM_MEAN"], "Mean_ASR": [total_mean]})
    group_stats = pd.concat([group_stats, total_row], ignore_index=True)
    
    group_stats.to_csv(results_json_dir / "exp2_group_level_ASR_statistics.csv", index=False)

    # 3. Agent-Level Statistics (Individual performance within pairs)
    # Explode the agent_list so each row is duplicated for each participating agent
    agent_level_df = detailed_df.explode('agent_list')
    agent_stats = agent_level_df.groupby('agent_list')['ASR'].mean().reset_index()
    agent_stats.columns = ["Individual_Agent", "Individual_Mean_ASR"]

    agent_stats.to_csv(results_json_dir / "exp2_agent_level_ASR_statistics.csv", index=False)

    print(f"Processing complete for {args.folder}")
    print(f"- Group stats saved to exp2_group_level_ASR_statistics.csv")
    print(f"- Agent stats saved to exp2_agent_level_ASR_statistics.csv")