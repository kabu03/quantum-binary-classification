import os
import json
import csv
import argparse
import pandas as pd

def load_metrics_from_dir(run_dir):
    """Loads all metrics JSON files from the specified directory."""
    metrics_data = []
    try:
        for filename in os.listdir(run_dir):
            if filename.startswith("metrics_") and filename.endswith(".json"):
                filepath = os.path.join(run_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        # Ensure essential keys exist, add filename for reference
                        if 'backend' in data and 'dataset' in data and 'accuracy' in data and 'f1_score' in data:
                             metrics_data.append({
                                'backend': data['backend'],
                                'dataset': data['dataset'],
                                'accuracy': data['accuracy'],
                                'f1_score': data['f1_score'],
                                'file': filename # Optional: keep track of source file
                            })
                        else:
                            print(f"[Warning] Skipping metrics file {filename}: missing required keys.")
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping metrics file {filename}: invalid JSON.")
                except Exception as e:
                    print(f"[Warning] Error reading metrics file {filename}: {e}")
    except FileNotFoundError:
        print(f"[Error] Metrics directory not found: {run_dir}")
    except Exception as e:
        print(f"[Error] Failed to list files in metrics directory {run_dir}: {e}")
    return metrics_data

def load_timings_from_csv(run_dir):
    """Loads timings from the timings.csv file."""
    timings_file = os.path.join(run_dir, "timings.csv")
    timings_data = []
    try:
        with open(timings_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                 # Ensure essential keys exist
                 if 'backend' in row and 'dataset' in row and 'train_time_seconds' in row and 'predict_time_seconds' in row:
                    timings_data.append({
                        'backend': row['backend'],
                        'dataset': row['dataset'],
                        'train_time_seconds': float(row['train_time_seconds']) if row['train_time_seconds'] != '--' else None,
                        'predict_time_seconds': float(row['predict_time_seconds']) if row['predict_time_seconds'] != '--' else None
                    })
                 else:
                     print(f"[Warning] Skipping row in timings.csv: {row} - missing required keys.")

    except FileNotFoundError:
        print(f"[Warning] timings.csv not found in {run_dir}. Timing data will be missing.")
    except Exception as e:
        print(f"[Error] Failed to read timings.csv from {run_dir}: {e}")
    return timings_data

def generate_latex_table(run_dir):
    """Generates a LaTeX table string from metrics and timings."""

    metrics = load_metrics_from_dir(run_dir)
    timings = load_timings_from_csv(run_dir)

    if not metrics:
        print("[Error] No valid metrics data found. Cannot generate table.")
        return None

    # Convert to Pandas DataFrames for easier merging
    metrics_df = pd.DataFrame(metrics)
    timings_df = pd.DataFrame(timings)

    # Merge metrics and timings data
    if not timings_df.empty:
        # Use outer merge to keep all metrics even if timings are missing
        combined_df = pd.merge(metrics_df, timings_df, on=['backend', 'dataset'], how='left')
    else:
        # If no timings, create placeholder columns
        combined_df = metrics_df
        combined_df['train_time_seconds'] = None
        combined_df['predict_time_seconds'] = None

    # Define the desired order for rows (backends and datasets)
    # Add any other backends you expect here
    backend_order = [
        "classical_svm_cpu",
        "qsvm_statevector",
        "qsvm_qasm",
        "qsvm_ibm_sherbrooke", # Add expected hardware backends
        "qsvm_ibm_brisbane"    # Add other potential backends
    ]
    # Filter and sort based on known backends and datasets
    combined_df['backend'] = pd.Categorical(combined_df['backend'], categories=backend_order, ordered=True)
    combined_df = combined_df.sort_values(by=['dataset', 'backend'])

    # --- LaTeX Table Generation ---
    latex_string = "\\begin{table}[htbp]\n"
    latex_string += "\\centering\n"
    # Adjust column count (6) and alignment (l l r r r r) as needed
    latex_string += "\\begin{tabular}{llrrrr}\n"
    latex_string += "\\toprule\n"
    # Header row
    latex_string += "Dataset & Backend & Accuracy & F1-Score & Train (s) & Predict (s) \\\\\n"
    latex_string += "\\midrule\n"

    # Data rows
    last_dataset = None
    for _, row in combined_df.iterrows():
        # Optional: Add a midrule between datasets for clarity
        if last_dataset is not None and row['dataset'] != last_dataset:
            latex_string += "\\midrule\n"

        # Format numerical values, handle None/NaN for timings
        acc_str = f"{row['accuracy']:.3f}" if pd.notna(row['accuracy']) else "--"
        f1_str = f"{row['f1_score']:.3f}" if pd.notna(row['f1_score']) else "--"
        train_str = f"{row['train_time_seconds']:.3f}" if pd.notna(row['train_time_seconds']) else "--"
        pred_str = f"{row['predict_time_seconds']:.3f}" if pd.notna(row['predict_time_seconds']) else "--"

        # Escape underscores in backend names for LaTeX
        backend_str = row['backend'].replace('_', '\\_')
        dataset_str = row['dataset'].replace('_', '\\_') # Escape underscores in dataset names too

        latex_string += f"{dataset_str} & {backend_str} & {acc_str} & {f1_str} & {train_str} & {pred_str} \\\\\n"
        last_dataset = row['dataset']

    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"
    # Add caption and label
    run_name = os.path.basename(run_dir) # Get the 'run_...' name
    latex_string += f"\\caption{{Comparison results for run \\texttt{{{run_name.replace('_', '\\_')}}}.}}\n"
    latex_string += f"\\label{{tab:results_{run_name}}}\n"
    latex_string += "\\end{table}\n"

    return latex_string

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from experiment results.")
    parser.add_argument("run_dir", help="Path to the specific run directory (e.g., results/run_20250502_182905)")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"[Error] Provided path is not a valid directory: {args.run_dir}")
        return

    latex_output = generate_latex_table(args.run_dir)

    if latex_output:
        print("\n--- Generated LaTeX Table ---")
        print(latex_output)
        # Optionally, save to a file:
        # output_file = os.path.join(args.run_dir, "results_table.tex")
        # with open(output_file, 'w') as f:
        #     f.write(latex_output)
        # print(f"\nLaTeX table saved to: {output_file}")

if __name__ == "__main__":
    main()