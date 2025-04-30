import os
import json
import csv

def save_metrics(metrics, backend_name, results_dir='results'):
    """
    Saves metrics to a JSON file.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    dataset_name = metrics["dataset"]
    file_path = os.path.join(results_dir, f'metrics_{dataset_name}_{backend_name}.json')
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_timing(training_duration, predicting_duration, backend_name, dataset_name, results_dir='results'):
    """
    Saves timing information to a CSV file.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timings_path = os.path.join(results_dir, 'timings.csv')
    file_exists = os.path.isfile(timings_path)
    with open(timings_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['dataset', 'backend', 'train_time_seconds', 'predict_time_seconds'])
        writer.writerow([dataset_name, backend_name, training_duration, predicting_duration])