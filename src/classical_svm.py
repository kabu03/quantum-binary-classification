from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import time
from src.metrics_logger import save_metrics, save_timing
from src.data_utils import load_banknote, generate_two_moons, clean_split_scale

def search_classical_hparams(X_train, y_train, X_val, y_val):
    """
    Performs a basic grid search over C and gamma values.
    Chooses best hyperparameters by validation accuracy, breaking ties using F1-score.
    """
    print("\nStarting hyperparameter search...")

    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.01, 0.1, 1, 10]
    best_acc = 0
    best_f1 = 0
    best_params = {}

    for C in C_values:
        for gamma in gamma_values:
            model = svm.SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred)

            print(f"Tested C={C}, gamma={gamma} -> Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

            # Update if:
            # (1) better accuracy, OR
            # (2) equal accuracy but better F1
            if (acc > best_acc) or (acc == best_acc and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_params = {'C': C, 'gamma': gamma}

    print(f"\nBest hyperparameters found: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"Best validation accuracy: {best_acc:.4f}, Best F1 Score: {best_f1:.4f}")
    return best_params


def train_classical_svm(X_train, y_train, best_params):
    """
    Trains the SVM model using the best found hyperparameters.
    """
    print("\nTraining final SVM model...")
    t0 = time.time()
    model = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=42)
    model.fit(X_train, y_train)
    training_duration = time.time() - t0
    print("Training completed.")
    return model, training_duration


def eval_and_log(dataset_name, model, training_duration, X_test, y_test, backend_name='classical_svm_cpu'):
    """
    Evaluates the trained model on the test set and logs metrics and timing.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    predicting_duration = time.time() - t0

    metrics = {
        "dataset": dataset_name,
        "backend": backend_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred).tolist()
    }
    print("\nFinal Evaluation on Test Set:")
    print(json.dumps(metrics, indent=2))

    save_metrics(metrics, backend_name)
    save_timing(training_duration, predicting_duration, backend_name, dataset_name)


def run_banknote(backend_name="classical_svm_cpu"):
    """
    Runs the classical SVM pipeline for the Banknote Authentication dataset.
    """
    print("\nRunning classical SVM on Banknote Authentication dataset...")
    df = load_banknote()
    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)
    
    # Perform hyperparameter search
    best_params = search_classical_hparams(X_train, y_train, X_val, y_val)
    
    # Train the model with the best hyperparameters
    model, training_duration = train_classical_svm(X_train, y_train, best_params)
    
    # Evaluate the model and log metrics
    eval_and_log("banknote", model, training_duration, X_test, y_test, backend_name=backend_name)


def run_two_moons(backend_name="classical_svm_cpu"):
    """
    Runs the classical SVM pipeline for the Two Moons dataset.
    """
    print("\nRunning classical SVM on Two Moons dataset...")
    df = generate_two_moons()
    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)
    
    # Perform hyperparameter search
    best_params = search_classical_hparams(X_train, y_train, X_val, y_val)
    
    # Train the model with the best hyperparameters
    model, training_duration = train_classical_svm(X_train, y_train, best_params)
    
    # Evaluate the model and log metrics
    eval_and_log("two_moons", model, training_duration, X_test, y_test, backend_name=backend_name)


if __name__ == "__main__":
    run_banknote()
    run_two_moons()