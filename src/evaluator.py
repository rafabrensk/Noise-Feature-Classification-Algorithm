import matplotlib.pyplot as plt
from pathlib import Path

def ResultsReportsPath():
    project_dir = Path(__file__).resolve().parent.parent
    dir_dataset = project_dir / "results" / "reports"
    print("[Results directory found]")
    return dir_results
    
def ResultsChartsPath():
    project_dir = Path(__file__).resolve().parent.parent
    dir_dataset = project_dir / "results" / "charts"
    print("[Results directory found]")
    return dir_results

def generate_charts(results: dict):
    model = results["model"]
    ac_treino = results["training_accuracy"]
    ac_teste = results["test_accuracy"]
    df_importances = results["df_importances"]
    n_train = results["n_train"]
    n_test = results["n_test"]
    print("[Metrics found successfully]")

    plt.figure(figsize=(10, 6))
    plt.barh(
        df_importances["Feature"],
        df_importances["Importance"],
        color="steelblue",
    )

    plt.suptitle(
        f"Feature Importance - R.F ({n_train} Train Samples) - "
        f"Train Acc: {ac_treino * 100:.2f}% - "
        f"Test Acc: {ac_teste * 100:.2f}% "
    )
    plt.title(
        f"RF Params | n_estimators: {model.n_estimators}, "
        f"random_state: {model.random_state}, "
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    print("[Charts created successfully]")

    plt.savefig(ResultsChartsPath()/"test.png")

def generate_reports(results: dict):
    model = results["model"]
    ac_treino = results["training_accuracy"]
    ac_teste = results["test_accuracy"]
    df_importances = results["df_importances"]
    n_train = results["n_train"]
    n_test = results["n_test"]
    oob_score = results["oob_score"]

    nome_txt = f"test.txt"

    with open(ResultsReportsPath() / nome_txt, "w") as arquivo_txt:
        arquivo_txt.write(
            f"Train Samples: {n_train}, Test Samples: {n_test}\n"
        )
        arquivo_txt.write(
            f"Number of Trees: {model.n_estimators}, "
            f"Random State: {model.random_state}, "
            f"Max Depth: {model.max_depth}\n\n"
        )
        arquivo_txt.write(
            f"Train Accuracy: {ac_treino:.4f}\n"
            f"Test Accuracy: {ac_teste:.4f}\n"
            f"OOB Score: {oob_score:.4f}\n\n"
        )
        arquivo_txt.write(df_importances.to_string(index=False))
    print("[Reports created successfully]")