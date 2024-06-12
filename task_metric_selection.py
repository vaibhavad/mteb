import os
import json
import argparse

from multiprocessing import Pool
import matplotlib.pyplot as plt

from tqdm import tqdm

from functools import reduce
from yaml import safe_load
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.repocard import metadata_load

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# supress warnings
import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import HfApi

import pandas as pd

from scipy.stats import spearmanr
from scipy.stats import pearsonr

import xgboost as xgb

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_INFOS = {}

def spearman(x, y):
    return spearmanr(x, y)[0]

def pearson(x, y):
    return pearsonr(x, y)[0]

def get_leaderboard_df():
    download_dir = snapshot_download(
        repo_id="mteb/leaderboard",
        repo_type="space",
    )

    MODEL_META_PATH = os.path.join(download_dir, "model_meta.yaml")
    with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
        MODEL_META = safe_load(f)

    LEADERBOARD_CONFIG_PATH = "config.yaml"
    with open(os.path.join(download_dir, LEADERBOARD_CONFIG_PATH), 'r', encoding='utf-8') as f:
        LEADERBOARD_CONFIG = safe_load(f)

    with open(os.path.join(download_dir, "EXTERNAL_MODEL_RESULTS.json")) as f:
        EXTERNAL_MODEL_RESULTS = json.load(f)

    TASKS_CONFIG = LEADERBOARD_CONFIG["tasks"]
    BOARDS_CONFIG = LEADERBOARD_CONFIG["boards"]


    TASKS = list(TASKS_CONFIG.keys())

    TASK_TO_METRIC = {k:v["metric"] for k,v in TASKS_CONFIG.items()}

    TASK_DESCRIPTIONS = {k: v["task_description"] for k,v in TASKS_CONFIG.items()}
    TASK_DESCRIPTIONS["Overall"] = "Overall performance across MTEB tasks."
    MODELS_TO_SKIP = MODEL_META["models_to_skip"]

    TASK_TO_TASK_TYPE = {task_category: [] for task_category in TASKS}
    for board_config in BOARDS_CONFIG.values():
        for task_category, task_list in board_config["tasks"].items():
            TASK_TO_TASK_TYPE[task_category].extend(task_list)

    # model_list = os.listdir(os.path.join(download_dir, "results"))

    task_dict = BOARDS_CONFIG["en"]["tasks"]
    all_tasks = reduce(lambda x, y: x + y, task_dict.values())

    tasks = list(task_dict.keys())
    datasets = all_tasks

    # print(model_list)
    api = HfApi(token=HF_TOKEN)
    models = api.list_models(filter="mteb")

    i = 0
    df_list = []
    for model in EXTERNAL_MODEL_RESULTS:
        results_list = []
        for task in tasks:
            # Not all models have InstructionRetrieval, other new tasks
            if task not in EXTERNAL_MODEL_RESULTS[model]:
                continue
            results_list += EXTERNAL_MODEL_RESULTS[model][task][TASK_TO_METRIC[task]]
    
        res = {k: v for d in results_list for k, v in d.items() if (k == "Model") or any([x in k for x in datasets])}

        # <a target="_blank" style="text-decoration: underline" href="https://huggingface.co/in...al-7b-instruct">e5-mistral-7b-instruct</a>
        # extract the model name e5-mistral-7b-instruct from the value
        res["Model"] = res["Model"].split('">')[1].split("</a>")[0]
        if len(res) == len(datasets) + 1 and res.keys() == set(datasets) | {"Model"}:
            df_list.append(res)

    for model in models:
        if model.modelId in MODELS_TO_SKIP: continue
        i += 1
        print("MODEL", model.modelId)
        if model.modelId not in MODEL_INFOS:
            readme_path = hf_hub_download(model.modelId, filename="README.md")
            meta = metadata_load(readme_path)
            MODEL_INFOS[model.modelId] = {
                "metadata": meta
            }
        meta = MODEL_INFOS[model.modelId]["metadata"]
        if "model-index" not in meta:
            continue
        if len(datasets) > 0:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and any([x in sub_res.get("dataset", {}).get("name", "") for x in datasets])]
        else:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks)]
        out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if score["type"] == TASK_TO_METRIC.get(res["task"]["type"])][0]} for res in task_results]
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = model.modelId
        # only keep the models that have results for all tasks
        if len(out) == len(datasets) + 1 and out.keys() == set(datasets) | {"Model"}:
            df_list.append(out)
    
    df = pd.DataFrame(df_list)
    # add Overall column with average of all tasks, excluding the Model column
    df["Overall"] = df.iloc[:, 1:].mean(axis=1)
    # sort by Overall
    df = df.sort_values("Overall", ascending=False)
    # make Model column first, Overall column second
    cols = df.columns.tolist()
    cols = ["Model", "Overall"] + [col for col in cols if col not in ["Model", "Overall"]]
    df = df[cols]

    df.to_csv("results.csv", index=False)

def _fit_predict(model_i, task, task_df, classifer):
    clf = classifer()
    X_train = task_df.drop([task], axis=1).drop(model_i)
    y_train = task_df[[task]].drop(model_i)
    clf.fit(X_train.values, y_train.values)
    X_test = task_df.drop(columns=[task]).iloc[model_i]
    y_pred = clf.predict(X_test.values.reshape(1, -1))
    return float(y_pred)

def leave_one_task_out(df: pd.DataFrame, classifer, num_cpus) -> pd.DataFrame:
    """Predicts the performance of a model on a task by training on all other tasks.
    
    Args:
        df: a DataFrame with columns: Model, Overall, and one column for each task.
        classifer: a scikit-learn model that has a fit and predict method. 

    Returns:
        a matrix of predictions for each model and task.
    """
    predictions = pd.DataFrame(columns=df.columns)
    # for task in df.columns:
    task_df = df.drop(["Model", "Overall"], axis=1)
    columns_tqdm = tqdm(task_df.columns)
    for task in columns_tqdm:
        columns_tqdm.set_description(f"Task: {task}")

        with Pool(num_cpus) as p:
            task_predictions = p.starmap(_fit_predict, [(model_i, task, task_df, classifer) for model_i in range(len(df))])
        predictions[task] = list(task_predictions)

    # add the model names and overall scores    
    predictions["Model"] = df["Model"]
    return predictions

def calculate_task_scores(observed_results: pd.DataFrame, predictions: pd.DataFrame, metric) -> pd.DataFrame:
    """Calculate how well the predictions match the observed results.

    Args:
        observed_results: a DataFrame with columns: Model, Overall, and one column for each task.
        predictions: a DataFrame with columns: Model, Overall, and one column for each task.
        metric: a function that takes two lists of numbers and returns a single number.

    Returns:
        a DataFrame with a single row that contains the score for each task.
    """
    scores = {}
    for task in predictions.columns:
        if task not in ["Model", "Overall"]:
            scores[task] = metric(predictions[task], observed_results[task])
    return pd.DataFrame(scores, index=[0])

def main(args):

    metric_map = {
        "spearman": spearman,
        "pearson": pearson,
        "mse": mean_squared_error
    }

    ascending = True if args.metric == "mse" else False 

    if not os.path.exists("results.csv"):
        get_leaderboard_df()
    df = pd.read_csv("results.csv")
    df_original = df.copy()

    model = xgb.XGBRegressor if args.model == "xgboost" else LinearRegression

    removed_tasks = []
    correlations = []
    task_scores = []

    print(f"Using Model {args.model} and Metric {args.metric}")

    for i in range(args.num_iterations):
        predictions = leave_one_task_out(df, model, args.num_cpus)
        predictions["Overall"] = predictions.drop(columns=["Model"]).mean(axis=1)
        results = calculate_task_scores(df, predictions, metric_map[args.metric])

        print(f"{args.metric} (ascending={ascending})")
        print(results.T.sort_values(by=0, ascending=ascending).head(3))

        # remove the top task from the results
        task = results.T.sort_values(by=0, ascending=ascending).head(1).index[0]
        task_score = results[task].values[0]
        task_scores.append(task_score)
        print()
        print(f"Removing task: {task}")
        df = df.drop(columns=[task])
        removed_tasks.append(task)

        print("Correlation with original rankings")
        df["Overall"] = df.drop(columns=["Model", "Overall"]).mean(axis=1)
        correlation = spearman(df["Overall"], df_original["Overall"])
        correlations.append(correlation)
        print(spearman(df["Overall"], df_original["Overall"]))
        print()

    print()
    print("Removed tasks:")
    for task in removed_tasks:
        print(task)
    
    # plot the correlations
    plt.plot(correlations, label="Correlation using remaining tasks")
    plt.plot(task_scores, label="Prediction correlation of excluded task")
    plt.legend()
    plt.title("Task Selection")
    plt.savefig("correlations.png")
    
    df.to_csv("final_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="linear", help="Model to use for training.", choices=["xgboost", "linear"])
    parser.add_argument("--metric", type=str, default="spearman", help="Metric to use for evaluation.", choices=["spearman", "pearson", "mse"])
    parser.add_argument("--num_cpus", type=int, default=32, help="Number of CPUs to use for training.")
    parser.add_argument("--num_iterations", type=int, default=50, help="Number of iterations to run.")
    args = parser.parse_args()
    main(args)
