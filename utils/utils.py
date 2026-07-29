import csv
import fcntl
from io import StringIO
import json
import logging
import os
import pandas as pd
from pathlib import Path
import subprocess
import sys

### hardcoded stuff ###

def get_secret(var_name, file_path=Path.home() / "store/personal.env"):
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith(f"{var_name}="):
                    return line.strip().split("=", 1)[1].strip('"\'')
    except FileNotFoundError:
        pass
    return None

def get_checkpoints_dir(experiment_key, dataset_name, artifacts_dir, wandb_project="RS_MAE-Thesis", json_path="all_runs.json"):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, "r") as f:
        runs_data = json.load(f)
    
    if dataset_name == "c2seg_wuhan": dataset_name = "c2seg_beijing"
    elif dataset_name == "c2seg_berlin": dataset_name = "c2seg_augsburg"
    run_id = runs_data.get(experiment_key, {}).get(dataset_name, {}).get("run_id")
    if not run_id:
        raise ValueError(f"Run ID not found for experiment_key='{experiment_key}', dataset_name='{dataset_name}' in {json_path}")
    
    ckpt_dir = Path(artifacts_dir) / wandb_project / run_id / "checkpoints"
   
    has_ckpts = any(ckpt_dir.glob("seg-*.ckpt")) or any(ckpt_dir.glob("mae-*.ckpt"))
    if not has_ckpts:
        logging.info(f"Checkpoint not found in {ckpt_dir}. Attempting to rsync from cluster...")
        
        username = get_secret("USERNAME") or os.environ.get("USER", "someone")
        remote_path = f"{username}@gateway.hpc.tu-berlin.de:/scratch/{username}/outputs/training_out/{wandb_project}/{run_id}"
        local_dest = Path(artifacts_dir) / wandb_project
        local_dest.mkdir(parents=True, exist_ok=True)
        try:
            # Clear LD_LIBRARY_PATH to avoid OpenSSL version mismatch in Conda/Miniforge env
            env = os.environ.copy()
            if "LD_LIBRARY_PATH" in env:
                del env["LD_LIBRARY_PATH"]
            subprocess.run(f'rsync -avP --exclude last.ckpt {remote_path} {str(local_dest)}', 
                           shell=True, executable="/bin/bash", check=True, env=env)
            logging.info("rsync completed successfully.")

            has_ckpts_after_rsync = any(ckpt_dir.glob("seg-*.ckpt")) or any(ckpt_dir.glob("mae-*.ckpt"))
            if not has_ckpts_after_rsync:
                raise FileNotFoundError(f"Could not find any valid checkpoints in {ckpt_dir} after rsync.")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Could not retrieve checkpoints from cluster for run_id {run_id}. Error: {e}")
        
    return ckpt_dir

def find_best_ckpt(directory):
    if not directory.exists():
        return None
    
    ckpts = list(directory.glob("seg-*.ckpt")) + list(directory.glob("mae-*.ckpt"))
    if not ckpts:
        return None
    
    best_ckpt, best_val = None, None
    
    for ckpt in ckpts:
        val = float(ckpt.stem.split("=")[-1]) # numerical value after last '='

        is_iou = "val_iou" in ckpt.name
        if best_val is None:
            best_val = val
            best_ckpt = ckpt
        elif is_iou and val > best_val:
            best_val = val
            best_ckpt = ckpt
        elif not is_iou and val < best_val:
            best_val = val
            best_ckpt = ckpt
            
    return best_ckpt

def log_run_to_json(experiment_name, seed, dataset_name, run_name, run_id, json_path="all_runs.json"):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            runs_data = {}
            if content:
                try:
                    runs_data = json.loads(content)
                except json.JSONDecodeError:
                    pass
                    
            exp_key = f"{experiment_name}-{seed}"
            if exp_key not in runs_data:
                runs_data[exp_key] = {}
                
            if dataset_name in runs_data[exp_key]:
                existing_run_name = runs_data[exp_key][dataset_name].get("run_name")
                existing_run_id = runs_data[exp_key][dataset_name].get("run_id")
                if existing_run_id != run_id:
                    logging.error(f"Run for experiment {exp_key} and dataset {dataset_name} already exists in {json_path} (run_name={existing_run_name}, run_id={existing_run_id}). Overwriting with run_name={run_name}, run_id={run_id}.")
                    
            runs_data[exp_key][dataset_name] = {
                "run_name": run_name,
                "run_id": run_id
            }
            
            f.seek(0)
            f.truncate()
            json.dump(runs_data, f, indent=4)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def log_results(results, dataset_name, experiment_name, csv_path="results.csv", group=None):
    """results: dict like {"acc":0.59, "iou":0.38, "f1":0.53, "loss":...}"""
    experiment_col = experiment_name.replace("_", " ").title().strip()

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            if content:
                df_raw = pd.read_csv(StringIO(content), header=None, dtype=str, keep_default_na=False)
                top_row = df_raw.iloc[0].tolist()
                columns = [str(c).strip() for c in df_raw.iloc[1].tolist()]
                df = df_raw.iloc[2:].copy()
                df.columns = columns
            else:
                top_row = ["", ""]
                columns = ["Dataset", "Metric"]
                df = pd.DataFrame(columns=columns)

            # normalize
            df = df.astype(object)
            df["Dataset"] = df["Dataset"].astype(str).str.strip().replace(r"^\s*$", pd.NA, regex=True).ffill()
            df["Metric"] = df["Metric"].astype(str).str.strip()

            full_groups = []
            current_g = ""
            for g in top_row:
                g_str = str(g).strip()
                if g_str != "":
                    current_g = g_str
                full_groups.append(current_g)

            # ensure column exists
            if experiment_col not in df.columns:
                df[experiment_col] = None
                
                if group:
                    group = str(group).lower().strip()
                    if group in full_groups:
                        last_idx = len(full_groups) - 1 - full_groups[::-1].index(group)
                        insert_idx = last_idx + 1
                        columns.insert(insert_idx, experiment_col)
                        full_groups.insert(insert_idx, group)
                    else:
                        columns.append(experiment_col)
                        full_groups.append(group)
                else:
                    columns.append(experiment_col)
                    full_groups.append("")
                
                df = df[columns]

            # insert / update
            for metric_name, val in results.items():
                if metric_name == "loss":
                    continue

                metric_name = metric_name.strip()
                mask = (df["Dataset"] == dataset_name) & (df["Metric"] == metric_name)

                if mask.any():
                    df.loc[mask, experiment_col] = f"{val*100:.2f}"
                else:
                    new_row = {
                        "Dataset": dataset_name,
                        "Metric": metric_name,
                        experiment_col: f"{val*100:.2f}"
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # keep table structured
            metric_order = ["OA", "mIoU", "mF1"]
            df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
            df = df.sort_values(by=["Dataset", "Metric"]).reset_index(drop=True)

            # blank repeated dataset names
            df_out = df.copy()
            df_out["Dataset"] = df_out["Dataset"].where(
                df_out["Dataset"] != df_out["Dataset"].shift(), "")

            # rebuild top_row to ensure exact formatting
            final_top_row = []
            seen_groups = set()
            for g in full_groups:
                if g != "" and g not in seen_groups:
                    final_top_row.append(g)
                    seen_groups.add(g)
                else:
                    final_top_row.append("")

            f.seek(0)
            f.truncate()
            
            writer = csv.writer(f)
            writer.writerow(final_top_row)
            writer.writerow(columns)
            
            # replace NaN with empty string
            df_out = df_out.fillna("")
            for _, row in df_out.iterrows():
                writer.writerow(row.tolist())
                
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def log_per_class_results(results, experiment_name, csv_path="benchmark_per_class_results.csv", group=None):
    """results: dict mapping class names to F1 values"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            if content:
                df_raw = pd.read_csv(StringIO(content), header=None, dtype=str, keep_default_na=False)
                top_row = df_raw.iloc[0].tolist()
                columns = [str(c).strip() for c in df_raw.iloc[1].tolist()]
                df = df_raw.iloc[2:].copy()
                df.columns = columns
            else:
                top_row = [""]
                columns = ["Class"]
                df = pd.DataFrame(columns=columns)

            df = df.astype(object)
            df["Class"] = df["Class"].astype(str).str.strip()
            
            full_groups = []
            current_g = ""
            for g in top_row:
                g_str = str(g).strip()
                if g_str != "":
                    current_g = g_str
                full_groups.append(current_g)

            experiment_col = experiment_name.replace("_", " ").title().strip()

            if experiment_col not in df.columns:
                df[experiment_col] = None
                
                if group:
                    group = str(group).lower().strip()
                    if group in full_groups:
                        last_idx = len(full_groups) - 1 - full_groups[::-1].index(group)
                        insert_idx = last_idx + 1
                        columns.insert(insert_idx, experiment_col)
                        full_groups.insert(insert_idx, group)
                    else:
                        columns.append(experiment_col)
                        full_groups.append(group)
                else:
                    columns.append(experiment_col)
                    full_groups.append("")
                
                df = df[columns]

            for class_name, val in results.items():
                class_name = str(class_name).strip()
                mask = df["Class"] == class_name

                if mask.any():
                    df.loc[mask, experiment_col] = f"{val*100:.2f}"
                else:
                    new_row = {
                        "Class": class_name,
                        experiment_col: f"{val*100:.2f}"
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            final_top_row = []
            seen_groups = set()
            for g in full_groups:
                if g != "" and g not in seen_groups:
                    final_top_row.append(g)
                    seen_groups.add(g)
                else:
                    final_top_row.append("")

            f.seek(0)
            f.truncate()
            
            writer = csv.writer(f)
            writer.writerow(final_top_row)
            writer.writerow(columns)
            
            df = df.fillna("")
            for _, row in df.iterrows():
                writer.writerow(row.tolist())
                
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def setup_logging(level="info"):
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[37m",
            logging.INFO: "",
            logging.WARNING: "\033[93m",
            logging.ERROR: "\033[31m",
        }
        RESET = "\033[0m"

        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt=fmt, datefmt=datefmt)

        def format(self, record):
            msg = super().format(record)
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"


    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter('[%(asctime)s] %(funcName)s(): %(message)s', datefmt='%H:%M:%S'))

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(handler)
    
    if level == "info":
        logger.setLevel(logging.INFO)
    elif level == "debug":
        logger.setLevel(logging.DEBUG)
