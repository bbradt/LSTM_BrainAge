import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import argparse
import json
import glob

parser = argparse.ArgumentParser("LSTM-BrainAge Plots")
parser.add_argument("--experiment", default="E0001_first_model")
parser.add_argument("--result-dir", default="figures")
parser.add_argument("--font-scale", default=2)
parser.add_argument(
    "--plots", default='["lineplot:step X loss;{}", "lineplot:step X correlation/mean;{}"]')
args = parser.parse_args()
plots = json.loads(args.plots)
outdir = args.result_dir
os.makedirs(outdir, exist_ok=True)
logdir = os.path.join("logs", args.experiment)
train_results = list(glob.glob(os.path.join(
    logdir, "**", "train.csv"), recursive=True))
train_dfs = [pd.read_csv(csv) for csv in train_results]
train_df = pd.concat(train_dfs).reset_index(drop=True)
valid_results = list(glob.glob(os.path.join(
    logdir, "**", "valid.csv"), recursive=True))
valid_dfs = [pd.read_csv(csv) for csv in valid_results]
valid_df = pd.concat(valid_dfs).reset_index(drop=True)

for plot_string in plots:
    plot_params, plot_kwargs = plot_string.split(";")
    plot_kwargs = json.loads(plot_kwargs)
    plot_type, plot_vars = plot_params.split(":")
    vars = plot_vars.split(" X ")
    h = None
    if len(vars) == 3:
        x, y, h = vars
    elif len(vars) == 2:
        x, y = vars
    else:
        raise(ValueError("Got variables %s - this type of plot isn't supported" % vars))
    sb.set(font_scale=args.font_scale)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    if plot_type.lower() == "lineplot":
        sb.lineplot(x=x, y=y, hue=h, ax=ax[0], data=train_df, **plot_kwargs)
        ax[0].set_title("Training")
        sb.lineplot(x=x, y=y, hue=h, ax=ax[1], data=valid_df, **plot_kwargs)
        ax[1].set_title("Validation")
        fname = os.path.join(outdir, plot_params.replace(
            " ", "_").replace("/", "-")+".png")
        plt.savefig(fname, bbox_inches="tight")
        print("Saved ", fname)
        plt.close("all")
