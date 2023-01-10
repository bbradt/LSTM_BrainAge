import os
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from catalyst import dl, utils
from dataloaders.get_dataset import get_dataset
from models.get_model import get_model
from models.bilstm import BiLSTM
from sklearn.model_selection import KFold
import numpy as np
import torch
import argparse
import json
import inspect
import pandas as pd
from callbacks.CorrelationCallback import CorrelationCallback
from default_args import DEFAULTS, HELP
#from inference_args import DEFAULTS, HELP
# Begin Argument Parsing
parser = argparse.ArgumentParser("LSTM for BrainAge")
for key, val in DEFAULTS.items():
    parser.add_argument("--%s" % key, default=val,
                        type=type(val), help=HELP[key])
args = parser.parse_args()
os.makedirs(args.logdir, exist_ok=True)
json.dump(args.__dict__, open(os.path.join(
    args.logdir, "parameters.json"), "w"))
# Set seed before ANYTHING
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Resolve Pytorch SubModules
args.optimizer = getattr(torch.optim, args.optimizer)
args.criterion = getattr(torch.nn, args.criterion)
# Resolve dataset and model
full_train_dataset = None
if args.train_dataset is not None:
    full_train_dataset = get_dataset(args.train_dataset,
                                     *json.loads(args.train_dataset_args),
                                     **json.loads(args.train_dataset_kwargs))
full_inference_dataset = None
if args.test_dataset is not None:
    if args.test_dataset.lower() != "valid":
        full_inference_dataset = get_dataset(args.test_dataset,
                                             *json.loads(args.test_dataset_args),
                                             **json.loads(args.test_dataset_kwargs))
model = get_model(args.model, *json.loads(args.model_args),
                  **json.loads(args.model_kwargs))
# Resolve Metrics
args.train_metrics = json.loads(args.train_metrics)
args.test_metrics = json.loads(args.test_metrics)
# END ARGUMENT PARSING

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

# initialize criterion and optimizer
criterion = args.criterion()

if full_train_dataset is not None:
    sig = inspect.signature(args.optimizer)
    optim_kwargs = {k: v for k, v in json.loads(
        args.optim_kwargs).items() if k in sig.parameters.keys()}
    optimizer = args.optimizer(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **optim_kwargs)
    full_idx = np.arange(len(full_train_dataset))
    kfold = KFold(n_splits=args.num_folds,
                  shuffle=True, random_state=args.seed)
    for k, (train_idx, valid_idx) in enumerate(kfold.split(full_idx)):
        if k == args.k:
            break
    train_dataset = Subset(full_train_dataset, train_idx)
    valid_dataset = Subset(full_train_dataset, valid_idx)
    if args.test_dataset.lower() == "valid":
        full_inference_dataset = valid_dataset
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size),
        "valid": DataLoader(valid_dataset, batch_size=args.batch_size),
    }

    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    # add callbacks
    callbacks = []
    for metric in args.train_metrics:
        if metric == "loss":
            callbacks.append(dl.CriterionCallback(metric_key="loss",
                                                  input_key="logits",
                                                  target_key="targets"))
        elif metric == "correlation":
            callbacks.append(
                CorrelationCallback(input_key="logits", target_key="targets")
            )
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=args.epochs,
        callbacks=callbacks,
        logdir=args.logdir,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True
    )
if full_inference_dataset is not None:
    test_loader = DataLoader(full_inference_dataset,
                             batch_size=args.batch_size)
    checkpoint = utils.load_checkpoint(args.inference_model)
    model.load_state_dict(checkpoint)
    # add callbacks
    callbacks = []
    for metric in args.test_metrics:
        if metric == "correlation":
            callbacks.append(
                CorrelationCallback(input_key="logits", target_key="targets")
            )
    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    runner.evaluate_loader(
        loader=test_loader,
        callbacks=callbacks,
        model=model
    )
    metrics = [runner.experiment_metrics[1]['valid']]
    df = pd.DataFrame(metrics)

    all_predictions = []
    for prediction in runner.predict_loader(loader=test_loader):
        all_predictions.append(prediction["logits"].detach().cpu().numpy())
    all_predictions = np.concatenate(all_predictions, 0)
    savedir = os.path.join(args.logdir, "predictions")
    os.makedirs(savedir, exist_ok=True)
    df.to_csv(os.path.join(savedir, "test.csv"), index=False)

    np.savetxt(os.path.join(savedir, "predictions.txt"),
               all_predictions, fmt="%f")
