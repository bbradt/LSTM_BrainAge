DEFAULTS = {
    "devices": "0",
    # passed to getattr from torch.nn
    "criterion": "MSELoss",
    # key passed to get_dataset
    "train-dataset": "ukbhcp1200",
    # args passed to get_dataset
    "train-dataset-args": "[]",
    # kwargs passed to get_dataset
    "train-dataset-kwargs": '{"N_subs": 10000, "sequential": true}',
    # key passed to get dataset
    "test-dataset": "valid",
    # args passed to get dataset
    "test-dataset-args": '{}',
    # kwargs passed to get dataset
    "test-dataset-kwargs": '[]',
    # key passed to get_model
    "model": "simpleseqrnn",
    # args passed to get_model
    "model-args": '[448,1378]',
    # kwargs passed to get_model
    "model-kwargs": '{}',
    "inference-model": "<EVAL>os.path.join(args.logdir,'checkpoints','best_full.pth')",
    # passed to getattr from torch.optim
    "optimizer": "Adam",
    "optim-kwargs": '{"rho": 1e-6, "betas": [0.9, 0.9999]}',
    "batch-size": 64,
    "lr": 1e-3,
    "weight-decay": 0,
    "num-folds": 10,
    "epochs": 100,
    "train-metrics": '["loss"]',     # json parsed
    "test-metrics": '["loss"]',
    "seed": 314159,
    "logdir": "logs/test",
    "k": 0,
}

HELP = {
    "devices": "Environment variable defining available GPU devices",
    "criterion": "Loss function for training/validation - must be a module name in torch.nn",
    "train-dataset": "Dataset to use for training <fakefnc/ukb/ukbhcp1200>",
    "train-dataset-args": "Arguments to pass to training dataset (JSON parsed list)",
    "train-dataset-kwargs": "Keyword Arguments to pass to training dataset (JSON parsed dict)",
    "test-dataset": "Dataset to use for inference <fakefnc/ukb/ukbhcp1200/valid> - 'valid' uses the validation data",
    "test-dataset-args": "Arguments to pass to inference dataset (ignored if validation)",
    "test-dataset-kwargs": "Keyword Arguments to pass to inference dataset (ignored if validation)",
    "model": "Model to use <bilstm/bilstm-classifier/bilstm-hierarchical>",
    "model-args": "Arguments to pass to model (JSON parsed list)",
    "model-kwargs": "Keyword arguments to pass to model (JSON parsed dict)",
    "inference-model": "Model name to load for inference (looks in logdir/checkpoints)",
    "optimizer": "Optimizer to use for training (must be found in torch.optim)",
    "optim-kwargs": "Keyword arguments for Optimizer",
    "batch-size": "Batch size for both training and inference",
    "lr": "Learning rate for training (float)",
    "weight-decay": "Weight decay for training (float)",
    "num-folds": "Number of Cross-Validation Folds",
    "k": "The current fold to evaluate",
    "epochs": "The number of epochs to train",
    "train-metrics": "The metrics to compute for training (JSON parsed list) <loss/correlation/auc>",
    "test-metrics": "The metrics to compute for testing (JSON parsed list) <loss/correlation/auc>",
    "seed": "The seed to initialize ALL RANDOM STATES",
    "logdir": "The logging directory, where results are saved (created upon runtime)"
}
