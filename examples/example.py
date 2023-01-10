import os
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from catalyst import dl, utils
from dataloaders.fake_fnc import FakeFNC
from models.bilstm import BiLSTM
from sklearn.model_selection import train_test_split
import numpy as np
import torch


criterion = nn.MSELoss()
full_dataset = FakeFNC(N_hc=4096, N_sz=4096)
model = BiLSTM(seqlen=full_dataset.seqlen, dim=full_dataset.dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
full_idx = np.arange(len(full_dataset))
train_idx, test_idx = train_test_split(full_idx, test_size=0.1)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

loaders = {
    "train": DataLoader(train_dataset, batch_size=32),
    "valid": DataLoader(test_dataset, batch_size=len(test_dataset)),
}

runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=10,
    callbacks=[
        dl.CriterionCallback(metric_key="loss",
                             input_key="logits",
                             target_key="targets"),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
checkpoint = utils.load_checkpoint("logs/checkpoints/best_full.pth")
utils.unpack_checkpoint(
    checkpoint=checkpoint,
    model=model,
    optimizer=optimizer,
    criterion=criterion
)
for data, label in test_loader:
    prediction = model(data)
    stacked_labels = torch.stack([prediction, label], 0).squeeze()
    corr = torch.corrcoef(stacked_labels)
    print("Correlation with Ground Truth from Bets Model ", corr)
