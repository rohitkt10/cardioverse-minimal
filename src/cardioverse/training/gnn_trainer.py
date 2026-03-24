import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric as tg
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from cardioverse.utils.regularization import lp_regularizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GNNTrainer:
    """Unified trainer for GCNModel and LiNetModel."""

    def __init__(self, model, optimizer, edge_index, lossfn=F.nll_loss, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.device = device
        self.edge_index = edge_index.to(device)
        self.best_val_auroc = -np.inf
        self.best_model_state = self.model.state_dict()
        self.best_epoch = 0

    def train_step(self, batch, config, weights=None):
        """Process 1 training batch."""
        self.model.train()
        x, y = batch[0].to(self.device), batch[1].to(self.device)

        # Create graph batch
        data_list = [tg.data.Data(x=x[i].unsqueeze(1), y=y[i].unsqueeze(0), edge_index=self.edge_index)
                     for i in range(len(x))]
        graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)

        self.optimizer.zero_grad()

        # Forward pass - handle both output types
        output = self.model(graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)

        if isinstance(output, tuple):
            y_logits, kl_loss = output
            has_kl = True
        else:
            y_logits = output
            has_kl = False

        # Compute loss
        data_loss = self.lossfn(F.log_softmax(y_logits, dim=1), y, weight=weights) if weights is not None \
                    else self.lossfn(F.log_softmax(y_logits, dim=1), y)
        l1_reg = config.lmbda_l1 * lp_regularizer(self.model, p=1)
        l2_reg = config.lmbda_l2 * lp_regularizer(self.model, p=2)

        total_loss = data_loss + l1_reg + l2_reg
        if has_kl:
            total_loss += config.lmbda_kl * kl_loss

        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        loss = data_loss.detach().cpu().item()
        y_true = y.detach().cpu().numpy()
        y_prob = F.softmax(y_logits, dim=1).detach().cpu().numpy()
        return loss, y_true, y_prob

    def train_epoch(self, loader, config, weights=None):
        """1 training epoch."""
        epoch_loss, all_y_true, all_y_pred, all_y_prob = [], [], [], []
        for batch in loader:
            loss, y_true, y_prob = self.train_step(batch, config, weights)
            y_pred = y_prob.argmax(axis=1)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob[:, 1])
            epoch_loss.append(loss)

        acc = balanced_accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred, average='macro')
        auroc = roc_auc_score(all_y_true, all_y_prob)
        loss = np.mean(epoch_loss)
        return {'loss': loss, 'acc': acc, 'f1': f1, 'auroc': auroc}

    def val_step(self, batch):
        """Process 1 validation batch."""
        self.model.eval()
        x, y = batch[0].to(self.device), batch[1].to(self.device)

        # Create graph batch
        data_list = [tg.data.Data(x=x[i].unsqueeze(1), y=y[i].unsqueeze(0), edge_index=self.edge_index)
                     for i in range(len(x))]
        graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)

        with torch.no_grad():
            output = self.model(graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)

            if isinstance(output, tuple):
                y_logits, _ = output
            else:
                y_logits = output

            data_loss = self.lossfn(F.log_softmax(y_logits, dim=1), y)

        loss = data_loss.detach().cpu().item()
        y_true = y.detach().cpu().numpy()
        y_prob = F.softmax(y_logits, dim=1).detach().cpu().numpy()
        return loss, y_true, y_prob

    def val_epoch(self, loader):
        """1 validation epoch."""
        epoch_loss, all_y_true, all_y_pred, all_y_prob = [], [], [], []
        for batch in loader:
            loss, y_true, y_prob = self.val_step(batch)
            y_pred = y_prob.argmax(axis=1)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob[:, 1])
            epoch_loss.append(loss)

        acc = balanced_accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred, average='macro')
        auroc = roc_auc_score(all_y_true, all_y_prob)
        loss = np.mean(epoch_loss)
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1, 'val_auroc': auroc}

    def fit(self, train_dataset, val_dataset, config):
        """Main training loop."""
        # Setup dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Class weights for imbalanced data
        counts = np.bincount(train_dataset.tensors[-2].detach().cpu().numpy())
        weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(self.device)

        # Training loop
        history = {}
        for epoch in range(1, 1 + config.nepochs):
            # Train epoch
            res = self.train_epoch(loader=train_loader, config=config, weights=weights)
            for k, v in res.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            # Validation epoch
            res = self.val_epoch(loader=val_loader)
            for k, v in res.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            # Checkpointing
            val_auroc = history["val_auroc"][-1]
            if epoch >= config.checkpoint_after and val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.best_model_state = self.model.state_dict()
                self.best_epoch = epoch

            # Logging
            if epoch % config.logstep == 0 or epoch == 1:
                log_str = f"[ Epoch {epoch} ] : "
                metric_keys = ['loss', 'val_loss', 'acc', 'val_acc', 'f1', 'val_f1', 'auroc', 'val_auroc']
                log_str += "; ".join(f"{k}: {history[k][-1]:.3f}" for k in metric_keys if k in history)
                print(log_str)

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        return history

    def predict(self, x):
        """Return logits."""
        self.model.eval()
        x = x.to(self.device)

        # Create graph batch
        data_list = [tg.data.Data(x=x[i].unsqueeze(1), edge_index=self.edge_index)
                     for i in range(len(x))]
        graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)

        with torch.no_grad():
            output = self.model(graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)

            if isinstance(output, tuple):
                y_logits, _ = output
            else:
                y_logits = output

        return y_logits
