from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import torch_geometric as tg
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from cardioverse.training.gnn_trainer import GNNTrainer
from cardioverse.utils.regularization import lp_regularizer


class GNNFusionTrainer(GNNTrainer):
    """
    Two-stage multiview training. Extends GNNTrainer with fusion-specific logic.

    Stage 1: Freeze GNNs, train fusion only
    Stage 2: Unfreeze all, train end-to-end
    """

    def __init__(self, model, optimizer, edge_indices: List, lossfn=F.nll_loss, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: GNNIntegrativeModel with pretrained GNNs
            optimizer: Optimizer for training
            edge_indices: List of edge_index tensors for each modality
            lossfn: Loss function. Default: F.nll_loss
            device: Device to run on. Default: 'cuda' if available else 'cpu'
        """
        super().__init__(model, optimizer, edge_indices[0], lossfn=lossfn, device=device)
        self.edge_indices = [ei.to(device) for ei in edge_indices]

    def freeze_feature_extractors(self):
        """Freeze all GNN parameters, only train fusion."""
        for extractor in self.model.models:
            for param in extractor.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def reset_optimizer(self, lr: float):
        """Recreate optimizer with new learning rate."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, batch, config, weights=None):
        """Process 1 training batch."""
        self.model.train()
        *xs, y, _ = batch
        xs = [x.to(self.device) for x in xs]
        y = y.to(self.device)

        # Create graph batches for each modality
        graphs = []
        for j, edge_index in enumerate(self.edge_indices):
            x = xs[j]
            data_list = [
                tg.data.Data(
                    x=x[i].unsqueeze(1),
                    y=y[i].unsqueeze(0),
                    edge_index=edge_index,
                )
                for i in range(len(x))
            ]
            graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)
            graphs.append(graph_batch)

        self.optimizer.zero_grad()
        y_logits, kl_losses = self.model(graphs)
        loss_kl = torch.stack(kl_losses).sum()

        data_loss = (
            self.lossfn(F.log_softmax(y_logits, dim=1), y, weight=weights)
            if weights is not None
            else self.lossfn(F.log_softmax(y_logits, dim=1), y)
        )
        l1_reg = config.lmbda_l1 * lp_regularizer(self.model, p=1)
        l2_reg = config.lmbda_l2 * lp_regularizer(self.model, p=2)
        total_loss = data_loss + l1_reg + l2_reg + config.lmbda_kl * loss_kl

        total_loss.backward()
        self.optimizer.step()

        loss = data_loss.detach().cpu().item()
        y_true = y.detach().cpu().numpy()
        y_prob = F.softmax(y_logits, dim=1).detach().cpu().numpy()
        return loss, y_true, y_prob

    def val_step(self, batch):
        """Process 1 validation batch."""
        self.model.eval()
        *xs, y, _ = batch
        xs = [x.to(self.device) for x in xs]
        y = y.to(self.device)

        # Create graph batches for each modality
        graphs = []
        for j, edge_index in enumerate(self.edge_indices):
            x = xs[j]
            data_list = [
                tg.data.Data(
                    x=x[i].unsqueeze(1),
                    y=y[i].unsqueeze(0),
                    edge_index=edge_index,
                )
                for i in range(len(x))
            ]
            graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)
            graphs.append(graph_batch)

        with torch.no_grad():
            y_logits, _ = self.model(graphs)
            data_loss = self.lossfn(F.log_softmax(y_logits, dim=1), y)

        loss = data_loss.detach().cpu().item()
        y_true = y.detach().cpu().numpy()
        y_prob = F.softmax(y_logits, dim=1).detach().cpu().numpy()
        return loss, y_true, y_prob

    def predict(self, *xs: Tuple[Tensor, ...]) -> Tensor:
        """
        Predict logits for given inputs.

        Args:
            *xs: Variable number of input tensors (one per modality)

        Returns:
            logits
        """
        self.model.eval()
        xs = [x.to(self.device) for x in xs]

        # Create graph batches for each modality
        graphs = []
        for j, edge_index in enumerate(self.edge_indices):
            x = xs[j]
            data_list = [
                tg.data.Data(x=x[i].unsqueeze(1), edge_index=edge_index)
                for i in range(len(x))
            ]
            graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)
            graphs.append(graph_batch)

        
        logits, _ = self.model(graphs)  
        return logits

    def fit(self, train_dataset, val_dataset, config):
        """
        Two-stage training:
        - Stage 1: Freeze GNNs, train fusion only (nepochs_stage1, lr_stage1)
        - Stage 2: Unfreeze all, train end-to-end (nepochs_stage2, lr_stage2)

        Args:
            train_dataset: Training dataset with (X1, X2, ..., y, sample_ids)
            val_dataset: Validation dataset with same structure
            config: FusionTrainingConfig

        Returns:
            {0: stage1_history, 1: stage2_history}
        """
        # Reset best model tracking
        self.best_val_auroc = -np.inf
        self.best_model_state = self.model.state_dict()
        self.best_epoch = 0

        # Stage 1: Freeze GNNs, train fusion only
        self.freeze_feature_extractors()
        stage1_config = deepcopy(config)
        stage1_config.nepochs = config.nepochs_stage1
        stage1_config.checkpoint_after = 0
        history1 = self._fit_stage(train_dataset, val_dataset, stage1_config)

        # Stage 2: Unfreeze all, train end-to-end
        self.unfreeze_all()
        self.reset_optimizer(lr=config.lr_stage2)
        stage2_config = deepcopy(config)
        stage2_config.nepochs = config.nepochs_stage2
        stage2_config.checkpoint_after = 0
        history2 = self._fit_stage(train_dataset, val_dataset, stage2_config)

        # Load best model from stage 2
        self.model.load_state_dict(self.best_model_state)
        return {0: history1, 1: history2}

    def _fit_stage(self, train_dataset, val_dataset, config):
        """Single stage training loop."""
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
            res = self._train_epoch(loader=train_loader, config=config, weights=weights)
            for k, v in res.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            # Validation epoch
            res = self._val_epoch(loader=val_loader)
            for k, v in res.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            # Checkpointing
            val_auroc = history["val_auroc"][-1]
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.best_model_state = self.model.state_dict()
                self.best_epoch = epoch

            # Logging
            if epoch % config.logstep == 0 or epoch == 1:
                log_str = f"[ Epoch {epoch} ] : "
                metric_keys = ['loss', 'val_loss', 'acc', 'val_acc', 'f1', 'val_f1', 'auroc', 'val_auroc']
                log_str += "; ".join(f"{k}: {history[k][-1]:.3f}" for k in metric_keys if k in history)
                print(log_str)

        return history

    def _train_epoch(self, loader, config, weights=None):
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

    def _val_epoch(self, loader):
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
