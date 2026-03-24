import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
import torch_geometric as tg
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IGExplainer:
    """
    Integrated Gradients explainer for GNN models.

    Handles graph construction and batch processing for attribution computation.
    """

    def __init__(self, model, edge_index, device=DEVICE, baseline=0.0):
        """
        Args:
            model: Trained GNN model
            edge_index: Graph edge indices
            device: Device to run on ('cuda' or 'cpu')
            baseline: Baseline value for IG computation. Default: 0.0
        """
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.device = device
        self.baseline = baseline
        self.model.eval()

        # Create forward wrapper for IG
        self.ig = IntegratedGradients(self._forward_func)

    def _forward_func(self, x):
        """Forward pass wrapper for IG."""
        batch_size = x.shape[0]
        data_list = [tg.data.Data(x=x[i].unsqueeze(1), edge_index=self.edge_index)
                     for i in range(batch_size)]
        graph_batch = tg.data.Batch.from_data_list(data_list).to(self.device)

        output = self.model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

        # Handle both output types
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output

        return logits

    def explain(self, dataset, target=1, feature_names=None, n_steps=50,
                internal_batch_size=None, show_progress=True):
        """
        Compute IG attributions for a dataset.

        Args:
            dataset: TensorDataset with (X, y, sample_ids)
            target: Target class for attribution. Default: 1
            feature_names: Column names for output DataFrame. Default: None
            n_steps: Number of integration steps. Default: 50
            internal_batch_size: Batch size for internal computations. Default: None
            show_progress: Show progress bar. Default: True

        Returns:
            pd.DataFrame: Attribution scores with shape (n_samples, n_features)
                         Index: sample IDs, Columns: feature names
        """
        X = dataset.tensors[0]
        sample_ids = dataset.tensors[2].detach().cpu().numpy()

        attributions = []
        iterator = tqdm(X, total=len(X)) if show_progress else X

        for x_input in iterator:
            x_input = x_input.unsqueeze(0).to(self.device)
            x_input.requires_grad = True

            # Compute baseline
            baseline = torch.full_like(x_input, self.baseline)

            # Compute attributions
            attr = self.ig.attribute(
                x_input,
                baselines=baseline,
                target=target,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )

            attributions.append(attr.detach().cpu().numpy())

        # Stack results
        attributions = np.vstack(attributions)

        # Create DataFrame
        if feature_names is not None:
            return pd.DataFrame(attributions, index=sample_ids, columns=feature_names)
        else:
            return pd.DataFrame(attributions, index=sample_ids)
