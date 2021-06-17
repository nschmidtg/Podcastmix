from torch.nn.modules.loss import _Loss
import torch

class L2Time(_Loss):
    
    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(
                f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
            )
        _, number_of_sources, length_of_sources = est_targets.shape
        squared_abs_dif = torch.abs((est_targets - targets) ** 2)
        sum_of_squared_abs_dif = torch.sum(squared_abs_dif)
        loss = 1 / (number_of_sources * length_of_sources) * sum_of_squared_abs_dif
        # avg accros batch
        loss = loss.mean(dim=0)
        return loss
