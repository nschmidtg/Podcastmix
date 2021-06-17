from torch.nn.modules.loss import _Loss
import torch

class LogL2Time(_Loss):
    
    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(
                f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
            )
        batch, number_of_sources, length_of_sources = est_targets.shape
        squared_abs_dif = torch.abs((est_targets - targets) ** 2)
        sum_of_squared_abs_dif = torch.sum(squared_abs_dif, dim=2)
        sum_of_log_of_previous = torch.sum(torch.log10(sum_of_squared_abs_dif), dim=1)
        loss = 10 / (number_of_sources * length_of_sources) * sum_of_log_of_previous
        loss = loss.mean(dim=0)
        return loss
