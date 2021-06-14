from torch.nn.modules.loss import _Loss
import torch

class LogL2Time(_Loss):
    r"""Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        - est_targets: :math:`(batch, ...)`.
        - targets: :math:`(batch, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)`

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # singlesrc_mse / multisrc_mse support both 'pw_pt' and 'perm_avg'.
        >>> loss_func = PITLossWrapper(singlesrc_mse, pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)
    """

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