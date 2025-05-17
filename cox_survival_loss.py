import torch

def survival_cox_sigmoid_loss(outputs, time, event, weights=None):
    """
    Cox partial likelihood loss with sigmoid applied to model output.
    
    Args:
        outputs: Tensor of shape (N,) - model predictions N_theta(y_i, t)
        time: Tensor of shape (N,) - survival times T_i
        event: Tensor of shape (N,) - event indicators E_i (1 if event observed, else 0)
        weights: Tensor of shape (N,), optional clinical weights S_i (default all ones)
        
    Returns:
        loss: scalar tensor (negative log partial likelihood)
    """

    device = outputs.device
    N = outputs.shape[0]

    if weights is None:
        weights = torch.ones_like(outputs, device=device)

    # Apply sigmoid to outputs: g_hat_beta(x_i, t) = sigmoid(N_theta(y_i, t))
    pred = torch.sigmoid(outputs)  # shape (N,)

    # Sort by descending time to construct risk sets
    sorted_time, sorted_idx = torch.sort(time, descending=True)
    sorted_event = event[sorted_idx]
    sorted_pred = pred[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Compute cumulative sums of weighted predictions for denominator
    cum_sum_pred = torch.cumsum(sorted_weights * sorted_pred, dim=0)  # shape (N,)

    # Gather numerator terms only for events (E_i = 1)
    event_idx = sorted_event == 1
    if event_idx.sum() == 0:
        # No events -> loss = 0
        return torch.tensor(0., device=device)

    numerator = torch.log(sorted_weights[event_idx] * sorted_pred[event_idx] + 1e-10)
    denominator = torch.log(cum_sum_pred[event_idx] + 1e-10)

    loss = -torch.mean(numerator - denominator)

    return loss
