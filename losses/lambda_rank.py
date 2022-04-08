import torch

def bweight_lambda_mrr_loss(y_pred, y_true, batch_weight, eps=1e-10, padded_value_indicator=-1, reduction="mean", sigma=1.):
    """
    y_pred: FloatTensor [bz, topk]
    y_true: FloatTensor [bz, topk]
    batch_weight: FloatTensor [bz]
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    clamp_val = 1e8 if y_pred.dtype==torch.float32 else 1e4

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")
    assert torch.sum(padded_mask) == 0

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)
    weights = torch.abs(inv_pos_idxs.view(1,-1,1) - inv_pos_idxs.view(1,1,-1)) # [1, topk, topk]

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    losses = torch.log(1. + torch.exp(-scores_diffs)) * weights #[bz, topk, topk]

    if reduction == "sum":
        assert losses.dim() == 3
        losses = losses * batch_weight.view(-1,1,1)
        loss = torch.sum(losses[padded_pairs_mask])
    elif reduction == "mean":
        assert losses.dim() == 3
        losses = losses * batch_weight.view(-1,1,1)
        loss = torch.mean(losses[padded_pairs_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss

def lambda_mrr_loss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, reduction="mean", sigma=1.):
    """
    y_pred: FloatTensor [bz, topk]
    y_true: FloatTensor [bz, topk]
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    clamp_val = 1e8 if y_pred.dtype==torch.float32 else 1e4

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")
    #assert torch.sum(padded_mask) == 0

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)
    weights = torch.abs(inv_pos_idxs.view(1,-1,1) - inv_pos_idxs.view(1,1,-1)) # [1, topk, topk]

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    losses = torch.log(1. + torch.exp(-scores_diffs)) * weights #[bz, topk, topk]

    if reduction == "sum":
        loss = torch.sum(losses[padded_pairs_mask])
    elif reduction == "mean":
        loss = torch.mean(losses[padded_pairs_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


if __name__ == "__main__":
    #y_true = torch.FloatTensor([[1.,1./2, 1./3, 0.,0., -1/2., -1.]])
    #y_pred_1 = torch.FloatTensor([[2.3, 1.2, 1.1, 0.5, 0.23, 0.21, 40]])
    #y_pred_2 = torch.FloatTensor([[0.5, 0.23, 2.3, 1.2, 1.1, 5, 20]])

    #print(lambda_mrr_loss(y_pred_1, y_true))
    #print(lambda_mrr_loss(y_pred_2, y_true))

    y_true = torch.FloatTensor([[1., 1/2., 0., 0.], [1., 1/2., 0., 0.]])
    y_pred_1 = torch.FloatTensor([[1.23, 2.01, 0.4, 1.02], [0.45, 1.04, 1.02, 3.12]])
    y_pred_2 = torch.FloatTensor([[2.01, 1.23, 1.02, 0.4], [3.12,1.04, 1.02, 0.45]])
    y_pred_3 = torch.FloatTensor([[2.01, 1.23, 1.02, 0.4], [0.45, 1.04, 1.02, 3.12]])
    y_pred_4 =  torch.FloatTensor([[1.23, 2.01, 0.4, 1.02], [3.12,1.04, 1.02, 0.45]])
    batch_weight = torch.FloatTensor([0.9, 1.3])
    print(bweight_lambda_mrr_loss(y_pred_3, y_true, batch_weight))
    print(bweight_lambda_mrr_loss(y_pred_4, y_true, batch_weight))