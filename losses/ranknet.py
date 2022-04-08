import torch

def ranknet_loss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, reduction="mean", sigma=1.):
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

    inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    losses = torch.log(1. + torch.exp(-scores_diffs))  #[bz, topk, topk]

    if reduction == "sum":
        loss = torch.sum(losses[padded_pairs_mask])
    elif reduction == "mean":
        loss = torch.mean(losses[padded_pairs_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


if __name__ == "__main__":
    y_pred = torch.FloatTensor([[103.8560, 104.2479, 102.9454, 103.0578,  98.6101, 100.2017, 100.1513,                                                                                                                            
         100.0354,  99.1560, 101.1047,  97.7531,  98.9953, 101.6970, 101.1184,                                                                                                                            
          98.9523,  98.2248,  99.3415,  98.2269,  98.9324,  97.9243,  99.5813,                                                                                                                            
          95.6870,  99.5487, 101.5185,  96.9145, 102.6490, 100.5021,  97.7515,                                                                                                                            
          97.8676,  99.5976],                                                                                                                                                                             
        [105.8982, 105.9335, 105.2820, 106.2369, 103.3414, 105.1359, 105.7083,                                                                                                                            
         103.9510, 105.5665, 105.3788, 104.6647, 104.4636, 102.8736, 104.4074,                                                                                                                            
         103.8423, 104.3142, 104.2956, 102.9430, 103.5177, 105.1869, 105.0547,                                                                                                                            
         104.9325, 104.3588, 104.5267, 104.2974, 103.2128, 102.7218, 104.0699,                                                                                                                            
         103.0756, 105.6170]])
    y_true = torch.FloatTensor([[6.2734, 6.2188, 6.0039, 4.9336, 3.6836, 3.3691, 3.3047, 3.2852, 3.2480,                                                  
         3.0371, 2.5020, 2.1699, 2.0488, 1.9375, 1.9375, 1.7100, 1.5947, 1.5781,                                                                                                                          
         1.5205, 1.4004, 1.3730, 1.3105, 1.3027, 1.2744, 1.2715, 1.2705, 1.0928,                                                                                                                          
         1.0557, 0.9521, 0.9409],                                                                                                                                                                         
        [8.2500, 8.2188, 8.0703, 7.9375, 7.8906, 7.7969, 7.7344, 7.7070, 7.6562,                                                                                                                          
         7.6484, 7.4609, 7.4102, 7.3789, 7.2930, 7.2383, 7.2148, 7.1836, 7.1836,                                                                                                                          
         7.0391, 6.9570, 6.9453, 6.9414, 6.7930, 6.7539, 6.6797, 6.6367, 6.5547,                                                                                                                          
         6.5430, 6.4531, 6.3438]])

    print(ranknet_loss(y_pred, y_true))
    
