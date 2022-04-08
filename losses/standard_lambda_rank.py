import torch

def lambda_loss(y_pred, y_true, eps=1e-4, padded_value_indicator=-1, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="mean", reduction_log="natural", gain="power"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]

    if gain == "power":
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]
    elif gain == "linear":
        maxDCGs = torch.sum(((y_true_sorted - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G =   (true_sorted_by_preds - 1) / maxDCGs[:, None]
    else:
        raise ValueError(f"{gain} not defined.")

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))

if __name__ == "__main__":
    y_true_1 = torch.FloatTensor([[6.59375,6.51171875,6.359375,5.7734375,5.73828125,5.7109375,5.18359375,5.12890625,4.921875,4.82421875]])
    y_true_2 =  torch.FloatTensor([[10.2734375,10.15625,9.640625,9.6328125,9.625,9.546875,9.453125,9.3046875,9.0703125,9.0546875]])
    y_true_3 = torch.FloatTensor([[4.765625,4.62109375,4.33203125,4.328125,4.29296875,4.20703125,4.1171875,4.02734375,3.94140625,3.91796875]])
    y_true_4 = torch.FloatTensor([[3., 3., 3, 3, 3, 3, 2, 2, 1,1]])

    #y_pred = torch.FloatTensor([[100, 101, 102, 104, 106, 103, 102.3, 98.1, 110.4, 100.5]])
    #y_pred = torch.FloatTensor([[110.4, 106, 104, 103, 102.3, 102, 101,100.5, 100, 98.1]])
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
    y_true_1 = torch.FloatTensor([[6.2734, 6.2188, 6.0039, 4.9336, 3.6836, 3.3691, 3.3047, 3.2852, 3.2480,                                                  
         3.0371, 2.5020, 2.1699, 2.0488, 1.9375, 1.9375, 1.7100, 1.5947, 1.5781,                                                                                                                          
         1.5205, 1.4004, 0,0,0,0,0,0,0,0,0,0],                                                                                                                                                                         
        [8.2500, 8.2188, 8.0703, 7.9375, 7.8906, 7.7969, 7.7344, 7.7070, 7.6562,                                                                                                                          
         7.6484, 7.4609, 7.4102, 7.3789, 7.2930, 7.2383, 7.2148, 7.1836, 7.1836,                                                                                                                          
         7.0391, 6.9570, 0,0,0,0,0,0,0,0,0,0]])
    y_true_2 = torch.FloatTensor([[3,3,3, 2, 1, 1, 1, 1, 1,                                                  
            1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],                                                                                                                                                                         
        [3,3,3, 2, 2, 2, 2, 2, 2,                                                                                                                          
         2, 2, 2, 2,2, 2, 2, 1, 1,                                                                                                                          
         1, 1, 0,0,0,0,0,0,0,0,0,0]])
    #print(lambda_loss(y_pred, y_true, weighing_scheme="ndcgLoss1_scheme", reduction_log="natural"))
    print(lambda_loss(y_pred, y_true_1, weighing_scheme="ndcgLoss1_scheme", reduction_log="natural"))
    print(lambda_loss(y_pred, y_true_2, weighing_scheme="ndcgLoss1_scheme", reduction_log="natural"))

    """
    print("lambdarank_scheme: ")
    print(lambda_loss(y_pred, y_true_1, weighing_scheme="lambdaRank_scheme"))
    print(lambda_loss(y_pred, y_true_2, weighing_scheme="lambdaRank_scheme"))
    print(lambda_loss(y_pred, y_true_4, weighing_scheme="lambdaRank_scheme"))

    print("rankNetWeightedByGTDiff_scheme: ")
    print(lambda_loss(y_pred, y_true_1, weighing_scheme="rankNetWeightedByGTDiff_scheme"))
    print(lambda_loss(y_pred, y_true_2, weighing_scheme="rankNetWeightedByGTDiff_scheme"))
    print(lambda_loss(y_pred, y_true_4, weighing_scheme="rankNetWeightedByGTDiff_scheme"))

    print("ndcgLoss1_scheme: ")
    print(lambda_loss(y_pred, y_true_1, weighing_scheme="ndcgLoss1_scheme"))
    print(lambda_loss(y_pred, y_true_2, weighing_scheme="ndcgLoss1_scheme"))
    print(lambda_loss(y_pred, y_true_4, weighing_scheme="ndcgLoss1_scheme"))

    print("ndcgLoss2_scheme: ")
    print(lambda_loss(y_pred, y_true_1, weighing_scheme="ndcgLoss2_scheme"))
    print(lambda_loss(y_pred, y_true_2, weighing_scheme="ndcgLoss2_scheme"))
    print(lambda_loss(y_pred, y_true_4, weighing_scheme="ndcgLoss2_scheme"))

    print("rankNet_scheme: ")
    print(lambda_loss(y_pred, y_true_1, weighing_scheme="rankNet_scheme"))
    print(lambda_loss(y_pred, y_true_2, weighing_scheme="rankNet_scheme"))
    print(lambda_loss(y_pred, y_true_4, weighing_scheme="rankNet_scheme"))
    """
