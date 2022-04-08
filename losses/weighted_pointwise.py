import torch

def weighted_pointwise_loss(y_pred, y_weight, T=1.):
    """
    y_pred: FloatTensor [bz, topk + topN]
    y_weight: FloatTensor [bz, topk + topN]
    T: temperature
    """
    assert torch.sum(y_weight<0) == 0. 

    losses = torch.log(1. + torch.exp(-y_pred / T)) * y_weight #[bz, topk + topN]
    loss = torch.mean(losses)

    return loss


if __name__ == "__main__":
    y_weight = torch.FloatTensor([[1.,1./2, 1./3, 0.,0., 0., 0.]])
    y_pred_1 = torch.FloatTensor([[2.3, 1.2, 1.1, 0.5, 0.23, 0., 40]])
    y_pred_2 = torch.FloatTensor([[1.4, 1.2, 1.1, 0.5, 20, 423, 40]])

    print(weighted_pointwise_loss(y_pred_1, y_weight))
    print(weighted_pointwise_loss(y_pred_2, y_weight))

