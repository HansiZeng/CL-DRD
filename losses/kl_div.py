import torch 
import torch.nn as nn 
import torch.nn.functional as F

class KLDiv(nn.Module):
    def __init__(self, T=1.):
        super(KLDiv, self).__init__()
        self.T = T
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, y_pred, y_true):
        """
        Args:
            M_s: BxK, student scores
            M_t: BxK, teacher scores
        """ 
        assert y_pred.dim() == y_true.dim() == 2
        
        y_pred = F.log_softmax(y_pred/self.T, dim=-1)
        y_true = F.softmax(y_true/self.T, dim=-1)
       
        return self.kl_loss(y_pred, y_true)



if __name__ == "__main__":
    M_s = torch.tensor([[2.0, 1.0, 1.0], [3.0, 1.5, 2.5]])
    M_t = torch.tensor([[2.5, 1.5, 2.], [3., 2, 2.5]])

    criterion = KLDiv()
    print(criterion(M_s, M_t)) 