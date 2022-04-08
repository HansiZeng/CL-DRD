import torch 
import torch.nn as nn 

class MarginMSE(nn.Module):
    def __init__(self):
        super(MarginMSE, self).__init__()
    
    def forward(self, M_s, M_t):
        """
        Args:
            M_s: BxK, student scores
            M_t: BxK, teacher scores
        """ 
        assert M_s.dim() == M_t.dim() == 2
        M_s = M_s.unsqueeze(2) - M_s.unsqueeze(1)
        M_t = M_t.unsqueeze(2) - M_t.unsqueeze(1)

        loss = (M_s-M_t) ** 2
        return torch.mean(loss)

if __name__ == "__main__":
    M_s = torch.tensor([[2.0, 1.0, 1.0], [3.0, 1.5, 2.5]])
    M_t = torch.tensor([[2.5, 1.5, 2.], [3., 2, 2.5]])

    criterion = MarginMSE()
    print(criterion(M_s, M_t)) 