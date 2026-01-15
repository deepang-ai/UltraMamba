import torch

def KL_divergence(mu1, logvar1, mu2=None, logvar2=None, eps=1e-8):
    " KLD(p1 || p2)"
    if mu2 is None:
        mu2 = mu1.new_zeros(mu1.shape)  # prior
        logvar2 = torch.log(mu1.new_ones(mu1.shape))
        eps = 0
    var1 = logvar1.exp()
    var2 = logvar2.exp()  # default : 1
    #     KLD = 0.5*torch.mean(torch.sum(-1  + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps), axis=1))
    KLD = 0.5 * torch.mean(-1 + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps))

    return KLD