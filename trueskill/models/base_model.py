

import pyro
# from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam
import pyro.distributions as dist
import torch

def model(outcome, sparse_game_matrix): 
    """
    Say there are P players
    Say there are G games 
    
    outcome [G, 1] where G is games and extra dimension is just 1 or zero depending on whether 
    game_matrix has entries [G,P] (use sparse multiplication COO)

    
    """
    
    # random normal distribution with vector [P, 1]
    # skill = pyro.sample(...)

    G, P = sparse_game_matrix.shape

    skill = pyro.sample("skill", dist.Normal(0, 1).expand([P])).to(torch.float64)
    skill = torch.reshape(skill, (-1, 1))
    print(skill.shape)

    diff = torch.sparse.mm(sparse_game_matrix, skill.to_sparse())
    diff = diff.to_dense()
    # random normal distribution with means as differences 
    score = pyro.sample("score", dist.Normal(diff, 1))
    prob = torch.sigmoid(score)

    # Outcome is drawn from a probability dist with p being result of sigmoid 
    return pyro.sample("obs", dist.Bernoulli(prob), obs=outcome)






    
    



