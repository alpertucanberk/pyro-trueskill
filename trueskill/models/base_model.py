


def model(outcome, player1, player2, game_matrix): 
    """
    outcome [N, 1] where N is games and extra dimension is just 1 or zero depending on whether 
    player 1 or player 2 wins
    player1 is one-hot vector encoding of player id
    player2 ""
    game_matrix has entries [G,P] (use sparse multiplication COO)


    Say there are P players
    Say there are G games 
    """
    
    # random normal distribution with vector [P, 1]
    skill = pyro.sample(...)
    diff = game_matrix @ skill

    # random normal distribution with means as differences 
    score = pyro.sample(Normal(diff, 1))
    prob = sigmoid(score)

    # Outcome is drawn from a probability dist with p being result of sigmoid 
    pyro.sample(dis.Bernoulli(prob), obs=outcome)


# For the guide do 
# Look it up 
guide = pyro.AutoDiagonalNormal()






    
    



