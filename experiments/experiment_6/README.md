# Flipping the objective

Here we run the same experiment as with the beta sweeps, but we flip the way in which we train the discriminator.
Now long surviving molecules obtain a score of 1 and the reference dataset a score of zero, and we will investigate positive and negative beta. This is, for negative beta we effectively _penalize_ long surviving molecules.
