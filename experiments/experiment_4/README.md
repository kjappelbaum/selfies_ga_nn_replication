# Constrained optimization: Generating molecules of specific chemical interest

This is based on an experimental setup from [You et al.](https://cs.stanford.edu/people/jure/pubs/gcpn-neurips18.pdf) in which one starts from on of 800 low performing structures and uses the GA to improve it.
For this, the discriminator is not used in the original study and instead a similarity penalty is added to penalize structures that are dissimilar from the original one (similarity measured using Morgan fingerprints).

In the original paper, "sucess" is reported when the logP improved and the GA remained within the similarity constraint.

On our machine one generation of the GA takes about ~1 min. Since each GA is run for 20 generation and the experiment is performed for 800 molecules, it can take significant time. 