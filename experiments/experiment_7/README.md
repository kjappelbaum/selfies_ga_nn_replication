# Extending stagnation measure

Instead of using just observing the fitness we now use the similarity of Morgan fingerprints for the adaptive penalty.
For this we can vary

- the "watchtime" (analogous to the "patience" in early stopping)
- the similarity threshold (mean Tanimoto distance of the highest scoring molecules)
- the beta we apply
