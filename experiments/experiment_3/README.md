# Clustering RDKit fingerprint to interpret the GA trajectory 

First, all molecules are fingerprinted using the [RDKit fingerprint](https://www.rdkit.org/docs/RDKit_Book.html#rdkit-fingerprints). 

K-means clustering (k=20) is performed for the 50 best structures and the 
k-means clustering is then used to analyze which cluster is explored in which generations of the GA.

The script is a CLI that take the path to a results folder.