# normed_constrastive_metric

This repo contains a minimal research level code-implementation for the paper DEEP NORMED EMBEDDINGS FOR PATIENT REPRESENTATION.

The file trip.py contains the code for the loss function, pytorch dataset class, and training loops of the representation learning.

The Dataset class assumes the data resides in a Pandas DataFrame with appropriate labels, however it should be straightforward to adapt to other forms.

To reproduce our experiments you should first gain access to the MIMIC_III data. The preprocessing steps and the representation learning which was used (prior to learning the new embedding) can be found at https://github.com/thxsxth/POMDP_RLSepsis.
