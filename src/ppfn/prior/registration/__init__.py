"""Generative prior for the encoder-decoder registration project — see
`docs/ARCHITECTURE.md` §1. NumPy throughout (the prior runs in dataloader
workers, per `CLAUDE.md`'s repo conventions); nothing here imports torch.

Distinct from `ppfn.prior.bnn` (the older, single-cloud BNN-prior-for-plain-PFN
sampler used by `configs/prior/bnn.yaml`) — that prior has no notion of two
warped point clouds, a latent frame, or registration, and is left untouched.
"""
