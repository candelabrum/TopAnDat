# TopAnDat
Official code repository for article Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story.

Text of the paper is available at <a href="https://arxiv.org/abs/2511.15210">ArXiv:2511.15210</a>

# Speeding up the PHDim calculation:
In the notebook TestGPUPHDimFinal.ipynb, you can find a comparison of two approaches for calculating PHDim - using cuda and cpu.
With embedding dimensions > 1000, it makes sense to use the cuda version to calculate the pairwise distance matrix.

# Credentials

This code is based on the repository https://github.com/ArGintum/GPTID
