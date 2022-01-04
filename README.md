# User-Centric Perspective in Random Access Cell-Free Aided by Spatial Separability
This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the article mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, T. Abrão, J. C., Marinello, “User-Centric Perspective in Random Access Cell-Free Aided by Spatial Separability,” to be published. Available on: https://arxiv.org/abs/2107.10294.

I hope this content helps in your reaseach and contributes to building the precepts behind open science. Remarkably, in order to boost the idea of open science and further drive the evolution of science, I also motivate you to share your published results to the public.

If you have any questions and if you have encountered any inconsistency, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
In a cell-free massive multiple-input multiple-output (CF-mMIMO) network, multiple access points (APs) actively cooperate to serve users' equipment (UEs). We consider how the random access (RA) problem can be addressed by such a network under the occurrence of pilot collisions. To find a solution, we embrace the user-centric perspective, which basically dictates that only a preferred set of APs needs to serve a UE. Due to the success of the strongest-user collision resolution (SUCRe) protocol for cellular (Ce) mMIMO, we extend it by considering the new setting. Besides, we establish that the user-centric perspective naturally equips a CF network with robust fundamentals for resolving collisions. We refer to this foundation as spatial separability, which enables multiple colliding UEs to access the network simultaneously. We then propose two novel RA protocols for CF-mMIMO: i) the baseline cell-free (BCF) that resolves collisions with the concept of spatial separability alone, and ii) the cell-free SUCRe (CF-SUCRe) that combines SUCRe and spatial separability principle to resolve collisions. We evaluate our proposed RA protocols against the Ce-SUCRe. Respectively, the BCF and CF-SUCRe can support 7x and 4x more UEs' access on average compared to the Ce-SUCRe with an average energy efficiency gain based on total power consumed (TPC) by the network per access attempt of 52$\times$ and 340$\times$. Among our procedures, even with a higher overhead, the CF-SUCRe is superior to BCF regarding TPC per access attempt. This is because the combination of methods for collision resolution allows many APs to be disconnected from the RA process without sacrificing much the performance. Finally, our numerical results can be reproduced using the code package available on: github.com/victorcroisfelt/cf-ra-spatial-separability.

## Content
The codes provided here can be used to simulate Figs. 2 to 7. The code is divided in the following way:
  - scripts starting with the keyword "plot_" actually plots the figures using matplotlib.
  - scripts starting with the keyword "data_" are used to generate data for curves that require a lot of processing. The data is saved in the /data folder and used by the respective "plot_" scripts.
  - scripts starting with the keyword "lookup_" are used to exhaustively find parameters, such as: number of nearby APs, Ccal_size, number of pilot-serving APs, Lmax, and effective DL transmit power for Estimator 3, delta. Considering the practical scenario, it also makes use of method proposed in Algorithm 1. 

Further details about each file can be found inside them.

## References
Part of the code is based on the following previous work:

Emil Björnson, Elisabeth de Carvalho, Jesper H. Sørensen, Erik G. Larsson, Petar Popovski, “A Random Access Protocol for Pilot Allocation in Crowded Massive MIMO Systems,” IEEE Transactions on Wireless Communications, vol. 16, no. 4, pp. 2220-2234, April 2017.

The authors provide a code package on: https://github.com/emilbjornson/sucre-protocol.

## Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider to cite our aforementioned work.
