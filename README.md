# Sim-to-real diffusion models for microstructure prediction in metal additive manufacturing
This repository contains the codes for implementing sim-to-real microstructure prediction in additive manufacturing (AM).

Two separate folders are archived in this repository: __diffusion__ and __jax-am__, where __diffusion__ contains the codes of __denoising diffusion probabilistic model (DDPM)__ and the corresponding fine-tuning and distillation version, __jax-am__ contains the multi-layer multi-track __computational fluid dynamics (CFD)__ model for temperature field and __phase-field (PF)__ simulation for microsturcture field in AM.

## JAX-AM for data preparation
This part is a modified version of the JAX-AM repository of our group (See [JAX-AM](https://github.com/CMSL-HKUST/jax-am)).

__Computational Fluid Dynamics (CFD)__

<p align="middle">
  <img src="docs/cfd.gif" width="700" />
</p>
<p align="middle">
    <em >Multi-layer multi-track CFD simulation for temperature field.</em>
</p>

__Phase-field (PF)__

<p align="middle">
  <img src="docs/pf.gif" width="700" />
</p>
<p align="middle">
    <em >Multi-layer multi-track PF simulation for microsturcture field.</em>
</p>

 ## Denoising Diffusion Probabilistic Model (DDPM)

A conditional DDPM is pre-trained with simulation microstructures. Then it is fine-tuned or distilled with experiment microstructures. The code is build upon [Diffusers](https://github.com/huggingface/diffusers).

<p align="middle">
  <img src="docs/tune.jpg" width="200" />
</p>
<p align="middle">
    <em >Microstructure predicted with fine-tuned diffusion model.</em>
</p>

<p align="middle">
  <img src="docs/dis.jpg" width="200" />
</p>
<p align="middle">
    <em >Microstructure predicted with distilled diffusion model.</em>
</p>

The pre-trained DDPM for cross-section microstructure generation can be found in [Releases](https://github.com/xiezy964/sim-to-real/releases/tag/model).
