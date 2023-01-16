# **MinMixer**

A minimal [MLP-Mixer](https://arxiv.org/abs/2105.01601) for image classification.


# Installation

To run *MinMixer*, install the latest master directly from GitHub. For a basic install, run:

```console
git clone https://github.com/kaifishr/MinMixer
cd MinMixer 
pip3 install -r requirements.txt
```


# Getting Started

Run the following commands to start a training:

```console
cd MinMixer 
python train.py 
```


# Weight Visualization

Some important metrics and trained parameters of the token-mixing MLP blocks can be visualized with Tensorboard:

```console
cd MinMixer 
tensorboard --logdir runs/
```

The following visualizations show some of the weights learned during training by the token-mixing MLPs.

<center>

![](/assets/images/weights.png)

</center>

<center>

| Layer 1 | Layer 2 | Layer 3  | Layer 4  | Layer 5  | Layer 6  | Layer 7  | Layer 8 |
|---|---|---|---|---|---|---|---|
| ![](/docs/images/layer_01.png) | ![](/docs/images/layer_02.png) | ![](/docs/images/layer_03.png) | ![](/docs/images/layer_04.png) | ![](/docs/images/layer_05.png) | ![](/docs/images/layer_06.png) | ![](/docs/images/layer_07.png) | ![](/docs/images/layer_08.png)

</center>


# References

[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)


# License

MIT