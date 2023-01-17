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

Besides loss and accuracy, the weights of the token-mixing MLP blocks can be visualized with Tensorboard:

```console
cd MinMixer 
tensorboard --logdir runs/
```

The following visualizations show some of the parameters learned during training by the token-mixing MLPs.

<center>

![](/assets/images/weights.png)

</center>


# References

[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)


# License

MIT