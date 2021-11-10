# SWN_DCGAN_MNIST

## Implementation of Specteral Weight Normalization with an attempt to integrate Gated Convs For contexual convolutions 
- Testing Environmet:
--Ubuntu 18
--Python 3.6
--1GPU 

- Commands to run the training script
- Create a conda env by:
--```conda env create --file environment.yml``` 
- All Training Configurations can be accessed through the train_config.json file
-- ``` {
    "arch_type" : "2D",
    "nn_type" : "convbn2d",
    "batch_size": "256",
    "n_epochs": "2",
    "display_step":"2",
    "device": "cuda",
    "visualize": "NO"
} ```
--The main configurations are **nn_type** and **arc_type** : ['linear', 'bn', 'convbn2d', 'swnconvbn2d', 'gconv2d', 'swngconv2d'] and **arch_type** either ["1D"] for the "linear" and "bn" architectures and ["2D"] for the rest of the architectures
--linear --> linear only layers
--bn --> batch norm layers
--convbn2d --> convolution bn 2d layers
--swnconvbn2d --> convolution bn 2d w/t SWN
--The rest of the configurations can be set to prefrence of training environment
- Run the Training Script:
--```python DCGAN.py```
- All Plots, Results and Saved (.pth) models will be saved accordingly
