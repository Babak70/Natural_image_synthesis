# Natural Image Synthesis for the Retina with Variational Information Bottleneck Representation  
  
Code for "Natural image synthesis for the retina with variational information bottleneck representation".  
  
## Description  
  
### Software Requirements  
  
1. Tensorflow v2.3.0  
2. Tensorflow-probability 0.11  
3. Numpy, imageio, matplotlib, shutil, sklearn  
  
### Hardware Requirements  
  
Our custom scripts have been executed on an NVIDIA RTX 3090 GPU.  
  
## How to Run the Code  
  
1. Change the directory to the path of the codes where `main.py` exists.  
2. To run the code on a trained model, make sure the `models` folder is downloaded and placed in the same directory as `main.py`. The `data` folder should be one level higher than `main.py` directory.  
  
Enter the following commands in the command prompt:  
  
```bash  
python main.py --task test_forward # to see the spiking predictions of the IB-GP model on the Natural dataset  
python main.py --task traversal # to run the traversal on the IB-GP model  
python main.py --task train_forward # to train the IB-GP/ IB-Disjoint model  
python main.py --task adaptive_train # to train the image synthesizer  
python main.py --task latent_analyze # to visualize the autocorrelation of the latents and neural dynamics for IB-GP model
```

The parameters of the network, such as the beta value, the number of training epochs, and the latent dimension, can be changed by:
```bash
python main.py --task train_forward --num_epochs 1 --beta 10 --latent_dims 15
```
The model type can be set by:
```bash
--model_type IB-GP (default)  or IB-Disjoint
```
Note: The beta value should be inverted. For example, for a beta value of 0.05, set the beta to 20.

# Reference
If you use this code, please cite the following paper:

@article{rahmani2022natural,  
  title={Natural image synthesis for the retina with variational information bottleneck representation},  
  author={Rahmani, Babak and Psaltis, Demetri and Moser, Christophe},  
  journal={Advances in Neural Information Processing Systems},  
  volume={35},  
  pages={6034--6046},  
  year={2022}  
}  
