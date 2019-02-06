# AdvGAN-tf
Tensorflow implementation of Generating Adversarial Examples with Adversarial Networks 

##USAGE
Create a ```./weights``` directory as well as subdirectories ```./weights/generator```, ```./weights/discriminator```, ```./weights/target_model``` to contain the saved weights for each model.

First run:
```
python target_models.py
```
which will extract the MNIST dataset using the Keras API and train a simple CNN model that will serve as the 'target model' for the generator to trick. 

Next, run:
```
python AdvGAN.py
```
This script will first train the generator. You can specifiy whether or not you want it to be targeted. A different generator will be trained for each target. You will want to tweak the weight paths for each target (or I will update that soon). 

Once the training process is complete, the function ```attack``` will be called. This function will load the weights from the generator and run an attack on the test set. It will also print out a before and after picture of two images from the last batch. 
