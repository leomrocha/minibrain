# Image Sensors

This section is dedicated to image sensors, which is the first step in the development of my ideas


I Will be working on different types of Autoencoders:

- Simple
- Convolutional
- Variational
- Multivariational * **this is a new development**

### The Autoencoders will be built by parts following the listed principles:

- Training will always be oriented to unsupervised or semi-supervised, the idea is to limit the supervision problem as much as possible and allow the network to train itself if possible
- Foveated sensors, with pyramidal resolution
- The complete image will also be used downsampled at a size that can be tackled in the sensor
- Each resolution level can be moved in position, but higher resolution will always be inside the next lower resolution capture size
- Zooming is allowed (means, changing -downsampling/upsampling- the resolution for the highest resolution part)
- Curriculum learning will take place, from simpler to more complex tasks
- Lower levels of the hierarchy 

### Steps:

1. Autoencoders will be generated for each resolution level independently
2. Once all resolution level autoencoders have been trained:
  * a controller to apply Transforming Autoencoders (Hinton et al.) ideas will be used
  * Input images might be separated by classes (prior knowledge) if available
  * Different tasks will be trained (simultaneously): denoise, transform (translate, rotate, rotate spatial, change ratio, etc)
3. Once the previous tasks have been trained, add another level for video ... TODO DEFINE IT WHEN NEEDED OR I HAVE A NEW IDEA


### Procedure

All experiments will be first developed in notebooks and later ported to the necessary libraries.

Development will be done with pytorch, because it is one of the easiest frameworks avaiable.


