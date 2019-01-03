
### Team : Donghao, Tiffany, Sahil, Chanand

Our goal is to use 3D-CNN to determine whether patients have alzheimer diseases based on the MRI scans. This is essential in the future as more countries are experiencing an aging population. As more people live longer, alzheimer are becoming more prevalent. If we could create a model to help doctors detect alzheimer disease, this will greatly increase the productivity of such doctors.

Following three steps were performed: 
### Visualisation
We started the modeling by trying to understand the structure of the MRI scans. We looked at the sample scans in different angles. We also tried to see based on these angles if we can visually detect the differences between patients with alzheimer disease and those without. We also tried to understand the effective area or the area of consideration where the brain is by plotting it using nibabel. This helped us in identifying if there is a need to crop the image to an appropriate size in order to reduce the space each image took.


### Data Preprocessing
1. OASIS - 
Mapped MRI image files for the labels Cognitively normal, AD dementia, and Uncertain dementia within the 180 day range.
2. NACC - 
Mapped MRI scan file names to labels from data_nacc_diagnosis.xlsx
3. Encoded sex and normalise age.
4. Merge OASIS & NACC data samples
5. Divide data samples into training, validation, and test sets
6. Having difficulty to load the dataset converted the data to into 4 NumPy arrays: img (just image path to save memory), age, sex, label

### Deep Learning Model
#### Input
1. Load all input tensors
2. Use one-hot encoding for labels
3. Add new axis to img, age, sex
4. Define function to retrieve images from img_path, including downsampling and normalizing images 


#### Model
1. Initialize_parameters (num_epochs=3, learning_rate = 0.01, and minibatch_size = 50)
2. Forward_propagation:
CONV3D -> RELU -> MAXPOOL -> CONV3D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED -> FULLYCONNECTED
3. Compute Cost using cross entropy
4. Random_mini_batches was used to increase training speed by dividing the dataset into batches 
5. The model specifies learning_rate, num_epochs, minibatch_size, and whether the model is pretrained, to get cost and training & validation accuracy rate

#### References – 

1.	Ehsan Hosseini et al: Alzheimer’s disease diagnostics by a 3D deeply supervised adaptable convolutional network
2.	Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; Microsoft Research: Deep Residual Learning for Image Recognition
3.	Karen Simonyan & Andrew Zisserman: Very Deep Convolutional Networks for Large-Scale Image Recognition
4.	Ilija Ilievski et al: Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates 
5.	Tobias Hinz et al: Speeding up the Hyperparameter Optimization of Deep Convolutional Neural Networks 
6.	Volodymyr Turchenko, Eric Chalmers, Artur Luczak: A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in Caffe 
7.	G. E. Hinton and R. R. Salakhutdinov: Reducing the Dimensionality of Data with Neural Networks 
8.	Salah Rifai et al: Contractive Auto-Encoders: Explicit Invariance During Feature Extraction 
9.	Xifeng Guo, Xinwang Liu, En Zhu, and Jianping Yin: Deep Clustering with Convolutional Autoencoders- 
10.	Maximilian Kohlbrenner et al: Pre-Training CNNs Using Convolutional Autoencoders 
11.	Soren Becker et al: Interpreting and Explaining Deep Neural Networks for 
12.	Classification of Audio Signals 
13.	Johannes Rieke et al: Visualizing Convolutional Networks for MRI-based 
14.	Diagnosis of Alzheimer’s Disease
Appendices (if used) - Github URL:

⋅⋅* https://github.com/quqixun/BrainPrep
⋅⋅* https://link.springer.com/chapter/10.1007/3-540-32390-2_64
⋅⋅* https://gist.github.com/shagunsodhani/5d726334de3014defeeb701099a3b4b3

