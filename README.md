
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

