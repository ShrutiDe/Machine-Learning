# B551 Assignment 4: Machine learning

## Classifier1: nearest

k-nearest neighbour algorithm simply finds k nearest samples of a specific object. It memorizes all the data and labels of the train data provided and predicts the  label of the most similar training image of a given test image. Here the labels given are orientation of the image which is to be predicted using k-nearest neighbour algorithm.

Most important was to find a good k value and efficient distance formula to use.
Following are the steps followed: 
#### Training:
For k-nearest neighbour algorithm, there is no training required. We have to just separate the useful data and save in model_file.txt. From this file the model will test the test_file.txt

Model is trained using following inputs to code:
			 
`./orient.py train train_file.txt model_file.txt nearest`

#### Testing:
Testing is done using data in model_file.txt and tested on test_file.txt

Model is tested using following inputs to code:.

`./orient.py test test_file.txt model_file.txt nearest`


In this approach, we trained the file as mentioned and tested the same in the test_file.txt. The data file consisted of image id along with orientation and image vector.
We tested each test image with every other train image and compared there Euclidean distance. This distances were saved in an array.
For k-nearest neighbour algorithm, we sorted this array and took the first k distances from the array. Calculated the count of the most common orientation and considered it as the predicated orientation for the corresponding test image.

The accuracy we obtained is around **71%**. We tried various values of k to get best possible accuracy.

## Classifier2: tree

A decision Tree uses a branching method to classify the data. It breaks down data to smaller subsets, each decision node has two branches, a True and False.

## Implementation

**Building the Tree**

The Tree is built recursively by deciding on the best partition question. For deciding the best partition, we calculate what question gives the highest information gain.

**Design choice**

Information gain is calculated by using Gini formula and is computed for the question.

To find best partition for a data column, partitions are made on the 20, 40, 60, 80 ,100th percentiles and the highest information gain is selected.

This is done over random columns of count 16. After extensive testing, we determined that a certain set of columns had the best information gain and we picked those alone for construction. This helped limit the computation time for partitioning.

Finally, a Tree depth of 9 was taken, this proved to give the best model.

**Classification**
For classifying the labels, we stored a dictionary of counts for each label at the leaf nodes of the Tree. The label with the maximum count was taken as the solution to the classification.

**Model**
For the model we used pickle package to write it to a file. The Tree class object was stored in the file.
The accuracy we obtained using this algorithm was around **58%**.

## Classifier1: nnet

We implemented the neural network somewhat like how TensorFlow 2.0 or Keras works. We can provide a variable architecture in the `train_nn()` function.

It was important to be able to test the model with different architectures to get good results. We fixiated on a model with the following architecture:
```
input (192)
hidden1 (70)
hidden2 (40)
hidden3 (20)
output (4) {one-hot encoding}
```

#### One-hot encoding
It was necessary to encode the given classes in one-hot encoding as not doing so would result in a regression task that is much harder than a classification task.

#### Training:
We initialize weights at random and train for 10 epochs.

Model is trained using following inputs to code:
			 
`./orient.py train train_file.txt model_file.txt nnet`

#### Testing:
Testing is done using data in model_file.txt and tested on test_file.txt

Model is tested using following inputs to code:.

`./orient.py test test_file.txt model_file.txt nnet`

The accuracy we obtained is around **62%**.

#### Room for improvement:
K-NN speed-up using KD trees.

Cooler features in the NN like
1. Dropout
2. RMS Prop
3. Momentum
4. ADAGrad
could have been experimented with in order to further increase accuracy, subject to more time.
