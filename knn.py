import pandas as pd 
import numpy as np
from collections import Counter
import math
import sys
from operator import itemgetter
import pickle


def train_kkn(mode_file,model_file):

    #Reading data from test_file.txt
    data_train = pd.read_csv(mode_file, sep=" ", header=None)
    with open(model_file,'wb') as file:
        pickle.dump(data_train,file)
    print("Training complete!")

def test_knn(mode_file,model_file):

    data_testFile = pd.read_csv(mode_file, sep=" ", header=None)
    image_id=data_testFile.iloc[:, 0]
    test_labels=data_testFile.iloc[:, 1]
    test_data=data_testFile.iloc[:, 2:]
        
    #Reading data from model_file.txt
    data_trainFile=None
    with open(model_file,'rb') as file:
        data_trainFile=pickle.load(file)
    
    train_labels=data_trainFile.iloc[:,1]
    train_data=data_trainFile.iloc[:,2:]
        
    #Converting the train and test data to list
    test_data = test_data.values.tolist()
    train_data = train_data.values.tolist()
        
    distance_list=[]
    predictions={}
    correct=0
    accuracy=0
    knn=65
    with open("nearest_output.txt",'w') as out:
        for test in range(len(test_data)):
            print("Testing sample", test, "of", len(test_data))
            for train in range(len(train_data[:500])):
                #applying euclidean distance
                distance=math.sqrt(np.sum(np.power((np.array(test_data[test]) - np.array(train_data[train])), 2)))
                distance_list.append((distance,str(train_labels[int(float(train))])))
                #sorting the array to get all distances in ascending order    
            distance_list.sort(key=itemgetter(0))
            #finding most common orientation using the knn value 
            counts = Counter(x[1] for x in distance_list[:knn])
            predictions[test]=counts.most_common(1)[0][0]
            test_pred=predictions[test]
            test_og = test_labels[test]
            #increasing the count is orientation is predicted correct
            if int(test_pred) == int(test_og):
                correct+=1 
            out.write(str(image_id[test])+" "+str(predictions[test]))
            out.write("\n")
            distance_list=[]
        accuracy=float(correct)/float(len(test_data))*100     
        print ("Accuracy: "+str(accuracy)+"%", "correct:", correct,"tested: "+str(len(test_data)+1))
        out.close()