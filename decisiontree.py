import pandas as pd 
import numpy as np
from collections import Counter
import pickle

#Tree class node
class TreeNode:

    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

#Label counter
def label_counter(labels):
    return Counter(labels)

#Question classification
def question(data_row,col,val):
    return val>=data_row[col]

#Partition based on value
def partition(rows,col,val):

    col_data=rows[:,col]
    true_data, false_data = rows[col_data<=val],rows[col_data>val]
    return true_data,false_data

#Gini calculation
def gini(rows):

    label_counts = label_counter(rows[:,0])
    impurity = 1
    for label in label_counts:
        label_prob = label_counts[label] / float(len(rows))
        impurity -= label_prob**2
    return impurity

#Information gain calculation
def info_gain(left, right, current_uncertainty):

    level_prob = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - level_prob * gini(left) - (1 - level_prob) * gini(right)

#Function to determine best partition
def find_best_partition(rows):
    max_gain=0
    best_question=None
    random_cols=[0,7,32,53,30,23,145,146,108,179,61,11,107,118,140,23]
    current_uncertainty=gini(rows[:,random_cols])
    for col in random_cols[1:]:
        feature_values=list(set(rows[:,col]))
        partition_val=[np.percentile(feature_values,20),np.percentile(feature_values,40),
                       np.percentile(feature_values,60),np.percentile(feature_values,80),
                       np.percentile(feature_values,100)]
        
        for val in partition_val:
            question=(col,val)
            true_data,false_data=partition(rows,*question)

            if len(true_data)==0 or len(false_data)==0:
                continue
            new_info_gain=info_gain(true_data,false_data,current_uncertainty)

            if new_info_gain>max_gain:
                max_gain, best_question = new_info_gain, question

    return max_gain,best_question

#Function to build the Tree
def build_tree(rows,depth):

    gain, question = find_best_partition(rows)
    if gain == 0 or depth==9:
        return label_counter(rows[:,0])

    true_rows, false_rows = partition(rows, *question)

    true_branch = build_tree(true_rows,depth+1)

    false_branch = build_tree(false_rows,depth+1)

    return TreeNode(question, true_branch, false_branch)

#Classification function
def classify(row, node):

    if type(node)==Counter:
        return node

    if question(row,*node.question):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

#Function to train the tree
def train_tree(mode_file,model_file):
    data=pd.read_csv(mode_file, sep=" ", header=None)
    train_data=data.values[:,1:]
    tree=build_tree(train_data,0)

    with open(model_file,'wb') as file:
        pickle.dump(tree,file)
    
    print("Training complete!")

#Function to test with the Tree
def test_tree(mode_file,model_file):
    res=[]
    tree=None
    data=pd.read_csv(mode_file, sep=" ", header=None)
    image_id=data.values[:,0]
    labels_test=data.values[:,1]
    test_data=data.values[:,2:]

    with open(model_file,'rb') as file:
        tree=pickle.load(file)

    for row in test_data:
        data=classify(row,tree)
        res+=[max(data, key=data.get)]
    
    with open("tree_output.txt",'w') as file:
        for iterator in range(len(image_id)):
            file.write(image_id[iterator]+" "+str(res[iterator])+'\n')
    res=np.asarray(res)
    sum_data=(sum(res==labels_test)/len(labels_test))*100
    print("Accuracy for given test data is - "+str(round(sum_data,2)))