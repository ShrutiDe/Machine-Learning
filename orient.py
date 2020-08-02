import decisiontree
import knn
import nn
import sys


#Function to call the appropriate Training function
def train_data(model,mode_file,model_file):
    if model=='nearest':
        knn.train_kkn(mode_file,model_file)
    elif model=='tree':
        decisiontree.train_tree(mode_file,model_file)
    elif model=='nnet':
        nn.train_nn(mode_file, model_file)
        return
    else:
        return

#Function to call the appropriate Testing function
def test_data(model,mode_file,model_file):
    if model=='nearest':
        knn.test_knn(mode_file,model_file)
    elif model=='tree':
        decisiontree.test_tree(mode_file,model_file)
    elif model=='nnet':
        nn.test_nn(model_file, model_file)
        return
    else:
        return

# Main Function
if __name__ == "__main__":
    if(len(sys.argv) != 5):
        raise Exception("usage: orient.py [train/test] [train_file.txt/test_file.txt] model_file.txt [model]")
    
    mode, mode_file, model_file, model = sys.argv[1:]

    if mode == 'train':
        train_data(model,mode_file,model_file)
    elif mode == 'test':
        test_data(model,mode_file,model_file)