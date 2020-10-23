import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split #don't use for time-series baseline!
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loan():
    """
    Define a standard Loan model for testing.
    """
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available
        
        self.train_df, self.test_df = self.gather_data()
        self.train_df = self.preprocess_df(self.train_df)
        self.test_df = self.preprocess_df(self.test_df, True)

        self.train_split_test() #gather train and test splits.
        self.model = self.get_model()

        # implement backprop
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005) #adam works well for this.

    def gather_data(self):
        train_df = pd.read_csv("../../data/loan/train.csv")
        test_df = pd.read_csv("../../data/loan/test.csv")
        return train_df, test_df
    
    def preprocess_df(self, dataframe, isTest=False):
        """Preprocess a dataframe, unique to the loan_prediction dataset"""
        #perform deep copy, fixes self assignment bug:
        #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        df = dataframe.copy(deep=True)
        
        ### remove all rows with null values
        del df['Loan_ID'] #remove Loan_ID (irrelevant)

        # convert to binary variables

        ##----------------------------------------------------------------------------
        #### ----------------------------------Table----------------------------------
        ##----------------------------------------------------------------------------

        #> ----Gender---
        ## - Male: 0
        ## - Female: 1
        df.loc[(df.Gender == 'Male'),'Gender']=0
        df.loc[(df.Gender == 'Female'),'Gender']=1

        #> ----Married---
        ## - No: 0
        ## - Yes: 1
        df.loc[(df.Married == 'Yes'),'Married']=0
        df.loc[(df.Married == 'No'),'Married']=1

        #> ----Education---
        ## - Not Graduate: 0
        ## - Graduate: 1
        df.loc[(df.Education == 'Not Graduate'),'Education']=0
        df.loc[(df.Education == 'Graduate'),'Education']=1

        #> ----Self_Employed---
        ## - No: 0
        ## - Yes: 1
        df.loc[(df.Self_Employed == 'No'),'Self_Employed']=0
        df.loc[(df.Self_Employed == 'Yes'),'Self_Employed']=1


        #> ----Property_area---
        ## - Rural: 0
        ## - Urban: 1
        ## - Semiurban: 2
        df.loc[(df.Property_Area == 'Rural'),'Property_Area']=0
        df.loc[(df.Property_Area == 'Urban'),'Property_Area']=1
        df.loc[(df.Property_Area == 'Semiurban'),'Property_Area']=2
        
        
        #> ----Loan_Status--- (ONLY for Training set)
        ## - No: 0
        ## - Yes: 1
        if(not isTest):
            df.loc[(df.Loan_Status == 'N'),'Loan_Status']=0
            df.loc[(df.Loan_Status == 'Y'),'Loan_Status']=1

        #> -----Dependents-----
        #set max as 
        df.loc[(df.Dependents == '3+'), 'Dependents'] = 3
        ##----------------------------------------------------------------------------
        #### ----------------------------------Table----------------------------------
        ##----------------------------------------------------------------------------

        #!!! Typecase to float (for tensors below)
        df = df.astype(float)
        
        return df
    
    def train_split_test(self):
        # split into training and testing
        X = self.train_df.drop('Loan_Status',axis=1).values
        y = self.train_df['Loan_Status'].values
        # X_test = test_df.values

        ### Create tensors from np.ndarry main data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_train = torch.LongTensor(y_train).to(self.device)
        self.y_test = torch.LongTensor(y_test).to(self.device)

    def get_model(self):
        # seed the model for reproducibility
        torch.manual_seed(0)
        model = NN()
        model = model.to(self.device)
        return model
    
    def train(self, epochs=int(1e3), print_every=100, epsilon=0.5):
        """
        Train the model.
        - assumes access to following global variables: X_train, y_train, y_pred, model, loss function, & optimizer.
        @Param:
        1. epochs - number of training iterations.
        2. print_every - for visual purposes (set to None to ignore), outputs loss
        3. epsilon - threshold to break training.
        """
        start_time = time.time() #set start time
        losses = [] #plot
        
        for i in range(1, epochs+1):
            y_pred = self.model(self.X_train)
            loss = self.loss_function(y_pred, self.y_train)
            losses.append(loss)
            
            if(loss.item() <= epsilon):
                print(f"\nCONVERGED at epoch {i} - loss : {loss.item()}")
                break #converged
            
            if(print_every is not None and i%print_every == 1):
                print(f"Epoch {i} - loss : {loss.item()}")
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        print("total training time (sec):", time.time()-start_time)
        #graph cost
        return losses

    def accuracy(self, data, verification, pprint=False):
        """Based on X_test or X_train, predict overall model accuracy."""
        predictions=[]
        with torch.no_grad():
            for _, data in enumerate(data):
                y_pred = self.model(data)
                predictions.append(y_pred.argmax().item())

        predictions = np.array(predictions, dtype=np.int8)
        
        if(pprint): #print logs and graph
            loan = np.where(predictions == 1)[0]
            not_loan = np.where(predictions == 0)[0]
            print(f"Prediction loans count: {len(loan)}")
            print(f"Prediction not loans count: {len(not_loan)}")
            plt.hist(loan, label='Loan')
            plt.hist(not_loan, label='Not Loan')
            plt.legend()
            plt.xlabel("Observation")
            plt.ylabel("Frequency")
            plt.title("Histogram of loan vs. not loan observations")
            plt.show()
        score = accuracy_score(verification, predictions)
        return score

    def save_model(self):
        #Read more: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(self.model.state_dict(), "../models/loan_prediction.pth")


#------------------------------------------------------------------------------------------

#main model for loan prediction
class NN(nn.Module):
    def __init__(self, input_features=11, layer1=20, layer2=20, out_features=2):
        """Initialize the model for loan prediction"""
        super().__init__()
        self.fc1 = nn.Linear(input_features, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, out_features)
        
    def forward(self, x):
        """Forward pass with 11 input features"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x