#-------------------------------------------------------------
#--- Utility Functions pour l'Exploraty Data Analysis --------
#-------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import parent classes to create scikit Learn pipeline custom classes

from sklearn.base import BaseEstimator, TransformerMixin

#--------------------------------------------------------------------------------------

# une petite utility function pour analyser/visualiser une variable continue

def analyze_continuous(x):
    """Utility function pour décrire une variable contiue

    Args:
        x (np.array of float): la variable à analyser
    """
    
    fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
    # sns.set(rc = {'figure.figsize':(20,20)})
    
    sns.histplot(data=x, kde=True, ax=ax0)
    sns.boxplot(data=x,ax=ax1, orient="h")
    plt.show()
    
    print(f"counting {len(x)} values")
    print(f"moyenne = {np.mean(x)}")
    print(f"std dev = {np.std(x)}")
    print(f"mediane = {np.median(x)}")
    
#-----------------------------------------------------------------------------------    
    
def analyze_categorical(c):
    """Utility function pour analyser un tableau de variables categorical

    Args:
        c (np.array): tableau de variables catégorical
    """
    
    c_net = pd.DataFrame(c)

    fig, ax = plt.subplots(figsize=(6,6))
    ax = c_net.value_counts().plot.bar()
    print(c_net.value_counts())
    
#-----------------------------------------------------------------------------------

def plot_cat_v_cat(output_var,
                   input_vars,
                   ds):
    """Plots a series of boxplot of output_var vs each variable in cat_vars

    Args:
        output_var (string): name of the output variable
        input_vars (list of strings): name of the variables to plot the output variable from.
        ds (dataframe): the dataframe data.
    """
    
    n_axes = len(input_vars)
    
    # fig, axs = plt.subplots(nrows=1,
    #                        ncols=n_axes
    #                        )
    
    for i, input_var in enumerate(input_vars):
        # ax[i] = fig.add_subplot(1,n_axes,i+1)
        sns.barplot(x=input_var, y=output_var, data=ds)
        plt.show()
        
#-----------------------------------------------------------------------------------

def plot_cat_v_cont(output_var,
                   input_vars,
                   ds):
    """Plots a series of boxplot of output_var vs each variable in cont_vars

    Args:
        output_var (string): name of the output variable
        input_vars (list of strings): name of the variables to plot the output variable from.
        ds (dataframe): the dataframe data.
    """
      
    # fig, axs = plt.subplots(nrows=1,
    #                        ncols=n_axes
    #                        )
    
    for i, input_var in enumerate(input_vars):
        # ax[i] = fig.add_subplot(1,n_axes,i+1)
        sns.boxplot(y=input_var, x=output_var, data=ds)
        plt.show()
        
#-----------------------------------------------------------------------------------

def load_full_train_dataset():
    """Basic utility to provide a dataframe with the whole train.csv dataset
    
    Returns : Dataframe from the train.csv file
    """
    
    cwd = os.getcwd()
    filepath = cwd + '/titanic/train.csv'
    # print(filepath)

    ds = pd.read_csv(filepath)
    return ds

#-----------------------------------------------------------------------------------

def load_full_test_dataset():
    """Basic utility to provide a dataframe with the whole test.csv dataset
    
    Returns : Dataframe from the test.csv file
    """
    
    cwd = os.getcwd()
    filepath = cwd + '/titanic/test.csv'
    # print(filepath)

    ds = pd.read_csv(filepath)
    return ds

#----------------------------------------------------------------------------------

def display_null_nan_values(ds):
    """Basic display of NaN and null values in Dataframe ds

    Args:
        ds (Dataframe): the dataframe to look at
    """
    
    # check NaN values

    fig, ax = plt.subplots(figsize=(10,5))

    sns.heatmap(ds.isna(),ax=ax)
    ax.set_title('NaN values')
    plt.show()

    print(ds.isna().sum())
    
#--------------------------------------------------------------------------------------
    
# custom class to drop features

class FeatureDropper(BaseEstimator, TransformerMixin):
    """Custom class for pipeline to drop the 'PassengerId', 'Name', 'Ticket' and 'Cabin' columns from the dataset

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    
    # no need for a specific constructor
    
    # required methods : https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    def fit(self, X, y=None):
        return X  # we do not do anything during the fit step
    
    def transform(self, X):
        # this is where we drop the columns
        # X is assumed to be a panda dataframe
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        X.drop(columns=columns_to_drop, inplace=True)
        return X
    
#--------------------------------------------------------------------------------------

# custom class to one-hot-encode 'Sex' and 'Embarked'

class OHEColumnsTransformer(BaseEstimator, TransformerMixin):
    """Custom class to one-hot-encode 'Sex' and 'Embarked'

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    
    # no need for a specific constructor
    
    def fit(self, X, y=None):
        return X  # does nothing
    
    def transform(self, X):
        # assumes X is a dataframe with a 'Sex' and 'Embarked' columns
        # 'Embarked' can have NaN value (there are 2 in the train dataset)
        
        X.dropna(subset=['Embarked'], inplace=True)  # handles the two NaN values in the train dataset
        
        columns_for_ohe = ['Sex', 'Embarked']
        X = pd.get_dummies(data=X,
                           columns=columns_for_ohe,
                           sparse=False,
                           dtype=float
                           )
        
        return X

#--------------------------------------------------------------------------------------

# custom class to handle NaN values in 'Age' and 'Fare' columns

class AgeFareHandler(BaseEstimator, TransformerMixin):
    """custom class to handle the NaN values in the Age and Fare columns

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    
    # no need for a specific constructor
    
    def fit(self, X, y=None):
        return X  # does nothing
    
    def transform(self, X):
        # assumes X is a dataframe with an Age and Fare columns
        age_mean = X['Age'].mean()
        # print(f'age_mean = {age_mean}')
        fare_mean = X['Fare'].mean()
        
        values = {'Age':age_mean, 'Fare':fare_mean}
        X.fillna(value=values, inplace=True)
        
        return X

#--------------------------------------------------------------------------------------

# custom class to scale 'Age' and 'Fare' columns

class AgeFareScaler(BaseEstimator, TransformerMixin):
    """custom class to scale the Age and Fare columns

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    
    # no need for a specific constructor
    
    def fit(self, X, y=None):
        return X  # does nothing
    
    def transform(self, X):
        # assumes X is a dataframe with an Age and Fare columns
        age_mean = X['Age'].mean()
        age_std = X['Age'].std()
        # print(f'age_mean = {age_mean}')
        fare_mean = X['Fare'].mean()
        fare_std = X['Fare'].std()
        
        X['Age'] = (X['Age'] - age_mean) / age_std
        X['Fare'] = (X['Fare'] - fare_mean) / fare_mean
        
        return X

#--------------------------------------------------------------------------------------