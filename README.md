# KNN-ML
ML on a ramdom dataset having a target class to be predicting.

EDA using seaborn pairplot

Standardizing the data set before creating the model to avoide Noise in data, for this using the
 StandardScaler from preproccessing of the sklearn.
 
Using Scikitlearn for ML.
Using the Train Test Split for splitting the x test data as 20% of the dataset.
 
 Using for loop for iterating over diff values of k, and getting diff predictions each time.
   Then addding these predictions to a error list by taking the mean i.e mean of predictions that
    are not equal to the y test data.\
     
 Finally ploting the graph of error rate vs k values(using Matplotlib)and optimixzxing the model by choosing the
    appropriate k value.
