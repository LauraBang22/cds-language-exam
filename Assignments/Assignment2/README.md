# Assignment 2 - Text classification benchmarks

### Assignment Discription
In this assignment I have solved the following tasks for the given data:
- Save the classification report to a text file the folder called out
- Save the trained models and vectorizers to the folder called models

### Repository Structure
In this repository you'll find four subfolders.
- In ```in``` you'll find the data that is being used in the code.
- In ```models``` you'll find the different models I have created to run on the data.
- In ```out``` you'll find the results the code have produced.
- In ```src``` you'll find the scripts of code written to solve the tasks given in the assignment.

I have also created a requirements.txt and a setup.sh file for you to run, for the setting up a virtual environment to run the code in.

### Data
The data I have used in this assignment, is called **Fake News Dataset**. The data can be found [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). You'll need to download the dataset and unpack it. When you unpack it you have a folder called **archive**, within that folder is a **.csv** file. That is the file you need to move it to the ```in``` folder in this repository.


### Reproducebility 
For this code to work, you need to be placed in the **Assignment2** folder in your terminal.

I have created a ```setup.sh``` file that can be run from the terminal using the code: 
```
bash setup.sh
``` 
When running it you create a virtual environment where you run the accompanying ```requirements.txt```. 

I have for this assignment created three different ```run.sh``` files that can be run from the terminal using the code:
```
bash run_logistic_regression.sh
bash run_neural_network.sh
bash run_vectorizer.sh
```
Each file opens the virtual environment again, then runs one of the scripts that I have written for this assignment, and finishes off by deactivating the virtual environment. 

### Results
#### Logistic Regression
When running the logistic regression model on the data, it gives an accuracy of 83% in the classification report. That is very good. However when you look at learning curves of the cross validation, it shows that the curves are quite far apart, which might indicate that there is a problem of the model overfitting. Something that might help fix that problem is if we had more data to train it on. 

#### Neural Network
When running the neural network model on the data, it  gives an accuracy of 84%, which is one percent better than the logistic regression model.
At first glance the loss curve for the model looks very good. It decreases and stabilizes, which is good. I don't have something to compare it to, which makes it difficult to tell is the model is either underfitting or overfitting. That is something that could be worked on adding to the code. 

#### CodeCarbon
I have used the package ```CodeCarbon``` to measure the environmantal impact of running the code in this repository. Please see Assignment 5 for the results of that.