from transformers import pipeline

import os
import sys
sys.path.append("..")
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from tqdm import tqdm

def load_model():
    '''
    This function loads the classifier the code uses on the data.
    It returns the classifier.
    '''
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=False)
    return classifier

def load_data():
    '''
    This function loads the data, that the code will run on.
    It returns the data.
    '''
    filename = os.path.join("in", "Game_of_Thrones_Script.csv")
    data = pd.read_csv(filename)
    data.dropna(inplace=True) #dropping rows with missing data
    return data

def generate_labels(classifier, data):
    '''
    This function generates an emotion label for each row in the data.
    It returns a list of labels. 
    '''
    labels = []
    for line in tqdm(data["Sentence"]):
            label = classifier(line)
            labels.append(label[0]["label"])
    return labels

def add_labels(labels, data):
    '''
    This function creates a new dataframe that contains two columns.
    One called season and one with the labels.
    That is all that is needed for calculating what is needed in the assignment.
    It returns the new dataframe.
    '''
    data["labels"] = labels
    simple_data = data.loc[:, ["Season", "labels"]]
    return simple_data

def save_data(simple_data):
    '''
    This function saves the new dataframe in the "out" folder.
    '''
    outpath = os.path.join("out", "labels.csv")
    simple_data.to_csv(outpath)

def main():
    tracker = EmissionsTracker(project_name="assignment4_data",
                           experiment_id="assignment4_data",
                           output_dir=os.path.join("..", "Assignment5", "emissions"),
                           output_file="emissions.csv")

    tracker.start_task("load_model")
    classifier = load_model()
    load_model_emissions = tracker.stop_task()

    tracker.start_task("load_data")
    data = load_data()
    load_data_emissions = tracker.stop_task()

    tracker.start_task("generate_labels")
    labels = generate_labels(classifier, data)
    generate_labels_emissions = tracker.stop_task()

    tracker.start_task("simple_data")
    simple_data = add_labels(labels, data)
    simple_data_emissions = tracker.stop_task()

    tracker.start_task("save_data")
    save_data(simple_data)
    save_data_emissions = tracker.stop_task()

    tracker.stop()

if __name__ == "__main__":
  main()
