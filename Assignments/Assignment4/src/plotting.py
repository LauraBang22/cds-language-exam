from transformers import pipeline

import os
import sys
sys.path.append("..")
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

def load_data():
    '''
    This function loads the data, the code will be running on.
    It returns the data.
    '''
    filename = os.path.join("out", "labels.csv")
    data = pd.read_csv(filename)
    return data

def plot_emotion_distribution(data):
    '''
    This function plots the emotion distribution in each season.
    It saves the results in the "out" folder
    '''
    #create lists over unique seasons and emotion labels in the data
    unique_seasons = data['Season'].unique()
    unique_labels = data['labels'].unique()

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Distribution of all emotion labels in that season", fontsize=20, y=1.3)
    axs = axs.flatten()

    for idx, season in enumerate(unique_seasons):
        season_data = data[data['Season'] == season]
        label_counts = season_data['labels'].value_counts()
        axs[idx].bar(label_counts.index, label_counts.values)
        axs[idx].tick_params(axis='x', which='major', rotation=45)
        axs[idx].set_title(season)

    plt.tight_layout()  
    plt.savefig("out/emotion_distribution.png")

def plot_relative_frequency(data):
    '''
    This function plots the relative frequency of an emotion across all seasons.
    It saves the results in the "out" folder.
    '''
    unique_seasons = data['Season'].unique()
    unique_labels = data['labels'].unique()

    for label in unique_labels:
        relative_freq_list = []
        for season in unique_seasons:
            label_data = data[(data["labels"] == label)&(data["Season"] == season)]
            relative_freq = len(label_data)/len(data[data["Season"]== season])*100
            relative_freq_list.append(relative_freq)

        plt.figure(figsize=(16,6))
        plt.plot(unique_seasons, relative_freq_list)
        plt.title(label)
        plt.xlabel("Season")
        plt.ylabel("Relative frequencies")

        plt.savefig("out/" + label + ".png")

def main():
    tracker = EmissionsTracker(project_name="assignment4_plotting",
                               experiment_id="assignment4_plotting",
                               output_dir=os.path.join("..", "Assignment5", "emissions"),
                               output_file="emissions.csv")

    tracker.start_task("load_data")
    data = load_data()
    data_emissions = tracker.stop_task()

    tracker.start_task("emotion_distribution")
    plot_emotion_distribution(data)
    emotion_distribution_emission = tracker.stop_task()

    tracker.start_task("relative_frequency")
    plot_relative_frequency(data)
    relative_frequency_emission = tracker.stop_task()

    tracker.stop() 

if __name__ == "__main__":
    main()
