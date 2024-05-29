import os
import sys
sys.path.append("..")
import gensim.downloader as api
import pandas as pd
import argparse
from codecarbon import EmissionsTracker

def parser():
    '''
    This function defines the argparse arguments, that will be usedd later
    It retuns the variable args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist",
                        "-a",
                        required = True)
    parser.add_argument("--search_term",
                        "-s_t",
                        required = True)
    args = parser.parse_args()
    return args

def load_model():
    '''
    This function roads the model used in this code.
    It returns the model.
    '''
    model = api.load("glove-wiki-gigaword-50")
    return model

def load_data():
    '''
    This function loads the data
    It transforms the everything in the "artist" column to lower case.
    It returns the transformed data.
    '''
    filename = os.path.join("in", "Spotify Million Song Dataset_exported.csv")
    data = pd.read_csv(filename)
    data["artist"] = data["artist"].str.lower()
    return data

def most_similar(model, args):
    '''
    This function finds the five most similar words to a searchterm given.
    It returns those words
    '''
    most_similar_words = model.most_similar(args.search_term, topn = 5)
    words, similarity = zip(*most_similar_words)
    return words

def select_artist(data, args):
    '''
    This function finds the songs of a given artist.
    It returns those songs.
    '''
    search_artist = args.artist.lower()
    search_songs = list(data[data["artist"]== search_artist]["text"])
    return search_songs

def relevant_songs(search_songs, words):
    '''
    This function looks through all the songs of a particular artist
    and counts if it contains one of the five similar words to the choosen searchterms.
    It returns the number of songs it has counted.
    '''
    song_counter = 0
    for song in search_songs:
        if any(word in song.split() for word in words): 
            song_counter += 1
    return song_counter

def percent(song_counter, search_songs, args):
    '''
    This function calculates the percentage of songs from the particular artist 
    that contains the words and the prints it.
    It also returns the result.
    '''
    percent = round(song_counter/len(search_songs)*100, 1)
    result = print(percent, "% of", args.artist,"'s songs contain words related to", args.search_term)
    return result

def main():
    tracker = EmissionsTracker(project_name="assignment3",
                            experiment_id="assignment3",
                            output_dir=os.path.join("..", "Assignment5", "emissions"),
                            output_file="emissions.csv") #define the emission tracker

    tracker.start_task("args")
    args = parser()
    args_emissions = tracker.stop_task()

    tracker.start_task("Load_model")
    model = load_model()
    model_emissions = tracker.stop_task()

    tracker.start_task("load_data")
    data = load_data()
    data_emissions = tracker.stop_task()

    tracker.start_task("most_similar")
    words = most_similar(model, args)
    most_similar_emission = tracker.stop_task()

    tracker.start_task("search_songs")
    search_songs = select_artist(data, args)
    search_songs_emissions = tracker.stop_task()

    tracker.start_task("song_counter")
    song_counter = relevant_songs(search_songs, words)
    song_counter_emissions = tracker.stop_task()

    tracker.start_task("results")
    result = percent(song_counter, search_songs, args)
    result_emissions = tracker.stop_task()    

    tracker.stop()

if __name__ == "__main__":
    main()

