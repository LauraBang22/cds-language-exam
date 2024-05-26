#Import the relevant data
import os
import sys
sys.path.append("..")
import gensim.downloader as api
import pandas as pd
import argparse
from numpy import triu

#Define my argparse arguments for later use
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist",
                        "-a",
                        required = True)
    parser.add_argument("--search_term",
                        "-s_t",
                        required = True)
    args = parser.parse_args()
    return args

#Load the model we'll be working with
def load_model():
    model = api.load("glove-wiki-gigaword-50")
    return model

#Load the data we'll be working with
def load_data():
    filename = os.path.join("in", "Spotify Million Song Dataset_exported.csv")
    data = pd.read_csv(filename)
    data["artist"] = data["artist"].str.lower()
    return data

#A function that finds the most similar words, using the model, when given a word
def most_similar(model, args):
    most_similar_words = model.most_similar(args.search_term, topn = 5)
    words, similarity = zip(*most_similar_words)
    return words

#A function to find only the songs of a given artist
def select_artist(data, args):
    search_artist = args.artist.lower()
    search_songs = list(data[data["artist"]== search_artist]["text"])
    return search_songs

#A function that looks through all the songs of the particular artist and counts up if it contains one of the words
def relevant_songs(search_songs, words):
    song_counter = 0
    for song in search_songs:
        if any(word in song.split() for word in words): 
            song_counter += 1
    return song_counter

#Calculating the percentage of songs from the artist that contains the words and then prints it
def percent(song_counter, search_songs, args):
    percent = round(song_counter/len(search_songs)*100, 1)
    result = print(percent, "% of", args.artist,"'s songs contain words related to", args.search_term)
    return result

def main():
    args = parser()

    model = load_model()

    data = load_data()

    words = most_similar(model, args)

    search_songs = select_artist(data, args)

    song_counter = relevant_songs(search_songs, words)

    result = percent(song_counter, search_songs, args)


if __name__ == "__main__":
    main()

