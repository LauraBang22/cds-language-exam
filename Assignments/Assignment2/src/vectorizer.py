import os
import sys
sys.path.append("..")

import pandas as pd
import utils.classifier_utils as clf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from codecarbon import EmissionsTracker
from joblib import dump, load

def create_vectorizer():
    '''
    This function defines and created a vectorizer model.
    It returns the vectorizer.
    '''
    vectorizer = TfidfVectorizer(ngram_range = (1,2),
                            lowercase = True,
                            max_df = 0.95,
                            min_df = 0.05,
                            max_features = 100)
    return vectorizer

def save_vectorizer(vectorizer):
    '''
    This function saves the model in the models folder.
    '''
    dump(vectorizer, os.path.join("models","tfidf_vectorizer.joblib"))

def main():
    tracker = EmissionsTracker(project_name="assignment2_vectorizer",
                        experiment_id="assignment2_vectorizer",
                        output_dir=os.path.join("..", "Assignment5", "emissions"),
                        output_file="emissions.csv") #defines the emissions tracker

    tracker.start_task("create_vectorizer")
    vectorizer = create_vectorizer()
    tracker.stop_task()

    tracker.start_task("save_vectorizer")
    save_vectorizer(vectorizer)
    tracker.stop_task()

    tracker.stop()

    

if __name__ == "__main__":
    main()