import spacy
import pandas as pd
import os
import re
from codecarbon import EmissionsTracker

def load_model():
    nlp = spacy.load("en_core_web_md")
    return nlp

def data_processing(nlp):
    main_folder_path = ("in/USEcorpus") #the folder that we will be working in
    sorted_dir = sorted(os.listdir(main_folder_path)) #sorting all the subfolders

    for folder in sorted_dir: #creating a "for loop" to reach all the subfolders
        folder_path = os.path.join(main_folder_path, folder) 
        filenames = sorted(os.listdir(folder_path)) #sorting all the files in the different subfolders
        folder_info = [] #Define a empty list for later use
        
        for filename in filenames: #creating a new "for loop" to reach all the files in the subfolders
            filepath = folder_path + "/" + filename
            
            with open(filepath, encoding="latin-1") as f: 
                text = f.read() #opening all the files in one text
                
            pattern = r"<.*?>" #cleaning the text
            cleaned_text = re.sub(pattern, "", text)

            doc = nlp(cleaned_text)
            
            #Define four variables, where the count of the different POS are 0        
            noun_count = 0
            verb_count = 0
            adj_count = 0
            adv_count = 0
            
            #"for loop", where for each token of the specific POS, then it'll andd one to the counter.        
            for token in doc:
                if token.pos_ == "NOUN":
                    noun_count += 1
                elif token.pos_ == "VERB":
                    verb_count += 1
                elif token.pos_ == "ADJ":
                    adj_count += 1
                elif token.pos_ == "ADV":
                    adv_count += 1
                    
            #Relative frequency of each of the POS
            relative_freq_noun = round((noun_count/len(doc)) * 10000, 2)
            relative_freq_verb = round((verb_count/len(doc)) * 10000, 2)
            relative_freq_adj = round((adj_count/len(doc)) * 10000, 2)
            relative_freq_adv = round((adv_count/len(doc)) * 10000, 2)
            
            persons = set() #First create a new set
            for ent in doc.ents:
                if ent.label_ == 'PERSON': #Find all the entities in the doc that has the label PERSON
                    persons.add(ent.text) #Then add them to the set cwe created
            num_persons = len(persons) #This is how we find out how many unique PERSON the previous code has found
            
            organisations = set()
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    organisations.add(ent.text)
            num_organisations = len(organisations)
            
            locations = set()
            for ent in doc.ents:
                if ent.label_ == 'LOC':
                    locations.add(ent.text)
            num_locations = len(locations)
            
            #Save the results               
            file_info = [filename, relative_freq_noun, relative_freq_verb, relative_freq_adj, relative_freq_adv, 
                        num_persons, num_organisations, num_locations] 
            
            folder_info.append(file_info)
            
            df = pd.DataFrame(folder_info, 
                            columns=["Filename", "RelFreq NOUN", "RelFreq VERB", "RelFreq ADJ", "RelFreq ADV", 
                                    "Unique PER", "Unique LOC", "Unique ORG"])


            outpath = os.path.join("out", folder + ".csv")
            df.to_csv(outpath)

def main():
    tracker = EmissionsTracker(project_name="assignment1",
                        experiment_id="assignment1",
                        output_dir=os.path.join("..", "Assignment5", "emissions"),
                        output_file="emissions.csv")
    
    tracker.start_task("load_model")
    nlp = load_model()
    model_emissions = tracker.stop_task()

    tracker.start_task("data_processing")
    data_processing(nlp)
    processing_emissions = tracker.stop_task()

    tracker.stop()

if __name__ == "__main__":
    main()