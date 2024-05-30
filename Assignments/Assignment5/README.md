# Assignment 3 - Query expansion with word embeddings

### Assignment Discription
In this assignment I have used CodeCarbon to measure the environmental impact of the code of the first four assignments. And answered the following questions:

- Which assignment generated the most emissions in terms of CO₂eq? Explain why this might be.
- Which specific tasks generated the most emissions in terms of CO₂eq? Again, explain why this might be.
- How robust do you think these results are and how/where might they be improved? 

### Repository Structure
In this repository you'll find three subfolders:
- One called ```emissions```. Within that folder, is another folder called ```examples```. It contains the **.csv** files I have generated when running the code from the first four assignments. It is the files I have used to generate plots in this assignments.
- One called ```out```, where you'll find saved **.png** files with all the plots I have created for this assignment.
- one called ```src```, where you can find the **jupyter notebook** I have created the code in, for this assignment.

For this assignment I have created a jupyter notebook, where  I have written my code. I have done that, instead of writing a script, because I found that it made more sense to have the code, the resulting plots and my discussion of it all gathered in one place.

I have also created a ```requirements.txt``` and a ```setup.sh``` file for you to run, for the setting up a virtual environment to run the code in.

### Data
The data I have worked with in this assignment, is estimations of emission of CO₂eq in my other assignments. It has been generated using **CodeCarbon**, which you can read more about [here](https://codecarbon.io/). 

### Reproducebility 
For this code to work, you need to be placed in the **Assignment5** folder in your terminal.

I have created a ```setup.sh``` file that can be run from the terminal using the code: 
```
bash setup.sh
``` 
When running it you create a virtual environment where you run the accompanying ```requirements.txt```.

To open the created virtual environment for the jupyter notebook, you need to first open the notebook. In the top right corner there is a button, with the text "select kernel". You need to press that. A pop-up will apear, where you need to press "Jupyter kernel...". To find the right kernel, you first need to press the reload button for the pop-up, and the you should be able to select the kernel called **"env (Python 3.12.3)"**

### Results
Overall it is assignment 4 that emits the most CO₂eq. But the common theme is that though out the assignments it is the tasks, that either handles large datasets or big models that generates much emission. It especially when the models are being used on the data, which requires a lot of calculations, that it creates much emission.

While the numbers are still very small, it is important to keep in mind, that they are just for running the code ones. I must admit that I have lost count of how many times I have run the different script and parts of code, when creating these assignments. That means that all the different runs quickly generates much more emission, and that is maybe something to keep in mind, for future coding. How much and often do we actually need to test our code, to get it to work? Is there a way to limit that? I don't know, but I think as long as we are mindful of it, then that is a good start. 

I have found this particulr task quite thought provoking, because I had not considered the environmental impact of what it was that we are doing, before being asked to, but of course it makes sense that all of the requiered computing power we have been using would have had some kind of environmental impact.
