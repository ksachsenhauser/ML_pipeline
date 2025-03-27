# ML_pipeline
Implementation of a machine learning processing of disaster response messages. The software focus is on cleaning messages and analyze through ML - mechanisms. A useful part in a overall toolchain for message priority to support emergency response teams

# Installation
The code is based on python. Development was made under Python 3.12, which is the recommended version, although lower versions are also likly to work. Make sure to have data analytics and machine learning packages like pandas, numpy, scikit learn along with natural language toolkit nltk available. Messages are stored in an SQLite database, the sqlite package is also necessesary. There are also jupyter notebooks with the code avaialble for a more convenient work with the code. If you want to access these notebooks, jupyter version 7.2 is recommended. If you want to run the web app for prompting ML - model, you will also need flask avaialble.

# Running the project
To run the code in this repository, run the following commands:

<b>Prepare data via ETL</b><br>
Change to data folder:<br>
python process_data.py messages.csv categories.csv DisasterResponse.db

<b>Build machine learning model on data in database</b><br>
Change to models folder:<br>
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

<b>Run the web app for visualisation and classification of messages:</b><br>
Change to app folder:<br>
python run.py


# Project
This project serves two purposes. First it is supposed to support improvements in automatic priority assignment for message streams. Additionally there is an educational purpose as this project serves to improve my data handling and machine learning experience. Hopefully, one of these purposes may also apply to you.

# Files
<b>ETL Pipeline Preparation.ipynb</b><br>
Jupyter notebook to read messages and categories form .cvs files, cleans data and writes to a SQLite db

<b>ML Pipeline Preparation.ipynb</b><br>
Jupyter notebook to read data from SQLite db, setup machine learning pipelines and check results from training

<b>data folder</b><br>
<b>process_data.py</b><br>
Python script containing the logic from <b>ETL Pipeline Preparation.ipynb</b>

<b>categories.csv and messages.csv</b><br>
Raw data text files with messages and categories to be processed by process_data.py

<b>DisasterResponse.db</b><br>
SQLite database to store processed data 

<b>models folder</b><br>
<b>train_classifier.py</b><br>
Python script wit logic to setup data retrieval, ml pipeline and training for data in <b>DisasterResponse.db</b>. Stores model in a pickle file in this folder.

<b>app folder</b><br>
<b>run.py</b><br>
Python script to run a web app for prompting machine learning model

<b>templates folder</b><br>
Web templates used for screens in <b>run.py</b>

# For additional information
In case you need additional information, you may contact me via my profile information here on github.

# Acknowlegements
This data was provided by Appen (formerly Figure Eight) via Udacity for educational purposes, which is also the purpose you may use this code for.
