# Disaster Response Pipeline Project
This Machine Learning Pipeline Project is built to categorize emergency messages, based on training data from emergency messages of varied genres. Project is materialized as a Flask Web App

![Disaster Response message](https://github.com/erikjamesmason/Disaster_Response/blob/master/DR_DS1.png)
![Diaster Response Result](https://github.com/erikjamesmason/Disaster_Response/blob/master/DR_DS2.png)

## Table of Contents
1. Introduction
2. Requirements
3. Project Files
4. Guidance
5. Licensing

### Introduction
As a part of the Data Science Nanodegree from Udacity, this project seeks to display an understanding and utlization of Natural Language Processing within a Machine Learning Pipeline. The context of the dataset is emergency messages (disaster related). The project consists of the data, data wrangling, model buildling and model packaging, and the web app.

### Requirements

The Requirements.txt should equip the project for required packages, but general requirements are:
1. Pandas
2. Numpy.
3. NLTK
4. Scikit-Learn
5. SQLAlchemy
6. Plotly
7. Flask

### Project Files

Disaster_Response
│───DR_DS1.png
│───DR_DS2.png
│───README.md
│───requirements.txt 
│   
├───app
│   │───run.py
│   │   
│   └───templates
│       │───go.html
│       └───master.html
│           
├───data
│   │───disaster_categories.csv
│   │───disaster_messages.csv
│   │───Disaster_Response.db
│   └───process_data.py 
│                       
└───models
    │───classifier.pkl
    └───train_classifier.py

### Guidance

##### This project utilizes Python 3.8
##### Create Virtual environment
###### While you may other have methods to create a virtual environment, I found success with this approach:
1. `python3 -m venv test_env`
2. `dr_env\scripts\activate`
3. `pip3 install -r requirements`

##### From the project directory, run these commands
###### The script is designed to be ran dynamically, but you may need to experiment between changing directories

###### ETL Pipeline: Clean data and save to database
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db
###### ML Pipeline: 
python models/train_classifier.py data/Disaster_Response.db models/classifier.pkl
###### initiate and run webapp 
python run.py
###### Web App should be located at your local host - http://0.0.0.0:3001/

### Licensing

This corporate message data is from one of the free datasets provided on the Figure Eight Platform, licensed under a Creative Commons Attribution 4.0 International License.

Copyright 2020 Erik James Mason

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
