# Disaster Response Pipeline Project

##Project Description:
In this project, a pipeline for data transformation with machine learning was developed. This pipeline was capable of filtering message. The pipeline will eventually be turned into a flask app. A web app is included in the project, which allows an emergency worker to enter a new message and receive classification results in different categories. The webapp's landing page also provides plotly-created visualizations of the training dataset.

##File Descriotions:
The following files are included in the project:
data/process data.py: This file contains the extract, transform, and load processes for preparing data for model construction.
data/disaster_categories.csv and data/disaster_messages.csv: these files contains data 
models/train_classifier.py: This file contains Fitting, tuning, evaluating, and exporting the model as a Python pickle using the Machine Learning pipeline.
app/templates/~.html: The web app's HTML pages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
