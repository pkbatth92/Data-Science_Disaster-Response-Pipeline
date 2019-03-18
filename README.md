# Disaster Response Pipeline Project

### Project Summary:
The objective of this project is to help the disaster response personnel to identify the message originating from the people who are in the most dire need of their services by categorizing the messages received into 36 categories.
The project makes use of supervised learning by training on the previously available data. It provides a web interface that lets the personnel to type in the message and the identify the categories that the message belongs to.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
