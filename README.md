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

### File description:
1. data/process_data.py
	- The script takes the file paths of the two datasets (categories and messages) and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.
    - It merges the messages and categories datasets, splits the categories column into separate columns, converts values to binary, and drops duplicates.

2. models/train_classifier.py
	- The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.
    - The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is then used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.
    - GridSearchCV is used to find the best parameters for the model that processes text and then performs multi-output classification on the 36 categories.
    
3. app/run.py
	- The app leverages the trained model to classify results for all 36 categories when a user inputs a message into the app.
    - It also shows three data visualizations describing the data from the SQLite database.
    