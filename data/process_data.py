import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # function to load data into the dataframe
    # input: messages_filepath -> the path where the file with all the messages is located
    #        categories_filepath -> the path where the file with the corresponding categories is located
    # output: df -> dataframe with the messages and categories joined by message ids
    # loading messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # loading categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merging the two datasets to create a new dataframe combining messages with corresponding categories
    df = messages.join(categories.set_index('id'), on='id')
    
    return df

   

def clean_data(df):
    
    # function to do all the cleaning of the data i.e. breaking the categories column out into separate columns, one for each category, keeping just the 0/1 labels for each category and removing the deleting rows. 
    # input: df -> the dataframe with the messages and categories
    # output: df -> the clean dataframe
    # create a new dataframe of the 36 individual category columns
    categories = df[['categories']]['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[1,0:]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x[:-2]))
    
    # rename the columns of `categories` dataframe
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1 by going over each column at a time
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    
    # drop duplicates
    df = df[~df.duplicated(subset=None, keep='first')]
    
    return df


def save_data(df, database_filename):
    
    # funcction to store the cleaned dataframe in a SQLite database
    # input: df -> dataframe that contains the cleaned dataset
    #        database_filename -> path of the database where the dataset needs to be stored
    # create an engine from the sqlalchemy
    engine = create_engine('sqlite:///'+database_filename)
    
    # load the datafrmae into a database table
    df.to_sql('messages_categories', engine, index=False)  

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
       
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()