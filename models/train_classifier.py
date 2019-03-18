import sys
import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):
    
    # this function helps load data into a dataframe and return the raw feature, labels and label names
    # input: database_filepath -> name of the filepath
    # output: X -> the column that contains the raw feature set
    #         Y -> the columns that will serve as our labels
    #         category_names -> names of the labels / categories
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    connection = engine.connect()
    df = pd.read_sql_table('messages_categories', con=connection)
    
    # Extracting the X and Y columns from the dataframe: X -> messages which will help build our feature set
    # and Y -> the 36 categoris that will serve as our labels
    X = df.message.values
    Y = df.drop(['id','message','original','genre'], axis = 1)
    category_names = list(Y.columns.values)
    Y = Y.values
    
    return X, Y, category_names

class MessageLength(BaseEstimator, TransformerMixin):
    
    # class for custom transformer that returns the length of each message
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_length = pd.Series(X).apply(lambda x: len(x)).values

        return pd.DataFrame(X_length)

def tokenize(text):
    
    # this function helps tokenize our raw messages into clean tokens that can be ingested by our model as features
    # input: text -> each message inputted as an array one at a time
    # output: clean_tokens -> the array containing the clean tokens generated out of our raw messages
    
    # creating tokens using work_tokenize function and iniitializing the WordNetLemmatizer class
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    # function that sets up the pipeline including the data transformation steps,  the model building steps and then performs the grid search cross validation to find the best set of features.
    # output: cv_model -> the generated model
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('length_message', MessageLength())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_df': (0.50,0.75,1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__smooth_idf': (True, False),
        'clf__estimator__min_samples_leaf': [1, 3, 5],
    }
    
    cv_model = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv_model
    #return pipeline
   
    
def evaluate_model(model, X_test, Y_test, category_names):
    
    # function to evaluate the performance of the model
    # input: model -> the generated model
    #        X_test -> the test feature set
    #        Y_test -> the test label set
    #        category_names -> the names of the labels
    
    Y_pred = model.predict(X_test)
    
    # trnasposing the Y_test and Y_pred arrays 
    Y_pred = np.asarray([*zip(*Y_pred)])
    Y_test = np.asarray([*zip(*Y_test)])
    
    [print('CATEGORY: '+category_name + '......\n\n' + classification_report(Y_i, Y_j)) for (Y_i, Y_j, category_name) in zip(Y_test, Y_pred, category_names)] 
    

def save_model(model, model_filepath):
    
    # function to save the final model in a specified filepath
    # input: model -> the final model
    #        model_filepath -> the path to store the model
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        #print(model.get_params())
        #sys.exit(0)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()