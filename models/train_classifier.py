import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report




def load_data(database_filepath):
    '''loads data from database given and returns X, Y and category name data'''

    #connect to database and retrieve data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)

    # separate data in target, features, feature names
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    cat_names = Y.columns

    return(X,Y,cat_names)


def tokenize(text):
    '''extracts words from english message given and provides clean tokens'''
    
    # replace non letter or non number characters 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # get tokens from text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # clean and lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return(clean_tokens)


def build_model():
    ''' builds and returns a model pipeline '''

    # setup ML-pipeline with a RandomForestClassifier
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier())),
    ])

    return(pipeline)


def evaluate_model(model, X_test, Y_test, category_names):
    '''provides classification data on categorical features given'''

    # predict test data with model
    y_pred = model.predict(X_test)

    # iterate through categorical columns for accuracy data
    y_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
    for col in Y_test.columns:
        print("Feature: "+ col)
        print(classification_report(Y_test[col],y_pred[col]))

    return(None)


def save_model(model, model_filepath):
    '''saves model to filepath as pickle file'''

    # open filepath and dump model 
    with open(model_filepath, 'wb') as exp_file:
        pickle.dump(model, exp_file)

    return(None)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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