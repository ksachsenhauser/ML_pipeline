import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''load message and cetrgory data provided in a dataframe'''

    # load data from csv
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # merge data to result dataframe
    df_result = pd.merge(df_messages, df_categories, on='id', how='inner')

    return(df_result)


def clean_data(df):
    '''prepares categorical variables for machine learning and cleans duplicates'''

    # separate categories
    categories = df['categories'].str.split(pat=";", expand=True)

    # extract column names for one-hot-encoded cols
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # extract number from column values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # prepare result dataframe by introducing new categorical encoding
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1, sort=False)

    # clean duplicates
    df.drop_duplicates(inplace=True)

    # clean "related = 2" data issue
    df = df[df.related != 2]

    return(df)


def save_data(df, database_filename):
    '''stores cleaned data to sqlite database provided'''

    # open database and write data
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

    return None


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