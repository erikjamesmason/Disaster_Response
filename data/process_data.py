import sys
import pandas as pd
import numpy as np
import sqlalchemy 

def load_data(messages_filepath, categories_filepath):
    """
    Function to load in data from csv and merge on 'id' to
    prepare for cleaning.
    requires the messages and categories .csv files located in data directory
    """
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str[0] 
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    categories['related'] = categories['related'].replace(2, 1)
        
    df = df.drop('categories', axis=1)
        
    df = pd.concat([df,categories], axis=1, sort=False)
        
        # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    print('sqlite:///{}'.format(database_filename))
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename)
    df.to_sql('Categorized_Messages', engine, if_exists='replace', index=False) 


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