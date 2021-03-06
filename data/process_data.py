import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def split_name(text, index, delimiter):
    """
    INPUT:
    text - text with delimiter to be splitted
    index - the position of text to be returned
    delimiter - the delimter to split twxt based on
    OUTPUT:
    text on spicific index after splitting
    """
    return text.split(delimiter)[index]


def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
    messages_filepath - file path for messages csv file
    categories_filepath - file path for categories csv file
    OUTPUT:
    df - dataframe contains merged data from two files
    """
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath) 
    df = messages.merge(categories, how="left", on="id")
    
    return df




def clean_data(df):
    """
    INPUT:
    df - messages - categories dataframe to be splitted and cleaned
    OUTPUT:
    df - dataframe after cleaning
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(split_name, args=(0, '-', )).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(split_name, args=(1, '-', ))
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # drop the original categories column from `df`
    df.drop("categories", axis='columns', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.related = df.related.replace(2, 1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    INPUT:
    df - cleaned dataframe to be saved to database
    database_filename - database file name to be created
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        # print(df.related.value_counts())
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
