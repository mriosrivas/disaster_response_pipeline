import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function extract two CSV datasets into a single dataframe by using their unique id to
    perform an inner join on them.
    :param messages_filepath: (String) path of the message dataset csv file
    :param categories_filepath: (String) path of the categories dataset csv file
    :return: (DataFrame) A merged dataframe with information of the messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, left_on='id', right_on='id')


def clean_data(df):
    """
    This function transform a dataframe provided by splitting the 'categories' column into 36 different columns,
    turn zeros and ones into integers and removes duplicates in the dataset.
    :param df: (DataFrame) A pandas dataframe generated by the load_data function.
    :return: (DataFrame) A pandas dataframe ready for the machine learning pipeline.
    """
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = [cat[:-2] for cat in categories.iloc[0].values]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories', 'id'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    df = df[df['related'] != 2].reset_index()

    return df


def save_data(df, database_filename):
    """
    This function loads into a SQLite database the processed data for future use in the machine learning pipeline.
    :param df: (DataFrame) A pandas dataframe transformed for future use in the machine learning pipeline.
    :param database_filename: (String) A string containing the path for the SQLite database to be stored.
    :return:
    """
    table_name = 'data'
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    return


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()