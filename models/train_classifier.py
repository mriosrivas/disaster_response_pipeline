import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def load_data(database_filepath):
    """
    This function returns X and Y values from a local database stored in database_filepath.
    :param database_filepath: (String) Location of the SQLite database.
    :return: X (Numpy Array) Array of numbers representing the complete dataset.
    :return: Y (Numpy Array) Array of numbers representing the labels on the dataset.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('data', engine)
    df = df[df['related'] != 2]
    X = df['message'].values
    Y = df[list(df.columns)[-36:]]
    category_names = list(df.columns[-36:])
    return X, Y, category_names


def tokenize(text, verbose=False):
    """
    This function normalizes, tokenizes, removes stop words and lemmatizes a sentence.
    :param text: (String) A string of words that defines the sentence to tokenize.
    :param verbose: (Boolean) A boolean value to print url replacements in console.
    :return: (List) A list of tokens of a tokenized text.
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    url_regex_merge = 'http[s]*\s:\s//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_regex_replace = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex_merge, text)
    for url in detected_urls:
        text = text.replace(url, url.replace(' ', ''))
        if verbose:
            print(f"Original url = {url}")
            print(f"New url = {url.replace(' ', '')}")

    detected_urls = re.findall(url_regex_replace, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # 1. Normalize
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()

    # 2. Tokenize
    tokens = word_tokenize(text)

    # 3. Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens if t not in stop_words]

    return tokens


def build_model():
    """
    This function creates a pipeline model with a CountVectorizer, a TfidfTransformer and a MultiOutputClassifier and
    uses GridSearchCV cross validator to select the best model given the specified parameters dict.
    :return: (GridSearchCV) A GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'moc__estimator__min_samples_split': [2, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates a given model with a X_test and Y_test sets.
    :param model: (Pipeline) An sklearn pipeline object with a predict method.
    :param X_test: (Array) An array of sentences to transform with a CountVectorizer and a TfidfTransformer.
    :param Y_test: (DataFrame) A dataframe with 36 different categories.
    :param category_names: (List) A list of names of the 36 different categories of Y_test.
    :return:
    """
    y_pred = model.predict(X_test)

    for k, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col].values, y_pred[:, k]))
        print('-------------------------------------------------------')
    return


def save_model(model, model_filepath):
    """
    This function serializes and saves a model into a pickle file.
    :param model: (Pipeline) An sklearn pipeline object with a predict method.
    :param model_filepath: (String) Location of to store the pickle file.
    :return:
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model.best_estimator_, file)
    return


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
