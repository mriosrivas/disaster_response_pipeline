# Disaster Response Pipeline Project

This project consist in a web app where an emergency worker can input a message and get classification results in several categories. The web app also displays visualizations of the training data.

## Project Components

There three components for this project.

### 1. ETL Pipeline

This is a Python stored in  `data/process_data.py`where a data cleaning pipeline performs the following tasks:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### 2. ML Pipeline

This is a Python script called `models/train_classifier.py`in which a machine learning pipeline performs the following tasks:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 3. Flask Web App

This a folder called `app` where all the web applications is created. Namely there are two parts:

* `run.py`where the web server is created and served

* `templates` folder where all the static documents are stored

### 

### Instructions:

1. To run the ETL pipeline type the following code in the root directory:
   
   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

2. To run the ML pipeline type the following code in the root directory:
   
   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. To run the web app go inside the `app` folder and type:
   
   ```bash
   python run.py
   ```
   
   and then go to http://0.0.0.0:3001/

## Prerequisites

The following packages are required:

- `flask 2.1.3`
- `joblib 1.2.0`
- `nltk 3.7`
- `numpy 1.21.6`
- `pandas 1.2.4`
- `plotly 5.11.0`
- `regex 2.2.1`
- `sqlalchemy 1.4.39`
- `sklearn 1.1.1`

## Authors

- **Manuel Rios**
  
  [mriosrivas (Manuel Ríos) · GitHub](https://github.com/mriosrivas)

## License

This project is licensed under MIT licence.
