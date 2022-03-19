import sys
import re
import pickle
import pandas as pd 
from sqlalchemy import create_engine
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
    This function is to connect to database file and read messages table
    INPUT:
    database_filepath - database file path to read
    OUTPUT:
    X - Features dataframe
    Y - prediction target dataframe
    category_names - list of prediction target columns names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages',con=engine)
    X = df['message']
    Y = df.iloc[:, 4:].apply(pd.to_numeric)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    INPUT:
    text - text to be toxenized
    OUTPUT:
    lemmed - list of words after normalized, tokenized, lemmatized
    """
    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize
    words = word_tokenize(text)
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    """
    This function is to drfine pipline and adjust parameters using GridSearchCV
    OUTPUT:
    cv - resulted model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    
    parameters = {
        'clf__estimator__n_jobs': [1, 2],
        'clf__estimator__n_estimators': [5, 10]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
#     cv = pipeline
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This functin is to print classification_report and model accuracy
    INPUT:
    model - model to be evaluated
    x_test - dataframe of features for test
    y_test - dataframe of prediction target for test 
    category_names - prediction target columns names
    """
    y_pred = model.predict(X_test)
    for column in Y_test:
        print('Feature {} :'.format(column))
        print(classification_report(Y_test[column], y_pred[:, Y_test.columns.get_loc(column)]))
    print('The model accuracy is {}'.format((y_pred == Y_test.values).mean()))



def save_model(model, model_filepath):
    """
    This function is to save model as pkl file
    INPUT:
    model - model we want to save
    model_filepath - model file path and name we want to save at
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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