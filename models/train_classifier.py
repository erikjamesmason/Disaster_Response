import sys
import re 

import pandas as pd 
import numpy as np

import pickle 

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import make_multilabel_classification


def load_data(database_filename):
    print('sqlite:///{}'.format(database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = pd.read_sql_table('Categorized_Messages', engine)
    print(df.head(5))
    X = df.message
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, max_df=1.0)),
            ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_features': (5000, 10000),
        'tfidf__use_idf': (True, False), 
        'clf__estimator__estimator__dual': (True, False),
        
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)
    return model
    


def evaluate_model(model, X_test, y_test, category_names):
    """
    Shows model's performance on test data
    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification report
    for i,col in enumerate(category_names):
        print(f'col: {col}, {classification_report(y_test[col],y_pred[:,i])}')
    # print(classification_report(np.hstack(y_test),np.hstack(y_pred)))
    # print(classification_report(y_test.values, y_pred.values, target_names=category_names))
    labels = np.unique(y_pred)
    
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    # print accuracy score
    print('Accuracy: {}'.format(np.mean(y_test.values == y_pred)))
    


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filename))
        X, Y, category_names = load_data(database_filename)
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