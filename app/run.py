import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/Disaster_Response.db')
df = pd.read_sql_table('Disaster_Response', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # news   
    news_df = df.loc[df['genre']=='news']
    news_df = news_df.iloc[:, 4:].sum().sort_values(ascending=False)
    news_counts = news_df.values
    news_names = news_df.index
    # direct   
    direct_df = df.loc[df['genre']=='direct']
    direct_df = direct_df.iloc[:, 4:].sum().sort_values(ascending=False)
    direct_counts = direct_df.values
    direct_names = direct_df.index
    # social   
    social_df = df.loc[df['genre']=='social']
    social_df = social_df.iloc[:, 4:].sum().sort_values(ascending=False)
    social_counts = social_df.values
    social_names = social_df.index
                     

    categ_count = df.iloc[:, 4:].sum().sort_values(ascending=False)
    categ_counts = categ_count.values
    categ_names = categ_count.index
    
    request_count = df.loc[df['request']==1]
    offer_count = df.loc[df['offer']==1]
    
    request_df = request_count.iloc[:, 4:].sum().sort_values(ascending=False)
    request_df_counts = request_df.values
    request_df_name = request_df.index
    
    offer_df = offer_count.iloc[:, 4:].sum().sort_values(ascending=False)
    offer_df_counts =  offer_df.values
    offer_df_name =  offer_df.index
  
    
    # create visuals
    graphs = [
        # categories break-out
        {
            'data': [
                Bar(
                    y=categ_counts,
                    x=categ_names
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Categories',
                'yaxis': {
                    'title': 'Category Count'
                },
                'xaxis': {
                    'tickangle': 45,
                    'title_standoff': 60
                },
                
            }
        },
        
        {
            'data': [
                Bar(
                    name='News',
                    x=news_names,
                    y=news_counts
                ),
                Bar(
                    name='Direct',
                    x=direct_names,
                    y=direct_counts
                ),
                Bar(
                    name='Social',
                    x=social_names,
                    y=social_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': 45
                },
                'barmode': 'stack'
            }
        },
        
        
        # request vs offers distribution plot
        {
            'data': [
                Bar(
                    name='Requests',
                    x=request_df_name,
                    y=request_df_counts
                ),
                Bar(
                    name='Offers',
                    x=offer_df_name,
                    y=offer_df_counts
                )
            ],
            

            'layout': {
                'title': 'Distribution of Disaster Categories for Offers and Requests',
                'yaxis': {
                    'title': 'Category Count'
                },
                'xaxis': {
                    'tickangle': 45
                },
                'barmode': 'stack'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()