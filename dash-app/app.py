# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from scipy import sparse
import time
from predict import parse_output_file, make_all_predictions, get_top_k

def generate_table(dataframe, max_rows=20):
    return html.Table(
        # Header
        [html.Tr([html.Th(col, style={'textAlign': 'center'}) for col in dataframe.columns], 
            style={'padding': '0px 12px'})] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col], 
                style={'textAlign': 'center', 'padding': '6px 12px'}) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
    style={'marginLeft': 'auto', 'marginRight': 'auto', 'marginTop': 25, 'marginBottom': 25})

app = dash.Dash()
server = app.server

app.layout = html.Div(children=[
    html.H2(children='What Should I Read Next?', style={'marginBottom': '12px'}),

    html.Div(children=[
        html.P("Hello there! First of all, who are you? You can play as one of us, or pick a random user."),
        dcc.RadioItems(
                id='user-choice',
                options=[{'label': i, 'value': i} for i in ['Adam', 'Marika', 'Mark', 'Steph', 'Random']],
                value='Random',
                labelStyle={'display': 'inline-block', 'padding': 5}
            )
        ]),

    html.Div(id='recs-table', 
        style={'width': '60%', 'marginLeft': 'auto', 'marginRight': 'auto'}),

    html.Div(children=[
        html.P("When you're ready to calculate your rating-based recommendations, press Submit! \
            Please note it may take 15 seconds or so to load."),

        html.Button(id='submit-button', n_clicks=0, children='Submit', 
            style={'marginTop': 15, 'marginBottom': 25}),

        html.P(children="Want to customize your recommendations? Try adjusting the weights for \
            some popular genres. Move a genre's slider to the right if you want more of \
            it, and to the left if you want less. When you're ready, hit the Submit button again.")
        
        ], style={'marginLeft': 'auto', 'marginRight': 'auto', 'width': '60%'}),

    html.Div(children=[
        
        html.Div(children=[

            html.Div(children=[
                dcc.Slider(id='scifi', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),
            
            html.Div(children=[
                dcc.Slider(id='mystery', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='romance', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='historical', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='comics', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='children', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 5})
        
        ], className='three columns', style={'marginLeft': '9%', 'marginRight': 10}),

        html.Div(children=[
            html.P('Science Fiction/Fantasy'),
            html.P('Mystery'),
            html.P('Romance'),
            html.P('Historical Fiction'),
            html.P('Comics'),
            html.P('Children')
        ], className='three columns', 
        style={'textAlign': 'left', 'width': '18%', 'margin': 0}),

        html.Div(children=[
            html.P('Science'),
            html.P('Business'),
            html.P('Art/Music'),
            html.P('Biography'),
            html.P('History'),
            html.P('Religion/Philosophy')
        ], className='three columns', 
        style={'textAlign': 'right', 'width': '18%', 'margin': 0}),
        
        html.Div(children=[

            html.Div(children=[
                dcc.Slider(id='science', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),
            
            html.Div(children=[
                dcc.Slider(id='business', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='art', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='biography', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='history', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='religion', min=-2, max=2, value=0, step=0.5)
            ], style={'marginTop': 5, 'marginBottom': 5}),
        
        ], className='three columns', style={'marginLeft': 10, 'marginRight': '9%'}),
    
    ], 
    style={'marginTop': 20, 'width': '100%'},
    className='row')

], style={'textAlign': 'center'})

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

features = sparse.load_npz('model_features.npz').tocsc() # model features
# features = sparse.load_npz('model_features_genres_only.npz').tocsc() # model features
books = pd.read_csv('books.csv') # book information
w0, wj, vj = parse_output_file('go_model_go.libfm') # model parameters
ratings = pd.read_csv('ratings_us.csv') # ratings
genre_recs = pd.read_csv('genre_diversity_recs.csv') # genre recommendations

@app.callback(
    dash.dependencies.Output('recs-table', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks'),
    dash.dependencies.Input('user-choice', 'value')],
    [dash.dependencies.State('scifi', 'value'),
    dash.dependencies.State('mystery', 'value'),
    dash.dependencies.State('romance', 'value'),
    dash.dependencies.State('historical', 'value'),
    dash.dependencies.State('comics', 'value'),
    dash.dependencies.State('children', 'value'),
    dash.dependencies.State('science', 'value'),
    dash.dependencies.State('business', 'value'),
    dash.dependencies.State('art', 'value'),
    dash.dependencies.State('biography', 'value'),
    dash.dependencies.State('history', 'value'),
    dash.dependencies.State('religion', 'value')])
def update_genres(n_clicks, user_choice, scifi, mystery, romance, historical, comics, 
    children, science, business, art, biography, history, religion):
    
    user_map = {
    'Adam': 53425,
    'Marika': 53426,
    'Mark': 53427,
    'Steph': 53428, 
    'Random': np.random.randint(1, 53429)
    }
    user = user_map[user_choice]
    
    if n_clicks == 0:
        genre_recs_user = genre_recs[genre_recs.user_id == user][['genre_label', 'title', 'authors']]
        genre_recs_user.columns = ['Genre', 'Title', 'Author']
        two_genres = genre_recs_user.Genre.unique()
        return html.Div([
            html.P('Before we get started recommending books based on your rating \
                history, we thought you might like to try something a little different. Based on \
                the genres you like to read, we think you might also like {} and {} books. Not \
                sure where to start? Try these!'.format(two_genres[0].upper(), two_genres[1].upper())),
            generate_table(genre_recs_user)
            ], style={'marginTop': 15})
    else:
        input_vector = np.array([art, biography, business, children, 0, comics, 0, 0, 
            0, 0, historical, history, 0, mystery, 0, religion, romance, science, scifi, 0, 0, 0])
        weight_vector = np.power(10.0, input_vector)
        pred_mat = make_all_predictions(vj, wj, w0, user, ratings, features, weight_vector)
        top_books = get_top_k(pred_mat, books, 5)
        top_books.columns = ['Title', 'Author']
        return html.Div([
            generate_table(top_books)
            ])

if __name__ == '__main__':
    app.run_server(debug=True)


