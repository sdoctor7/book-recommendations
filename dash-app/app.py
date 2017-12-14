# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from scipy import sparse
from predict import parse_output_file, output_top_k, get_book_info

# book info
books_genres = sparse.load_npz('books_genres.npz').tocsc()
books = pd.read_csv('../part2/books.csv')

# get model parameters
w0, wj, vj = parse_output_file('model1.libfm')

# make predictions
weight_vector = np.ones(22)
top_ids = output_top_k(vj, wj, w0, 5, 42652, books_genres, weight_vector)
top_books = get_book_info(top_ids, books)

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col, style={'textAlign': 'center'}) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col], style={'textAlign': 'center'}) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
    style={'marginLeft': 'auto', 'marginRight': 'auto'}
    )

app = dash.Dash()
server = app.server

app.layout = html.Div(children=[
    html.H2(children='What Should I Read Next?'),

    html.Div(id='recs-table', 
        style={'width': '60%', 'marginLeft': 'auto', 'marginRight': 'auto'}),

    html.P(children="Want to narrow it down? Try adjusting the weights for some \
        popular genres. Move a genre's slider to the right if you want more of \
        it, and to the left if you want less. When you're ready for new \
        suggestions, hit the Submit button!",
     style={'marginTop': 25}),

    html.Div(children=[
        
        html.Div(children=[

            html.Div(children=[
                dcc.Slider(id='scifi', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),
            
            html.Div(children=[
                dcc.Slider(id='mystery', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='romance', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='historical', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='comics', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='children', min=0, max=1, value=0.5, step=0.1)
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
                dcc.Slider(id='science', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),
            
            html.Div(children=[
                dcc.Slider(id='business', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='art', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='biography', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='history', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 18}),

            html.Div(children=[
                dcc.Slider(id='religion', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 5}),
        
        ], className='three columns', style={'marginLeft': 10, 'marginRight': '9%'}),
    
    ], 
    style={'marginTop': 20, 'width': '100%'},
    className='row'), 

    html.Button(id='submit-button', n_clicks=0, children='Submit')

], style={'textAlign': 'center'})

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

@app.callback(
    dash.dependencies.Output('recs-table', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
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
def update_genres(n_clicks, scifi, mystery, romance, historical, comics, 
    children, science, business, art, biography, history, religion):
    weight_vector = np.ones(22)
    top_ids = output_top_k(vj, wj, w0, 5, 42652, books_genres, weight_vector)
    top_books = get_book_info(top_ids, books)
    return generate_table(top_books)

if __name__ == '__main__':
    app.run_server(debug=True)


