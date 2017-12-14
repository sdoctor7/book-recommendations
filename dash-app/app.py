# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

# df = pd.read_csv()

df = pd.DataFrame([
    {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
    {'title': "Harry Potter and the Sorcerer's Stone", 'author': 'J.K. Rowling'},
    {'title': 'My Brilliant Friend', 'author': 'Elena Ferrante'},
    {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
    {'title': 'Shantaram', 'author': 'Gregory David Roberts'}
    ])

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
    html.H2(children='What Should You Read Next?'),
    html.H6(children="We've compiled some suggestions based on your Goodreads history!"),

    html.Div(children=generate_table(df), 
        style={'width': '60%', 'marginLeft': 'auto', 'marginRight': 'auto'}),

    html.P(children="Not satisfied? Try adjusting the weights for some popular \
     genres. For example, if you want more Romance recommendations, move the \
     Romance slider to the right. If you want fewer Science Fiction \
     recommendations, move the Science Fiction slider to the left.",
     style={'marginTop': 25}),

    html.Div(children=[

        html.Div(children=[
            html.P('Sci-fi'),
            html.P('Horror')
        ], className='two columns', style={'textAlign': 'right'}),
        
        html.Div(children=[

            html.Div(children=[
                dcc.Slider(id='scifi', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 15}),
            
            html.Div(children=[
                dcc.Slider(id='horror', min=0, max=1, value=0.5, step=0.1)
            ], style={'marginTop': 5, 'marginBottom': 5})
        
        ], className='four columns'),

        html.Div(id='genre-values', className='six columns')
            # html.H6(id='genre-values')
            # html.Div(id='genre-values')
        # , className='six columns')
    
    ], 
    style={'marginTop': 20},
    className='row')
], style={'textAlign': 'center'})

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

@app.callback(
    dash.dependencies.Output('genre-values', 'children'),
    [dash.dependencies.Input('scifi', 'value'),
    dash.dependencies.Input('horror', 'value')])
def update_scifi(scifi, horror):
    return html.Table(
        [html.Tr([html.Td('Sci-fi'), html.Td(scifi)]),
        html.Tr([html.Td('Horror'), html.Td(horror)])]
        )

if __name__ == '__main__':
    app.run_server(debug=True)


