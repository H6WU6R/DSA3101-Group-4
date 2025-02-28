import sys, os
# Add pages and src folders to system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from home import home_layout
from overview import overview_layout
from extract import extract_layout, register_callbacks as register_extract_callbacks
from individual_dashboard import individual_dashboard_layout

app = dash.Dash(__name__, assets_folder='Resources', suppress_callback_exceptions=True)
server = app.server
app.title = "Customer Segmentation App"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style={'margin': '0', 'padding': '0'})

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/" or pathname is None:
        return home_layout
    elif pathname == "/overview":
        return overview_layout
    elif pathname == "/extract":
        return extract_layout
    elif pathname == "/individual":
        return individual_dashboard_layout
    else:
        return html.Div([html.H1("404: Not Found")])

register_extract_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
