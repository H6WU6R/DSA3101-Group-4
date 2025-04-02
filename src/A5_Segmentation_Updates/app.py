import sys
import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from A5_Segmentation_Updates.pages.home import home_layout
from A5_Segmentation_Updates.pages.overview import overview_layout
from A5_Segmentation_Updates.pages.extract import extract_layout, register_callbacks as register_extract_callbacks
from A5_Segmentation_Updates.pages.individual_dashboard import individual_dashboard_layout

# Update width constant to be full screen
CONTENT_WIDTH = '100%'

app = dash.Dash(__name__, assets_folder='Resources', suppress_callback_exceptions=True)
server = app.server
app.title = "Customer Segmentation App"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        style={
            'width': '100%',
            'minHeight': '100vh',
            'margin': '0',
            'padding': '0',
            'backgroundColor': 'var(--body--background-color)',  # Match background color
            'display': 'flex',
            'justifyContent': 'center'  # Center the content horizontally
        },
        children=html.Div(
            id='page-content',
            style={
                'width': CONTENT_WIDTH,
                'padding': '0 40px',
                'boxSizing': 'border-box',
                'backgroundColor': 'var(--body--background-color)'  # Match background color
            }
        )
    )
])

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

def main(debug=True, **kwargs):
    if __name__ == '__main__':
        app.run_server(debug=True)

