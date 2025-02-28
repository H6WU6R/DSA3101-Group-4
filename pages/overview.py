from dash import dcc, html
from pages.topbar import top_bar

# Theme Colors
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

overview_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',  # leave space for fixed top bar
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': BACKGROUND_COLOR,
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("overview"),
        html.H1("Overview", style={'marginBottom': '20px'}),
        html.P("Total number of customers: 1000", style={'fontSize': '20px'}),
        html.P("Current clusters: 5", style={'fontSize': '20px'}),
        html.P("Cluster details:\nCluster 0: High spenders\nCluster 1: Moderate spenders\nCluster 2: Low spenders\n...", 
               style={'whiteSpace': 'pre-line', 'fontSize': '16px'})
    ]
)
