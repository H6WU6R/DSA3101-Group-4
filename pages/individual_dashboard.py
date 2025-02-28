from dash import dcc, html
from pages.topbar import top_bar

# Theme Colors
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

individual_dashboard_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',  # space for fixed top bar
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': BACKGROUND_COLOR,
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("individual"),
        html.H1("Individual Dashboard", style={'marginBottom': '20px'}),
        html.P("Select a customer to view detailed segmentation information:", style={'fontSize': '18px'}),
        dcc.Dropdown(
            id='customer-dropdown',
            options=[{'label': f'Customer {i}', 'value': i} for i in range(1, 11)],
            placeholder="Select a customer",
            style={'width': '40%', 'margin': 'auto'}
        ),
        html.Br(),
        html.Div(id='customer-output', style={'whiteSpace': 'pre-line', 'fontSize': '16px'})
    ]
)
