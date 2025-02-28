from dash import dcc, html

nav_button_style = {
    'backgroundColor': "#6c904c",
    'color': 'white',
    'border': 'none',
    'padding': '10px 15px',
    'fontSize': '14px',
    'borderRadius': '5px',
    'textDecoration': 'none',
    'marginLeft': '10px',
    'cursor': 'pointer'
}

nav_bar = html.Div(
    children=[
        dcc.Link("Overview", href="/overview", style=nav_button_style),
        dcc.Link("Extract", href="/extract", style=nav_button_style),
        dcc.Link("Individual", href="/individual", style=nav_button_style),
    ],
    style={
        'position': 'absolute',
        'top': '10px',
        'right': '10px',
        'display': 'flex',
        'justifyContent': 'flex-end'
    }
)
