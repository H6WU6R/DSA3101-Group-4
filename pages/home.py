# pages/home.py
from dash import dcc, html

# Theme Colors
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"
BUTTON_COLOR = "#6c904c"
ACCENT_COLOR = "#93af1c"

home_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'margin': '0',
        'padding': '0',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': BACKGROUND_COLOR,
        'display': 'flex',
        'flexDirection': 'column'
    },
    children=[
        html.Div(
            style={
                'flex': '1',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center'
            },
            children=[
                html.H1("AI-Powered Banking Marketing Solutions",
                        style={'color': TEXT_COLOR, 'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px'}),
                html.H2("Optimize Your Marketing with AI-Driven Insights",
                        style={'color': ACCENT_COLOR, 'fontSize': '24px', 'textAlign': 'center', 'marginBottom': '40px'}),
                dcc.Link(
                    html.Button("Get Started",
                                style={
                                    'backgroundColor': BUTTON_COLOR,
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '15px 30px',
                                    'fontSize': '18px',
                                    'borderRadius': '8px',
                                    'cursor': 'pointer'
                                }),
                    href="/overview"
                )
            ]
        ),
        # Footer with round logo at bottom center
        html.Footer(
            style={'textAlign': 'center', 'padding': '20px'},
            children=[
                html.Div(
                    style={
                        'width': '150px',
                        'height': '150px',
                        'borderRadius': '50%',
                        'overflow': 'hidden',
                        'border': f'5px solid {ACCENT_COLOR}',
                        'margin': '0 auto'
                    },
                    children=[
                        html.Img(
                            src="/assets/App Logo.webp",
                            style={'width': '100%', 'height': '100%', 'objectFit': 'cover'}
                        )
                    ]
                )
            ]
        )
    ]
)
