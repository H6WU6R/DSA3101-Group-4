import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Theme Colors
PRIMARY_COLOR = "#acd42c"
SECONDARY_COLOR = "#ced897"
ACCENT_COLOR = "#93af1c"
BUTTON_COLOR = "#6c904c"
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"

app = dash.Dash(
    __name__,
    assets_folder='Resources',
    suppress_callback_exceptions=True
)
app.title = "AI-Powered Banking Marketing Solutions"

app.layout = html.Div(
    style={
        # Make the background fill the screen
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
        # HERO / LANDING SECTION
        html.Div(
            style={
                'padding': '40px 20px',
                'flex': '1'  # Take up remaining vertical space
            },
            children=[
                # Circle container for the logo
                html.Div(
                    style={
                        'width': '200px',
                        'height': '200px',
                        'borderRadius': '50%',
                        'overflow': 'hidden',
                        'border': f'5px solid {ACCENT_COLOR}',
                        'margin': '0 auto 30px auto',  # Center horizontally & add bottom margin
                        'display': 'block'
                    },
                    children=[
                        html.Img(
                            src=app.get_asset_url("App Logo.webp"),  # Replace with your own image file
                            style={
                                'width': '100%',
                                'height': '100%',
                                'objectFit': 'cover'
                            }
                        )
                    ]
                ),
                
                html.H1(
                    "AI-Powered Banking Marketing Solutions",
                    style={
                        'color': TEXT_COLOR,
                        'fontSize': '48px',
                        'marginTop': '0',
                        'marginBottom': '10px',
                        'textAlign': 'center'
                    }
                ),
                html.H2(
                    "Optimize Your Marketing with AI-Driven Insights",
                    style={
                        'color': ACCENT_COLOR,
                        'fontSize': '24px',
                        'textAlign': 'center',
                        'marginBottom': '30px'
                    }
                ),
                
                # Central CTA Button
                html.Div(
                    style={'textAlign': 'center', 'marginBottom': '40px'},
                    children=[
                        html.Button(
                            "Get Started",
                            style={
                                'backgroundColor': BUTTON_COLOR,
                                'color': 'white',
                                'border': 'none',
                                'padding': '15px 30px',
                                'fontSize': '18px',
                                'borderRadius': '8px',
                                'cursor': 'pointer'
                            }
                        )
                    ]
                ),
                
                # Subtext or bullet sections (mimicking the design's "Welcome" / "AI-Powered" etc.)
                html.Div(
                    style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'flex-start',
                        'flexWrap': 'wrap',
                        'maxWidth': '1200px',
                        'margin': 'auto'
                    },
                    children=[
                        html.Div(
                            style={'flex': '1', 'margin': '20px'},
                            children=[
                                html.H3("Welcome", style={'color': TEXT_COLOR, 'marginBottom': '10px'}),
                                html.P(
                                    "Harness AI-driven strategies to revolutionize your banking campaigns.",
                                    style={'color': TEXT_COLOR, 'fontSize': '16px'}
                                )
                            ]
                        ),
                        html.Div(
                            style={'flex': '1', 'margin': '20px'},
                            children=[
                                html.H3("AI-Marketing", style={'color': TEXT_COLOR, 'marginBottom': '10px'}),
                                html.P(
                                    "Dynamically segment and personalize customer journeys in real-time.",
                                    style={'color': TEXT_COLOR, 'fontSize': '16px'}
                                )
                            ]
                        ),
                        html.Div(
                            style={'flex': '1', 'margin': '20px'},
                            children=[
                                html.H3("UI Design", style={'color': TEXT_COLOR, 'marginBottom': '10px'}),
                                html.P(
                                    "Enhance user engagement with intuitive interfaces tailored to your brand.",
                                    style={'color': TEXT_COLOR, 'fontSize': '16px'}
                                )
                            ]
                        ),
                    ]
                )
            ]
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
