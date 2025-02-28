# pages/home.py
from dash import dcc, html, callback, Input, Output, no_update
import segmentation  # Make sure your segmentation module is available

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
        'flexDirection': 'column',
        'position': 'relative'
    },
    children=[
        # Hidden Location component for redirection
        dcc.Location(id='redirect-home', refresh=True),
        html.Div(
            style={
                'flex': '1',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center'
            },
            children=[
                html.H1(
                    "AI-Powered Banking Marketing Solutions",
                    style={
                        'color': TEXT_COLOR,
                        'fontSize': '48px',
                        'textAlign': 'center',
                        'marginBottom': '20px'
                    }
                ),
                html.H2(
                    "Optimize Your Marketing with AI-Driven Insights",
                    style={
                        'color': ACCENT_COLOR,
                        'fontSize': '24px',
                        'textAlign': 'center',
                        'marginBottom': '40px'
                    }
                ),
                # Changed from a dcc.Link to a normal button with an ID to trigger a callback
                html.Button(
                    "Get Started",
                    id="start-btn",
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
                            src="/assets/App Logo.webp",  # Ensure your logo is in the Resources folder
                            style={'width': '100%', 'height': '100%', 'objectFit': 'cover'}
                        )
                    ]
                )
            ]
        )
    ]
)

# Callback: When "Get Started" is clicked, run the segmentation model and redirect to /overview
@callback(
    Output('redirect-home', 'pathname'),
    [Input('start-btn', 'n_clicks')]
)
def start_segmentation(n_clicks):
    if n_clicks and n_clicks > 0:
        # Use the segmentation model to cluster the current dataset.
        # This function reads the CSV from your data folder and trains the model.
        segmentation.initial_model_training()
        # Redirect to the overview page once training is complete.
        return "/overview"
    return no_update
