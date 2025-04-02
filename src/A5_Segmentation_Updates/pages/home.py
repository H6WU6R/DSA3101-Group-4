from dash import dcc, html, callback, Input, Output, no_update
import pandas as pd
import segmentation  # Make sure your segmentation module is available

# Theme Colors
PRIMARY_COLOR = "#acd42c"
SECONDARY_COLOR = "#ced897"
ACCENT_COLOR = "#93af1c"
BUTTON_COLOR = "#6c904c"
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"

home_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'margin': '0',
        'padding': '0',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': 'var(--body--background-color)',
        'display': 'flex',
        'flexDirection': 'column',
        'position': 'relative'
    },
    children=[
        dcc.Location(id='redirect-home', refresh=True),
        # Logo at the top
        html.Div(
            style={
                'width': '300px',
                'height': '300px',
                'overflow': 'hidden',
                'margin': '0px auto 0px auto',  # Reduced bottom margin to 10px
            },
            children=[
                html.Img(
                    src="/assets/App Logo.png",  # Replace with your own image file
                    style={
                        'width': '100%',
                        'height': '100%',
                        'objectFit': 'contain'
                    }
                )
            ]
        ),
        # Main content
        html.Div(
            style={
                'flex': '0.8',  # Reduce flex value to move content up
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'flex-start',  # Change from 'center' to 'flex-start'
                'alignItems': 'center',
                'marginTop': '-20px'  # Add negative margin to move content up
            },
            children=[
                html.H1(
                    "AI-Powered Banking Marketing Solutions",
                    className='shine-text',  # Apply the animation class
                    style={
                        'color': TEXT_COLOR,
                        'fontSize': '60px',  # Adjusted font size
                        'textAlign': 'center',
                        'marginBottom': '20px'
                    }
                ),
                html.H2(
                    "Optimize Your Marketing with AI-Driven Insights",
                    className='shine-text',  # Apply the animation class
                    style={
                        'color': ACCENT_COLOR,
                        'fontSize': '24px',
                        'textAlign': 'center',
                        'marginBottom': '40px'
                    }
                ),
                html.Button(
                    "Get Started",
                    id="start-btn",
                    className="shine-button",
                    style={
                        'marginTop': '20px',  # Add some spacing from the text above
                        'fontWeight': 'bold'  # Make text bolder for better visibility
                    }
                )
            ]
        ),
        # Subtext or bullet sections at the bottom
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'space-between',  # Changed from 'center' to 'space-between'
                'alignItems': 'center',  # Changed from 'flex-start' to 'center'
                'flexWrap': 'wrap',
                'maxWidth': '1200px',
                'margin': '40px auto',
                'padding': '0 40px',  # Added padding for better spacing
                'width': '100%',  # Added to ensure full width usage
            },
            children=[
                html.Div(
                    style={
                        'flex': '1',
                        'margin': '20px',
                        'minWidth': '300px',  # Added minimum width
                        'textAlign': 'center',  # Center the content within each box
                        'maxWidth': '350px'  # Added maximum width for consistency
                    },
                    children=[
                        html.H3(
                            "AI-Marketing", 
                            className='shine-subtext',  # Added shine effect
                            style={
                                'marginBottom': '10px',
                                'fontSize': '24px'  # Added consistent font size
                            }
                        ),
                        html.P(
                            "Dynamically segment and personalize customer journeys in real-time.",
                            className='shine-subtext',  # Added shine effect
                            style={
                                'fontSize': '16px',
                                'lineHeight': '1.5'  # Added line height for better readability
                            }
                        )
                    ]
                ),
                html.Div(
                    style={
                        'flex': '1',
                        'margin': '20px',
                        'minWidth': '300px',
                        'textAlign': 'center',
                        'maxWidth': '350px'
                    },
                    children=[
                        html.H3(
                            "Welcome", 
                            className='shine-subtext',  # Added shine effect
                            style={
                                'marginBottom': '10px',
                                'fontSize': '24px'
                            }
                        ),
                        html.P(
                            "Harness AI-driven strategies to revolutionize your banking campaigns.",
                            className='shine-subtext',  # Added shine effect
                            style={
                                'fontSize': '16px',
                                'lineHeight': '1.5'
                            }
                        )
                    ]
                ),
                html.Div(
                    style={
                        'flex': '1',
                        'margin': '20px',
                        'minWidth': '300px',
                        'textAlign': 'center',
                        'maxWidth': '350px'
                    },
                    children=[
                        html.H3(
                            "UI Design", 
                            className='shine-subtext',  # Added shine effect
                            style={
                                'marginBottom': '10px',
                                'fontSize': '24px'
                            }
                        ),
                        html.P(
                            "Enhance user engagement with intuitive interfaces tailored to your brand.",
                            className='shine-subtext',  # Added shine effect
                            style={
                                'fontSize': '16px',
                                'lineHeight': '1.5'
                            }
                        )
                    ]
                )
            ]
        ),
    ]
)

@callback(
    Output('redirect-home', 'pathname'),
    [Input('start-btn', 'n_clicks')]
)
def start_segmentation(n_clicks):
    if n_clicks and n_clicks > 0:
        try:
            # Load pre-generated segmentation results
            df_segmented = pd.read_csv("./src/A1_Customer_Segmentation/A1-segmented_df.csv")
            segmentation.global_dataset = df_segmented
        except Exception as e:
            print("Error reading segmented data:", e)
        return "/overview"
    return no_update