# pages/topbar.py
from dash import dcc, html

# Define theme colors (these should match your overall palette)
PRIMARY_COLOR = "#acd42c"
BUTTON_COLOR = "#6c904c"
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"
# Accent color used for borders or highlights
ACCENT_COLOR = "#93af1c"

def top_bar(current_page):
    """
    Returns a top bar with the app logo and name on the left and
    navigation buttons on the right. The active page button is highlighted.
    
    current_page: a string; one of "overview", "extract", "individual"
    """
    # Base style for nav buttons
    base_style = {
        'backgroundColor': BUTTON_COLOR,
        'color': 'white',
        'border': 'none',
        'padding': '10px 15px',
        'fontSize': '14px',
        'borderRadius': '5px',
        'textDecoration': 'none',
        'marginLeft': '10px',
        'cursor': 'pointer'
    }
    # Active style: change background to PRIMARY_COLOR (or any chosen highlight)
    active_style = base_style.copy()
    active_style['backgroundColor'] = PRIMARY_COLOR
    
    def nav_button(label, href, page_key):
        style = active_style if current_page == page_key else base_style
        return dcc.Link(label, href=href, style=style)
    
    return html.Div(
        style={
            'width': '100%',
            'padding': '10px 20px',
            'backgroundColor': BACKGROUND_COLOR,
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'zIndex': '1000',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        },
        children=[
            # Left side: logo and app name
            html.Div(
                style={'display': 'flex', 'alignItems': 'center'},
                children=[
                    html.Img(
                        src="/assets/App Logo.webp",  # Dash serves Resources as /assets
                        style={
                            'width': '50px',
                            'height': '50px',
                            'borderRadius': '50%',
                            'marginRight': '10px'
                        }
                    ),
                    html.H2("Customer Segmentation App", style={'margin': '0', 'color': TEXT_COLOR, 'fontSize': '20px'})
                ]
            ),
            # Right side: navigation buttons
            html.Div(
                style={'display': 'flex', 'alignItems': 'center'},
                children=[
                    nav_button("Overview", "/overview", "overview"),
                    nav_button("Extract", "/extract", "extract"),
                    nav_button("Individual", "/individual", "individual")
                ]
            )
        ]
    )
