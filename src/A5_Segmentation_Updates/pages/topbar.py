from dash import dcc, html

# Define theme colors
PRIMARY_COLOR = "#acd42c"
BUTTON_COLOR = "#6c904c"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#3c6454"

def top_bar(current_page):
    """
    Returns a top bar with the app logo and name on the left and
    navigation buttons on the right. Highlights the current page.
    current_page should be one of: "overview", "extract", "individual"
    """
    base_style = {
        'backgroundColor': BUTTON_COLOR,
        'color': 'white',
        'border': 'none',
        'padding': '10px 10px',
        'fontSize': '14px',
        'borderRadius': '5px',
        'textDecoration': 'none',
        'cursor': 'pointer'
    }
    active_style = base_style.copy()
    active_style['backgroundColor'] = PRIMARY_COLOR
    
    def nav_button(label, href, page_key):
        style = active_style if current_page == page_key else base_style
        return dcc.Link(label, href=href, style=style)
    
    return html.Div(
        style={
            'width': '100%',
            'padding': '0 10px',
            'backgroundColor': BACKGROUND_COLOR,
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'height': '70px',
            'zIndex': '1000',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        },
        children=[
            html.Div(
                style={'display': 'flex', 'alignItems': 'center', 'width': '250px'},
                children=[
                    html.Img(
                        src="/assets/App Logo.webp",
                        style={'width': '50px', 'height': '50px', 'borderRadius': '50%', 'marginRight': '10px'}
                    ),
                    html.H1("Customer Segmentation", style={'margin': '0', 'color': TEXT_COLOR, 'fontSize': '16px'})
                ]
            ),
            html.Div(
                style={'flex': '1', 'textAlign': 'center', 'color': TEXT_COLOR, 'fontSize': '16px', 'marginLeft': '25px'},
                children=[
                    html.H3("DSA3101 Group 4", style={'margin': '0'})
                ]
            ),
            html.Div(
                style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginRight': '15px'},
                children=[
                    nav_button("Overview", "/overview", "overview"),
                    nav_button("Update", "/extract", "extract"),
                    nav_button("Individual", "/individual", "individual")
                ]
            )
        ]
    )
