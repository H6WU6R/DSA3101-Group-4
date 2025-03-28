from dash import dcc, html, callback, Input, Output
from pages.topbar import top_bar
from src.recommendation import query_llm  # Import the LLM query function
from src.prompts import get_prompt_for_cluster  # Optional: if you use pre-defined prompts

# Theme Colors
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"
BUTTON_COLOR = "#6c904c"

individual_dashboard_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',  # leave space for the fixed top bar
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': "var(--body--background-color)",
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("individual"),
        html.H1("Individual Dashboard", style={'marginBottom': '20px'}),
        html.P(
            "Select a customer to view detailed segmentation information and receive a personalized marketing recommendation:",
            style={'fontSize': '18px'}
        ),
        dcc.Dropdown(
            id='customer-dropdown',
            options=[{'label': f'Customer {i}', 'value': i} for i in range(1, 11)],
            placeholder="Select a customer",
            style={'width': '40%', 'margin': 'auto'}
        ),
        html.Br(),
        # Card to display predicted cluster immediately upon selection
        html.Div(id='predicted-cluster-card', style={
            'border': f'2px solid {PRIMARY_COLOR}',
            'borderRadius': '8px',
            'padding': '15px',
            'width': '40%',
            'margin': '20px auto',
            'backgroundColor': "#ffffff",
            'fontSize': '16px'
        }),
        html.Br(),
        # Loading component for recommendation output
        dcc.Loading(
            id='loading-recommendation',
            type='default',
            children=html.Div(id='recommendation-output', style={
                'whiteSpace': 'pre-line',
                'fontSize': '16px',
                'marginTop': '20px'
            })
        )
    ]
)

@callback(
    Output('predicted-cluster-card', 'children'),
    [Input('customer-dropdown', 'value')]
)
def update_predicted_cluster(selected_customer):
    if selected_customer is None:
        return "Please select a customer to see the predicted cluster."
    
    # Replace this dummy logic with your actual segmentation logic.
    try:
        # Example: assume our segmentation model has 5 clusters.
        n_clusters = 5  
        predicted_cluster = selected_customer % n_clusters
    except Exception:
        predicted_cluster = "N/A"
    
    card = html.Div([
        html.H3("Predicted Cluster", style={'marginBottom': '10px', 'color': PRIMARY_COLOR}),
        html.P(f"Customer {selected_customer} is in Cluster {predicted_cluster}.")
    ])
    return card

@callback(
    Output('recommendation-output', 'children'),
    [Input('customer-dropdown', 'value')]
)
def update_recommendation(selected_customer):
    if selected_customer is None:
        return ""
    
    # Determine the predicted cluster (this should match the logic in update_predicted_cluster)
    try:
        n_clusters = 5
        predicted_cluster = selected_customer % n_clusters
    except Exception:
        predicted_cluster = "N/A"
    
    # Build a prompt. Option 1: using our prompts module
    # prompt_text = get_prompt_for_cluster(predicted_cluster, additional_context=f"Customer ID: {selected_customer}")
    # Option 2: build the prompt directly:
    prompt_text = (
        f"Customer {selected_customer} is predicted to be in Cluster {predicted_cluster}. "
        "Based on our marketing documentation for this cluster, provide a detailed, personalized marketing recommendation. "
        "Include quantitative methods for tracking engagement and optimizing campaign performance."
    )
    
    # Query the LLM via our recommendation function
    recommendation = query_llm(prompt_text)
    return f"Recommendation:\n{recommendation}"
