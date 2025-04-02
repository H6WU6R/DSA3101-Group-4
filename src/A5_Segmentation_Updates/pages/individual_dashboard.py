from dash import dcc, html, callback, Input, Output, State, ctx
import pandas as pd
import dash
from pages.topbar import top_bar
from src.recommendation import query_llm
from dash import callback_context
from src.prompts import USER_PROMPT  # Add this import at the top with other imports
import json

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
        dcc.Store(id='uploaded-data-store'),  # Store for uploaded data
        top_bar("individual"),
        html.H1("Individual Dashboard", style={'marginBottom': '20px'}, className="card-like"),
        html.P(
            "Select a customer from uploaded data to view segmentation information and receive recommendations:",
            style={'fontSize': '18px'}
        ),
        dcc.Dropdown(
            id='customer-dropdown',
            options=[],  # Will be populated from uploaded data
            placeholder="Select a customer",
            style={
                'width': '40%', 
                'margin': 'auto', 
                'borderRadius': '16px',
                'transition': 'all 0.3s ease'
            },
            className='custom-dropdown',
            searchable=False,  # Disable typing/searching
            clearable=False   # Prevent clearing selection
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
    [Output('customer-dropdown', 'options'),
     Output('customer-dropdown', 'value')],
    [Input('uploaded-data-store', 'data')]
)
def update_dropdown_options(stored_data):
    if not stored_data:
        return [], None
    
    try:
        # Convert stored data back to DataFrame
        df = pd.DataFrame(stored_data)
        
        if 'CustomerID' not in df.columns:
            print("CustomerID column not found in data")
            return [], None
        
        # Create dropdown options
        options = [
            {
                'label': f"Customer {cid} (Cluster {cluster})", 
                'value': cid
            }
            for cid, cluster in zip(df['CustomerID'], df['Cluster_Label'])
        ]
        
        # Sort by CustomerID
        options.sort(key=lambda x: x['value'])
        
        print(f"Generated {len(options)} dropdown options")  # Debug print
        return options, None
        
    except Exception as e:
        print(f"Error in update_dropdown_options: {str(e)}")
        return [], None

@callback(
    Output('predicted-cluster-card', 'children'),
    [Input('customer-dropdown', 'value')],
    [State('uploaded-data-store', 'data')]
)
def update_predicted_cluster(selected_customer, data):
    if selected_customer is None:
        return "Please select a customer to see the predicted cluster."
    
    if not data:
        return "No data available. Please upload data in the Extract page first."
    
    try:
        df = pd.DataFrame(data)
        customer_data = df[df['CustomerID'] == selected_customer].iloc[0]
        predicted_cluster = customer_data.get('Cluster_Label', 'N/A')
        
        return html.Div([
            html.H3("Predicted Cluster", style={'marginBottom': '10px', 'color': PRIMARY_COLOR}),
            html.P(f"Customer {selected_customer} is in Cluster {predicted_cluster}."),
            html.Div([
                html.P(f"Gender: {customer_data.get('Gender', 'N/A')}"),
                html.P(f"Campaign Channel: {customer_data.get('CampaignChannel', 'N/A')}"),
                html.P(f"Campaign Type: {customer_data.get('CampaignType', 'N/A')}")
            ], style={'marginTop': '10px', 'textAlign': 'center'})
        ])
    except Exception as e:
        return f"Error retrieving customer data: {str(e)}"

@callback(
    Output('recommendation-output', 'children'),
    [Input('customer-dropdown', 'value')],
    [State('uploaded-data-store', 'data')]
)
def update_recommendation(selected_customer, data):
    if not data:
        return "No data available. Please upload data in the Extract page first."
    
    try:
        df = pd.DataFrame(data)
        customer_data = df[df['CustomerID'] == selected_customer].iloc[0]
        
        customer_profile = {
            "cluster": int(customer_data.get('Cluster_Label', -1)),
            "gender": str(customer_data.get('Gender', 'N/A')),
            "campaign_channel": str(customer_data.get('CampaignChannel', 'N/A')),
            "campaign_type": str(customer_data.get('CampaignType', 'N/A'))
        }
        
        recommendation = query_llm(f"{USER_PROMPT}\n\nCustomer Profile:\n{json.dumps(customer_profile, indent=2)}")
        
        return html.Div([
            html.H3("Marketing Recommendation", 
                    style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
            dcc.Markdown(
                recommendation,
                dangerously_allow_html=True,
                style={
                    'whiteSpace': 'pre-wrap',
                    'fontSize': '16px',
                    'lineHeight': '1.5',
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'textAlign': 'left'  # Added for better markdown readability
                }
            )
        ])
    except Exception as e:
        print(f"Recommendation error details: {str(e)}")
        return f"Error generating recommendation: {str(e)}"
