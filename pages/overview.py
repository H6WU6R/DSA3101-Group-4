# pages/overview.py
from dash import dcc, html, callback, Input, Output
from pages.topbar import top_bar
import segmentation  # Ensure segmentation.global_dataset and segmentation.global_model are set
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json
import ast

# Theme Colors
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

def load_cluster_profiles():
    """Load cluster profiles from JSON file"""
    with open('data/profiles.json', 'r') as f:
        return json.load(f)

def create_profile_visualizations():
    """Create visualizations for cluster profiles"""
    profiles = load_cluster_profiles()
    
    return html.Div([
        # Dropdown for cluster selection
        html.Div([
            dcc.Dropdown(
                id='cluster-profile-dropdown',
                options=[
                    {'label': f'Cluster {k}', 'value': k}
                    for k in profiles.keys()
                ],
                value='0',  # Default to first cluster
                style={'width': '200px', 'margin': '20px auto'}
            )
        ]),
        
        # Container for profile visualizations
        html.Div(id='cluster-profile-content')
    ])

@callback(
    Output('cluster-profile-content', 'children'),
    [Input('cluster-profile-dropdown', 'value')]
)
def update_cluster_profile(selected_cluster):
    if not selected_cluster:
        return html.P("Please select a cluster to view its profile.")
    
    profiles = load_cluster_profiles()
    profile = profiles[selected_cluster]
    
    # Create visualizations for selected cluster
    # Top Features Bar Chart
    top_features = ast.literal_eval(profile['top_features'])
    features_df = pd.DataFrame(top_features)
    
    fig_features = px.bar(
        features_df,
        x='feature',
        y='deviation',
        color='direction',
        title=f'Top 5 Features for Cluster {selected_cluster}',
        labels={'deviation': 'Importance Score', 'feature': 'Feature'},
        color_discrete_map={'higher': '#6c904c', 'lower': '#d64545'}
    )
    
    # Update layout to remove background and customize appearance
    fig_features.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent surrounding
        xaxis=dict(
            showgrid=False,  # Remove x-axis gridlines
            showline=True,   # Show x-axis line
            linecolor='#3c6454'  # Match text color
        ),
        yaxis=dict(
            showgrid=True,   # Keep y-axis gridlines
            gridcolor='rgba(60,100,84,0.1)',  # Light gridlines
            showline=True,   # Show y-axis line
            linecolor='#3c6454'  # Match text color
        ),
        title_x=0.5,  # Center the title
        font=dict(color=TEXT_COLOR)  # Match text color
    )
    
    # Engagement Spider Chart
    patterns = profile['engagement_patterns']
    values = [float(v.strip('%').strip(' min')) for v in patterns.values()]
    
    fig_engagement = go.Figure()
    fig_engagement.add_trace(go.Scatterpolar(
        r=values,
        theta=list(patterns.keys()),
        fill='toself',
        name=f'Cluster {selected_cluster}'
    ))
    
    fig_engagement.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values)])),
        title='Engagement Patterns',
        showlegend=True
    )
    
    # Channel Distribution Pie Chart
    channels = profile['channel_preferences']['distribution']
    fig_channels = px.pie(
        values=list(channels.values()),
        names=list(channels.keys()),
        title='Channel Distribution',
        color_discrete_sequence=['#6c904c', '#acd42c', '#3c6454', '#ced897', '#d6dcb0']
    )
    
    return html.Div([
        # Cluster Summary Card
        html.Div([
            html.H3(f"Cluster {selected_cluster} Summary", 
                   style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            html.P(f"Size: {profile['size']['count']} customers ({profile['size']['percentage']})"),
            html.P(f"Best Channel: {profile['channel_preferences']['best_channel']}"),
            html.P(f"Conversion Rate: {profile['value_metrics']['conversion_rate']}")
        ], className="statistics-card", 
           style={'margin': '20px auto', 'maxWidth': '500px', 'padding': '20px'}),
        
        # Visualizations Grid
        html.Div([
            # First row
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_features, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
                html.Div([
                    dcc.Graph(figure=fig_engagement, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'display': 'inline-block', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            
            # Second row
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_channels, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'margin': '20px auto'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
    ])

overview_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': "var(--body--background-color)",
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("overview"),
        html.H1("Cluster Overview", style={'marginBottom': '20px'}, className="card-like"),
        html.Div(id="overview-summary"),  # Add this back to show the original charts
        html.H2("Cluster Profile Analysis", style={'marginTop': '40px', 'marginBottom': '20px'}),
        create_profile_visualizations()
    ]
)

# Import the A1 dataset
try:
    df = pd.read_csv('data/A1-segmented_df.csv')
    segmentation.global_dataset = df
except Exception as e:
    print(f"Error loading global dataset: {e}")
    segmentation.global_dataset = None

@callback(
    Output("overview-summary", "children"),
    [Input("url", "pathname")]
)
def update_overview(pathname):
    if pathname == "/overview":
        # Load data for visualizations
        if segmentation.global_dataset is not None:
            df = segmentation.global_dataset
            total_customers = len(df)
            n_clusters = len(df["Cluster_Label"].unique())
            
            # Create cluster distribution bar chart
            cluster_counts = df["Cluster_Label"].value_counts().sort_index().reset_index()
            cluster_counts.columns = ["cluster", "count"]
            
            fig_bar = px.bar(
                cluster_counts,
                x="cluster",
                y="count",
                title="Customer Distribution by Cluster",
                labels={"cluster": "Cluster", "count": "Number of Customers"},
                text="count",
                color_discrete_sequence=['#6c904c']
            )
            
            # Update layout to remove background and customize appearance
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent surrounding
                xaxis=dict(
                    showgrid=False,  # Remove x-axis gridlines
                    showline=True,   # Show x-axis line
                    linecolor='#3c6454'  # Match text color
                ),
                yaxis=dict(
                    showgrid=True,   # Keep y-axis gridlines
                    gridcolor='rgba(60,100,84,0.1)',  # Light gridlines
                    showline=True,   # Show y-axis line
                    linecolor='#3c6454'  # Match text color
                ),
                title_x=0.5,  # Center the title
                font=dict(color=TEXT_COLOR)  # Match text color
            )
            fig_bar.update_traces(textposition='outside')  # Move text labels outside bars
            
            fig_pie = px.pie(
                cluster_counts,
                names="cluster",
                values="count",
                title="Customer Distribution (%) by Cluster",
                color_discrete_sequence=['#6c904c', '#acd42c', '#3c6454', '#ced897']
            )
            
            # Summary cards
            summary_section = html.Div([
                # Summary card
                html.Div(
                    [
                        html.P(f"Total number of customers: {total_customers}", 
                               style={'fontSize': '20px', 'fontWeight': 'bold'}),
                        html.P(f"Number of clusters: {n_clusters}", 
                               style={'fontSize': '20px', 'fontWeight': 'bold'})
                    ],
                    className="statistics-card",
                    style={'margin': '20px auto', 'padding': '20px', 'maxWidth': '500px'}
                ),
                
                # Charts container
                html.Div([
                    html.Div([
                        dcc.Graph(figure=fig_bar, config={'displayModeBar': False})
                    ], className="statistics-card", 
                       style={'width': '40%', 'display': 'inline-block', 'margin': '10px'}),
                    
                    html.Div([
                        dcc.Graph(figure=fig_pie, config={'displayModeBar': False})
                    ], className="statistics-card",
                       style={'width': '40%', 'display': 'inline-block', 'margin': '10px'})
                ], style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'flexWrap': 'wrap',
                    'gap': '20px',
                    'marginTop': '20px'
                })
            ])
            
            return summary_section
            
        return html.Div([
            html.P("No data available. Please ensure the model is loaded correctly.",
                   style={'fontSize': '18px', 'color': TEXT_COLOR})
        ])
    return ""