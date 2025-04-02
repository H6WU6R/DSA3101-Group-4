# pages/overview.py
from dash import html, dcc, callback, Input, Output
from pages.topbar import top_bar
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import ast
from overview_pages import overview_tabs_layout  # Modified import statement
from src import segmentation  # Add this import

# Theme Colors
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

def load_cluster_profiles():
    """Load cluster profiles from JSON file"""
    with open('./src/A3_Behavioral_Patterns_Analysis/profiles.json', 'r') as f:
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
                style={'width': '200px', 'margin': '20px auto', 'borderRadius': '16px'},
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
    # Convert values and normalize them to 0-1 scale
    raw_values = [float(v.strip('%').strip(' min')) for v in patterns.values()]
    max_val = max(raw_values)
    min_val = min(raw_values)
    normalized_values = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for v in raw_values]

    fig_engagement = go.Figure()
    fig_engagement.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=list(patterns.keys()),
        fill='toself',
        name=f'Cluster {selected_cluster}',
        line=dict(color=PRIMARY_COLOR)
    ))

    fig_engagement.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],  # Set fixed range for normalized values
                tickformat='.0%',  # Format as percentage
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                tickvals=[0, 0.25, 0.5, 0.75, 1]
            )
        ),
        showlegend=True,
        title='Engagement Patterns',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent surrounding
        font=dict(color=TEXT_COLOR),
        title_x=0.5  # Center title
    )
    
    # Channel Distribution Pie Chart
    channels = profile['channel_preferences']['distribution']
    fig_channels = px.pie(
        values=list(channels.values()),
        names=list(channels.keys()),
        title='Channel Distribution',
        color_discrete_sequence=['#6c904c', '#acd42c', '#3c6454', '#ced897', '#d6dcb0']
    )
    fig_channels.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_COLOR),
        title_x=0.5,
        margin=dict(t=40, b=20, l=20, r=20)
    )

    # Campaign Distribution Pie Chart
    campaigns = profile['campaign_preferences']['distribution']
    fig_campaigns = px.pie(
        values=list(campaigns.values()),
        names=list(campaigns.keys()),
        title='Campaign Distribution',
        color_discrete_sequence=['#6c904c', '#acd42c', '#3c6454', '#ced897']
    )
    fig_campaigns.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_COLOR),
        title_x=0.5,
        margin=dict(t=40, b=20, l=20, r=20)
    )

    # Update the cluster summary card
    return html.Div([
        # Cluster Summary Card
        html.Div([
            html.H3(f"Cluster {selected_cluster} Summary", 
                   style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            html.P(f"Size: {profile['size']['count']} customers ({profile['size']['percentage']})"),
            html.P(f"Best Channel: {profile['channel_preferences']['best_channel']} "
                  f"({profile['channel_preferences']['best_channel_conversion']})",),
            html.P(f"Best Campaign: {profile.get('campaign_preferences', {}).get('best_campaign', 'N/A')} "
                  f"({profile.get('campaign_preferences', {}).get('best_campaign_conversion', 'N/A')})",),
            html.P(f"Conversion Rate: {profile['value_metrics']['conversion_rate']}")
        ], className="statistics-card", 
           style={'margin': '20px auto', 'maxWidth': '500px', 'padding': '20px'}),
        
        # Visualizations Grid
        html.Div([
            # First row - Features and Engagement
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_features, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
                html.Div([
                    dcc.Graph(figure=fig_engagement, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'display': 'inline-block', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
            
            # Second row - Channel and Campaign Distribution
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_channels, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
                html.Div([
                    dcc.Graph(figure=fig_campaigns, config={'displayModeBar': False})
                ], className="statistics-card", 
                   style={'width': '45%', 'display': 'inline-block', 'margin': '10px'})
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
        overview_tabs_layout
    ]
)

@callback(
    Output('overview-tabs-content', 'children'),
    [Input('overview-tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-summary':
        return html.Div([
            html.Div(id="overview-summary")
        ])
    elif tab == 'tab-profiles':
        return html.Div([
            create_profile_visualizations()
        ])

# Import the A1 dataset
try:
    df = pd.read_csv('./src/A1_Customer_Segmentation/A1-segmented_df.csv')
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
            
            # Update bar chart layout
            fig_bar.update_layout(
                height=600,  # Increased height
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    showline=True,
                    linecolor='#3c6454'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(60,100,84,0.1)',
                    showline=True,
                    linecolor='#3c6454'
                ),
                title_x=0.5,
                font=dict(color=TEXT_COLOR),
                margin=dict(t=40, b=40, l=40, r=40)
            )
            fig_bar.update_traces(textposition='outside')  # Move text labels outside bars
            
            fig_pie = px.pie(
                cluster_counts,
                names="cluster",
                values="count",
                title="Customer Distribution (%) by Cluster",
                color_discrete_sequence=['#6c904c', '#acd42c', '#3c6454', '#ced897']
            )
            
            # Update pie chart layout
            fig_pie.update_layout(
                height=600,  # Increased height
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_COLOR),
                title_x=0.5,
                margin=dict(t=40, b=40, l=40, r=40)
            )
            
            # Summary section with improved layout
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
                       style={
                           'width': '48%',
                           'margin': '0 10px',
                           'display': 'inline-block',
                           'verticalAlign': 'top'
                       }),
                    
                    html.Div([
                        dcc.Graph(figure=fig_pie, config={'displayModeBar': False})
                    ], className="statistics-card",
                       style={
                           'width': '48%',
                           'margin': '0 10px',
                           'display': 'inline-block',
                           'verticalAlign': 'top'
                       })
                ], style={
                    'width': '100%',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'flex-start',
                    'margin': '20px 0'
                })
            ])
            
            return summary_section
            
        return html.Div([
            html.P("No data available. Please ensure the model is loaded correctly.",
                   style={'fontSize': '18px', 'color': TEXT_COLOR})
        ])
    return ""

from dash import html, dcc

overview_tabs_layout = html.Div([
    dcc.Tabs(
        id='overview-tabs',
        value='tab-summary',
        className='custom-tabs',
        children=[
            dcc.Tab(
                label='Customer Summary',
                value='tab-summary',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Cluster Profiles',
                value='tab-profiles',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
        ]
    ),
    html.Div(id='overview-tabs-content')
])