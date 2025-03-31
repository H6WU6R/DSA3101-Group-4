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
    
    # Segment Size Chart
    sizes_df = pd.DataFrame([
        {
            'Cluster': f"Cluster {k}",
            'Count': int(v['size']['count']),
            'Percentage': float(v['size']['percentage'].strip('%'))
        }
        for k, v in profiles.items()
    ])
    
    fig_size = px.bar(
        sizes_df,
        x='Cluster',
        y='Count',
        text='Percentage',
        labels={'Count': 'Number of Customers'},
        title='Cluster Sizes',
        color_discrete_sequence=['#6c904c']
    )
    fig_size.update_traces(texttemplate='%{text}%')
    
    # Top Features Heatmap
    features_data = []
    for cluster, profile in profiles.items():
        top_features = ast.literal_eval(profile['top_features'])
        for feature in top_features:
            features_data.append({
                'Cluster': f"Cluster {cluster}",
                'Feature': feature['feature'],
                'Deviation': feature['deviation'],
                'Direction': feature['direction']
            })
    
    features_df = pd.DataFrame(features_data)
    fig_features = px.imshow(
        features_df.pivot(index='Feature', columns='Cluster', values='Deviation'),
        color_continuous_scale=['red', 'white', 'green'],
        title='Feature Importance by Cluster',
        aspect='auto'
    )
    
    # Engagement Patterns Spider Chart
    fig_engagement = go.Figure()
    for cluster, profile in profiles.items():
        patterns = profile['engagement_patterns']
        values = [float(v.strip('%').strip(' min')) for v in patterns.values()]
        fig_engagement.add_trace(go.Scatterpolar(
            r=values,
            theta=list(patterns.keys()),
            name=f'Cluster {cluster}'
        ))
    
    fig_engagement.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values)])),
        title='Engagement Patterns by Cluster',
        showlegend=True
    )
    
    return html.Div([
        # First row - Size and Features
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_size)
            ], className="statistics-card", style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=fig_features)
            ], className="statistics-card", style={'width': '45%', 'display': 'inline-block', 'marginLeft': '2%'})
        ]),
        # Second row - Engagement Patterns
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_engagement)
            ], className="statistics-card", style={'width': '92%', 'margin': '20px auto'})
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

# Import the global dataset
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
            fig_bar.update_traces(textposition='outside')
            
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
                       style={'width': '38%', 'display': 'inline-block', 'margin': '10px'}),
                    
                    html.Div([
                        dcc.Graph(figure=fig_pie, config={'displayModeBar': False})
                    ], className="statistics-card",
                       style={'width': '38%', 'display': 'inline-block', 'margin': '10px'})
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