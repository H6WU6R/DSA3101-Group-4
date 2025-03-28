from dash import dcc, html, callback, Input, Output
from pages.topbar import top_bar
import segmentation  # Ensure segmentation.global_dataset and segmentation.global_model are set
import plotly.express as px
import pandas as pd

# Theme Colors
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

overview_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',  # leave space for fixed top bar
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': "var(--body--background-color)",
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("overview"),
        html.H1("Overview", style={'marginBottom': '20px'}, className="card-like"),
        html.Div(id="overview-summary")
    ]
)

@callback(
    Output("overview-summary", "children"),
    [Input("url", "pathname")]
)
def update_overview(pathname):
    if pathname == "/overview":
        # Total customers from global dataset
        if hasattr(segmentation, "global_dataset") and segmentation.global_dataset is not None:
            total_customers = len(segmentation.global_dataset)
        else:
            total_customers = "N/A"
        # Get number of clusters from global model
        if hasattr(segmentation, "global_model") and segmentation.global_model is not None:
            n_clusters = segmentation.global_model.n_clusters
        else:
            n_clusters = "N/A"
        
        # Build cluster details from the marketing recommendations dictionary
        cluster_details = ""
        if hasattr(segmentation, "marketing_recs"):
            for cl in sorted(segmentation.marketing_recs.keys()):
                rec = segmentation.marketing_recs.get(cl, "No recommendation available.")
                cluster_details += f"Cluster {cl}: {rec}\n"
        else:
            cluster_details = "No cluster details available."
        
        # Create card for total customers
        card_total = html.Div(
            [
                html.P("Total number of customers:", style={'fontSize': '20px', 'margin': '0 0 10px 0'}),
                html.H3(f"{total_customers}", style={'margin': '0'})
            ],
            className="statistics-card",
            style={'width': '30%'}
        )
        
        # Create the pie chart for current clusters
        if hasattr(segmentation, "global_dataset") and segmentation.global_dataset is not None and segmentation.global_model is not None:
            # Scale the global dataset
            scaled_data, _, _ = segmentation.scale_data(segmentation.global_dataset, scaler=segmentation.global_scaler)
            # Predict clusters for all customers
            all_labels = segmentation.global_model.predict(scaled_data)
            df_labels = pd.DataFrame({'cluster': all_labels})
            cluster_counts = df_labels['cluster'].value_counts().sort_index().reset_index()
            cluster_counts.columns = ['cluster', 'count']
            fig = px.pie(cluster_counts, names='cluster', values='count')
        else:
            fig = {}

        # Create card for current clusters with the pie chart
        card_clusters = html.Div(
            [
                html.P("Current clusters:", style={'fontSize': '20px', 'margin': '0 0 10px 0'}),
                html.H3(f"{n_clusters}", style={'margin': '0 0 10px 0'}),
                dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '300px'})
            ],
            className="statistics-card",
            style={'width': '30%'}
        )
        
        # Wrap the two cards in a row
        row_cards = html.Div(
            [card_total, card_clusters],
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'center',
                'gap': '20px'
            }
        )
        
        # Create card for cluster details
        card_details = html.Div(
            [
                html.P("Cluster details:", style={'fontSize': '20px', 'margin': '0 0 10px 0'}),
                html.P(cluster_details, style={'whiteSpace': 'pre-line', 'fontSize': '16px', 'margin': '0'})
            ],
            className="statistics-card",
            style={'margin': '20px auto'}
        )
        
        # Return the overall layout: two cards in a row and one card below
        return html.Div([row_cards, card_details])
    return ""