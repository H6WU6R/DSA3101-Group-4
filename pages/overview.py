# pages/overview.py
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
        # Get number of clusters from global model (if available)
        if hasattr(segmentation, "global_dataset") and segmentation.global_dataset is not None:
            n_clusters = segmentation.global_dataset["Cluster_Label"].nunique()
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
        
        # Create the pie chart using the "Cluster_Label" column
        if hasattr(segmentation, "global_dataset") and segmentation.global_dataset is not None:
            df_labels = segmentation.global_dataset.copy()
            if "Cluster_Label" in df_labels.columns:
                cluster_counts = df_labels["Cluster_Label"].value_counts().sort_index().reset_index()
                cluster_counts.columns = ["cluster", "count"]
                fig_bar = px.bar(
                    cluster_counts,
                    x="cluster",
                    y="count",
                    title="Customer Distribution by Cluster",
                    labels={"cluster": "Cluster", "count": "Number of Customers"},
                    text="count"
                )
                fig_bar.update_traces(textposition='outside') 
            else:
                fig_bar = {}
        else:
            fig_bar = {}

        # Card for total customers
        card_total = html.Div(
            [
                dcc.Graph(figure=fig_bar, config={'displayModeBar': False}, style={'height': '300px'})
            ],
            className="statistics-card",
            style={'width': '30%'}
        )
        
        # Create the pie chart using the "Cluster_Label" column
        if hasattr(segmentation, "global_dataset") and segmentation.global_dataset is not None:
            df_labels = segmentation.global_dataset.copy()
            if "Cluster_Label" in df_labels.columns:
                cluster_counts = df_labels["Cluster_Label"].value_counts().sort_index().reset_index()
                cluster_counts.columns = ["cluster", "count"]
                fig = px.pie(cluster_counts, names="cluster", values="count", title="Customer Distribution by Cluster")
            else:
                fig = {}
        else:
            fig = {}

        # Card for current clusters with the pie chart
        card_clusters = html.Div(
            [
                html.P("Current clusters:", style={'fontSize': '20px', 'margin': '0 0 10px 0'}),
                html.H3(f"{n_clusters}", style={'margin': '0 0 10px 0'}),
                dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '300px'})
            ],
            className="statistics-card",
            style={'width': '30%'}
        )
        
        # Wrap the two cards in a row container
        row_cards = html.Div(
            [card_total, card_clusters],
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'gap': '20px',
                'flexWrap': 'wrap',
            }
        )
        
        # Create card for cluster details
        card_details = html.Div(
            [
                html.P(f"Total number of customers: {total_customers}", style={'fontSize': '20px'}),
                html.P(f"Total number of clusters: {n_clusters}", style={'fontSize': '20px'})
            ],
            className="statistics-card",
            style={'margin': '20px auto'}
        )
        
        # Return the overall layout: two cards in a row and one card below
        return html.Div([card_details, row_cards])
    return ""