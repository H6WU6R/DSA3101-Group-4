# pages/overview.py
from dash import dcc, html, callback, Input, Output
from pages.topbar import top_bar
import segmentation  # Ensure segmentation.global_dataset and segmentation.global_model are set

# Theme Colors
BACKGROUND_COLOR = "#d6dcb0"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

overview_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',  # leave space for fixed top bar
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': BACKGROUND_COLOR,
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("overview"),
        html.H1("Overview", style={'marginBottom': '20px'}),
        html.Div(id="overview-summary")
    ]
)

@callback(
    Output("overview-summary", "children"),
    [Input("url", "pathname")]
)
def update_overview(pathname):
    if pathname == "/overview":
        # Ensure segmentation data has been generated via initial_model_training() on home page
        if hasattr(segmentation, "global_dataset") and segmentation.global_dataset is not None:
            total_customers = len(segmentation.global_dataset)
        else:
            total_customers = "N/A"
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

        # Format the text into three paragraphs
        return html.Div([
            html.P(f"Total number of customers: {total_customers}", style={'fontSize': '20px'}),
            html.P(f"Current clusters: {n_clusters}", style={'fontSize': '20px'}),
            html.P("Cluster details:\n" + cluster_details, style={'whiteSpace': 'pre-line', 'fontSize': '16px'})
        ])
    return ""
