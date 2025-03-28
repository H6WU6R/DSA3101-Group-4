from dash import dcc, html
from pages.topbar import top_bar

# Theme Colors
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#3c6454"
PRIMARY_COLOR = "#acd42c"

extract_layout = html.Div(
    style={
        'minHeight': '100vh',
        'width': '100%',
        'marginTop': '70px',  # space for fixed top bar
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': "var(--body--background-color)",
        'color': TEXT_COLOR,
        'position': 'relative',
        'textAlign': 'center'
    },
    children=[
        top_bar("extract"),
        html.H1("Data Extraction", style={'marginBottom': '20px'}),
        # Wrap the upload prompt and dcc.Upload in a card-like container.
        html.Div(
            children=[
                html.P(
                    "Upload new user data (CSV) to predict clusters for new customers:",
                    style={'fontSize': '18px'}
                ),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a File', style={'color': PRIMARY_COLOR})
                    ]),
                    style={
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '12px',
                        'margin': '20px auto',
                        'backgroundColor': "#ced897",
                        'color': TEXT_COLOR
                    },
                    multiple=False
                )
            ],
            className="card-like"  # This class should be defined in your CSS file.
        ),
        html.Br(),
        html.Button("Start Predict", id="update-data-btn", n_clicks=0, style={
            'backgroundColor': "#6c904c",
            'color': 'white',
            'border': 'none',
            'padding': '10px 20px',
            'fontSize': '16px',
            'borderRadius': '5px',
            'marginTop': '20px'
        }),
        html.Br(),
        html.Div(id="output-div", style={
            'whiteSpace': 'pre-line',
            'fontSize': '16px',
            'marginTop': '40px',
            'color': TEXT_COLOR
        })
    ]
)

def register_callbacks(app):
    from dash.dependencies import Input, Output, State
    import segmentation  # Import segmentation logic from src
    @app.callback(
        Output('output-div', 'children'),
        [Input('update-data-btn', 'n_clicks')],
        [State('upload-data', 'contents'),
         State('upload-data', 'filename')]
    )
    def update_segmentation(n_clicks, contents, filename):
        if n_clicks == 0:
            return "Please upload new user data (CSV) and click 'Update Data' to see its segmentation and marketing recommendation."
        if contents is not None:
            new_df = segmentation.parse_contents(contents, filename)
            if new_df is None:
                return "Error reading uploaded file."
        else:
            new_df = segmentation.initial_model_training()[0].copy()
        new_df = segmentation.preprocess_data(new_df)
        new_scaled, segmentation.global_scaler, _ = segmentation.scale_data(new_df, scaler=segmentation.global_scaler)
        segmentation.global_model.partial_fit(new_scaled)
        new_labels = segmentation.global_model.predict(new_scaled)
        output_text = ""
        for i, label in enumerate(new_labels):
            rec = segmentation.marketing_recs.get(label, "No recommendation available.")
            output_text += f"Data row {i+1} belongs to Cluster {label}.\nRecommendation: {rec}\n\n"
        return output_text

if __name__ == '__main__':
    from dash import Dash
    test_app = Dash(__name__, assets_folder='../Resources', suppress_callback_exceptions=True)
    register_callbacks(test_app)