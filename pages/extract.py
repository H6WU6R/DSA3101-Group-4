from dash import dcc, html, callback, Input, Output, State
from pages.topbar import top_bar
import segmentation  # Import segmentation module functions from src
import plotly.express as px
import pandas as pd

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
        dcc.Store(id='uploaded-data-store'),  # Add this line at the top of children
        top_bar("extract"),
        html.H1("Update Segmentation", style={'marginBottom': '20px'}, className="card-like"),
        # Wrap the upload prompt and dcc.Upload in a card-like container.
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.P(
                            "Upload new user data (CSV) to predict clusters for new customers:",
                            style={'fontSize': '18px'}
                        ),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or Select a File',
                                html.A('', style={'color': PRIMARY_COLOR})
                            ]),
                            style={
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '12px',
                                'margin': '20px auto',
                                'backgroundColor': "#ced897",
                                'borderColor': '#ced897',
                                'color': TEXT_COLOR,
                                'width': '80%',
                                'cursor': 'pointer',
                            },
                            multiple=False
                        ),
                        # Add new div for file information
                        html.Div(
                            id='file-info',
                            style={
                                'marginTop': '10px',
                                'fontSize': '14px',
                                'color': TEXT_COLOR
                            }
                        )
                    ],
                    className="statistics-card",
                    style={'marginBottom': '20px', 'marginTop': '20px', 'borderRadius': '12px'}
                )
            ]
        ),
        html.Br(),
        html.Button("Start Predict", id="update-data-btn", n_clicks=0, style={
            'backgroundColor': "#6c904c",
            'color': 'white',
            'border': 'none',
            'padding': '10px 20px',
            'fontSize': '16px',
            'borderRadius': '5px',
            'cursor': 'pointer',
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
    from datetime import datetime
    import plotly.express as px
    import pandas as pd

    # Add new callback for file info
    @app.callback(
        Output('file-info', 'children'),
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def update_file_info(contents, filename):
        if contents is not None:
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return [
                html.P(f"Uploaded file: {filename}", style={'margin': '5px'}),
                html.P(f"Upload time: {upload_time}", style={'margin': '5px'})
            ]
        return ""

    @app.callback(
        [Output('output-div', 'children'),
         Output('uploaded-data-store', 'data')],
        [Input('update-data-btn', 'n_clicks')],
        [State('upload-data', 'contents'),
         State('upload-data', 'filename')]
    )
    def update_segmentation(n_clicks, contents, filename):
        if contents is None:
            return "No new data uploaded.", None
        
        # Parse and process the data
        new_df = segmentation.parse_contents(contents, filename)
        if new_df is None:
            return "Error reading uploaded file.", None
        
        # Get predictions and add to dataframe
        new_labels = segmentation.predict_clusters(new_df)
        new_df['Cluster_Label'] = new_labels
        
        # Store the processed data
        stored_data = new_df.to_dict('records')
        
        # Create visualizations
        # Get predictions for new data
        new_labels = segmentation.predict_clusters(new_df)
        
        # Add cluster labels to dataframe
        new_df['Cluster_Label'] = new_labels
        
        # Create DataFrame with cluster labels for visualization
        cluster_counts = pd.Series(new_labels).value_counts().sort_index().reset_index()
        cluster_counts.columns = ["cluster", "count"]
        
        # Create bar chart
        fig_bar = px.bar(
            cluster_counts,
            x="cluster",
            y="count",
            title="New Customer Distribution by Cluster",
            labels={"cluster": "Cluster", "count": "Number of Customers"},
            text="count",
            color_discrete_sequence=['#6c904c']  # Match overview.py color
        )
        
        # Update bar chart layout
        fig_bar.update_layout(
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
        fig_bar.update_traces(textposition='outside')
        
        # Create pie chart
        fig_pie = px.pie(
            cluster_counts,
            values="count",
            names="cluster",
            title="Cluster Distribution (%)",
            hole=0.3,
            color_discrete_sequence=['#6c904c', '#acd42c', '#3c6454', '#ced897']  # Match overview.py colors
        )
        
        # Update pie chart layout
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_COLOR),
            title_x=0.5,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        # Generate text summary
        summary_text = f"Total new customers analyzed: {len(new_labels)}\n\n"
        summary_text += "Cluster Distribution:\n"
        for _, row in cluster_counts.iterrows():
            summary_text += f"Cluster {row['cluster']}: {row['count']} customers\n"
        
        # Return layout with visualizations
        return html.Div([
            # Summary card at the top
            html.Div(
                children=[
                    html.P(summary_text, 
                        style={'whiteSpace': 'pre-line', 'textAlign': 'center', 'marginBottom': '10px'}),
                ],
                className="statistics-card",
                style={'marginBottom': '20px', 'padding': '20px', 'width': 'auto'}
            ),
            # Container for charts side by side
            html.Div(
                children=[
                    # Bar chart
                    html.Div(
                        children=[
                            dcc.Graph(figure=fig_bar, config={'displayModeBar': False}),
                        ],
                        className="statistics-card",
                        style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}
                    ),
                    # Pie chart
                    html.Div(
                        children=[
                            dcc.Graph(figure=fig_pie, config={'displayModeBar': False}),
                        ],
                        className="statistics-card",
                        style={'width': '30%', 'display': 'inline-block'}
                    ),
                ],
                style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start'}
            )
        ]), stored_data

if __name__ == '__main__':
    from dash import Dash
    test_app = Dash(__name__, assets_folder='../Resources', suppress_callback_exceptions=True)
    register_callbacks(test_app)
    test_app.layout = extract_layout
    test_app.run_server(debug=True)  # Fixed the incomplete line