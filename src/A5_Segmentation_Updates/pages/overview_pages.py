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

# Add this line to explicitly define what gets exported
__all__ = ['overview_tabs_layout']