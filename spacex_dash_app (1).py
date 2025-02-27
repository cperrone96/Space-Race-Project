# Import required libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Get unique launch sites from the dataframe
launch_sites = spacex_df['Launch Site'].unique()

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
    # Task 1: Dropdown to select launch site
    dcc.Dropdown(
        id='site-dropdown',
        options=[
            {'label': 'All Sites', 'value': 'ALL'},
            *[{'label': site, 'value': site} for site in launch_sites]
        ],
        value='ALL',  # Default value is 'ALL'
        placeholder="Select a Launch Site here",
        searchable=True
    ),
    html.Br(),

    # Task 2: Pie chart for success/failure count
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    # Task 3: Payload range slider
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        marks={i: f'{i}' for i in range(int(spacex_df['Payload Mass (kg)'].min()),
                                        int(spacex_df['Payload Mass (kg)'].max())+1, 1000)},
        value=[spacex_df['Payload Mass (kg)'].min(), spacex_df['Payload Mass (kg)'].max()],
    ),
    html.Br(),

    # Task 4: Scatter chart for payload vs. success
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])

# Task 2: Callback for the pie chart
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    # If 'ALL' sites are selected
    if entered_site == 'ALL':
        # Group by 'Launch Site' and calculate the total successes (class == 1)
        site_summary = spacex_df.groupby('Launch Site')['class'].sum().reset_index()
        
        # Create the pie chart for the total successes at each launch site
        fig = px.pie(site_summary, values='class', names='Launch Site', 
                     title='Total Launch Success by Site', 
                     color_discrete_map={'Launch Site': 'green'})

    else:
        # Filter the dataframe based on the selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]

    # Count successes (class == 1) and failures (class == 0)
        success_count = filtered_df[filtered_df['class'] == 1].shape[0]
        failure_count = filtered_df[filtered_df['class'] == 0].shape[0]

    # Prepare the data for the pie chart
        pie_data = {'Outcome': ['Success', 'Failure'], 'Count': [success_count, failure_count]}
        pie_df = pd.DataFrame(pie_data)

    # Create the pie chart using Plotly Express
        fig = px.pie(pie_df, values='Count', names='Outcome', title=f"Launch Success vs Failure - {entered_site}")
    
    return fig

# Task 4: Callback for the scatter plot
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_scatter_chart(selected_site, payload_range):
    # Filter data based on selected site
    if selected_site == 'ALL':
        filtered_df = spacex_df
    else:
        filtered_df = spacex_df[spacex_df['Launch Site'] == selected_site]
    
    # Further filter data based on payload range
    filtered_df = filtered_df[
        (filtered_df['Payload Mass (kg)'] >= payload_range[0]) &
        (filtered_df['Payload Mass (kg)'] <= payload_range[1])
    ]
    
    # Scatter plot between payload and launch success
    fig = px.scatter(
        filtered_df, 
        x='Payload Mass (kg)', 
        y='class', 
        color='Booster Version Category', 
        title=f"Correlation between Payload and Success for {selected_site if selected_site != 'ALL' else 'All Sites'}",
        labels={'class': 'class', 'Payload Mass (kg)': 'Payload Mass (kg)'}
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
