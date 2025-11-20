"""Interactive dashboard for IUU fishing detection"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/dashboard.log")

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load configuration
config = load_config()

# Load data
def load_data():
    """Load anomaly predictions"""
    try:
        predictions_path = Path("outputs/anomaly_predictions.csv")
        df = pd.read_csv(predictions_path, parse_dates=['timestamp'])
        logger.info(f"Loaded {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("IUU Fishing Anomaly Detection System", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        html.P("Real-time monitoring of vessel activities in Indian EEZ",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Anomaly Threshold:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='threshold-slider',
                min=0,
                max=1,
                step=0.05,
                value=0.7,
                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Select Vessel (MMSI):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='vessel-dropdown',
                options=[],
                value=None,
                placeholder="Select a vessel..."
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                       style={'backgroundColor': '#3498db', 'color': 'white', 
                             'padding': '10px 20px', 'border': 'none', 
                             'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px', 'textAlign': 'center'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Statistics cards
    html.Div([
        html.Div([
            html.H3(id='total-vessels', style={'color': '#3498db', 'margin': 0}),
            html.P("Total Vessels", style={'color': '#7f8c8d', 'margin': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 
                 'backgroundColor': '#ffffff', 'margin': '1%', 'textAlign': 'center',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3(id='anomaly-count', style={'color': '#e74c3c', 'margin': 0}),
            html.P("Anomalies Detected", style={'color': '#7f8c8d', 'margin': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px',
                 'backgroundColor': '#ffffff', 'margin': '1%', 'textAlign': 'center',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3(id='anomaly-rate', style={'color': '#f39c12', 'margin': 0}),
            html.P("Anomaly Rate", style={'color': '#7f8c8d', 'margin': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px',
                 'backgroundColor': '#ffffff', 'margin': '1%', 'textAlign': 'center',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3(id='avg-score', style={'color': '#9b59b6', 'margin': 0}),
            html.P("Avg Anomaly Score", style={'color': '#7f8c8d', 'margin': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px',
                 'backgroundColor': '#ffffff', 'margin': '1%', 'textAlign': 'center',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'borderRadius': '5px'})
    ], style={'marginBottom': '20px'}),
    
    # Main content
    html.Div([
        # Map
        html.Div([
            html.H3("Vessel Trajectories & Anomalies", style={'color': '#2c3e50'}),
            dcc.Graph(id='map-plot', style={'height': '600px'})
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px',
                 'backgroundColor': '#ffffff', 'verticalAlign': 'top',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'borderRadius': '5px'}),
        
        # Time series
        html.Div([
            html.H3("Anomaly Score Timeline", style={'color': '#2c3e50'}),
            dcc.Graph(id='timeline-plot', style={'height': '270px'}),
            
            html.H3("Model Scores Comparison", style={'color': '#2c3e50', 'marginTop': '20px'}),
            dcc.Graph(id='scores-plot', style={'height': '270px'})
        ], style={'width': '33%', 'display': 'inline-block', 'padding': '20px',
                 'backgroundColor': '#ffffff', 'verticalAlign': 'top', 'marginLeft': '1%',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'borderRadius': '5px'})
    ]),
    
    # Data store
    dcc.Store(id='data-store'),
    dcc.Interval(id='interval-component', interval=300*1000, n_intervals=0)  # 5 min
])

# Callbacks
@app.callback(
    [Output('data-store', 'data'),
     Output('vessel-dropdown', 'options')],
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_data(n_clicks, n_intervals):
    """Load and update data"""
    df = load_data()
    
    if df.empty:
        return {}, []
    
    # Vessel options
    vessels = sorted(df['MMSI'].unique())
    vessel_options = [{'label': f'MMSI: {v}', 'value': v} for v in vessels]
    
    return df.to_dict('records'), vessel_options

@app.callback(
    [Output('total-vessels', 'children'),
     Output('anomaly-count', 'children'),
     Output('anomaly-rate', 'children'),
     Output('avg-score', 'children')],
    [Input('data-store', 'data'),
     Input('threshold-slider', 'value')]
)
def update_stats(data, threshold):
    """Update statistics cards"""
    if not data:
        return "0", "0", "0%", "0.00"
    
    df = pd.DataFrame(data)
    
    total_vessels = df['MMSI'].nunique()
    anomalies = (df['ensemble_score'] >= threshold).sum()
    anomaly_rate = f"{anomalies/len(df)*100:.1f}%"
    avg_score = f"{df['ensemble_score'].mean():.3f}"
    
    return str(total_vessels), str(anomalies), anomaly_rate, avg_score

@app.callback(
    Output('map-plot', 'figure'),
    [Input('data-store', 'data'),
     Input('threshold-slider', 'value'),
     Input('vessel-dropdown', 'value')]
)
def update_map(data, threshold, selected_vessel):
    """Update map visualization"""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # Filter by vessel if selected
    if selected_vessel:
        df = df[df['MMSI'] == selected_vessel]
    
    # Mark anomalies
    df['is_anomaly'] = df['ensemble_score'] >= threshold
    
    # Create map
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='is_anomaly',
        color_discrete_map={True: 'red', False: 'blue'},
        hover_data=['MMSI', 'timestamp', 'ensemble_score'],
        zoom=4,
        height=600
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": df['lat'].mean(), "lon": df['lon'].mean()},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True,
        legend=dict(title="Anomaly Status", orientation="h", y=1, x=0.5, xanchor='center')
    )
    
    return fig

@app.callback(
    Output('timeline-plot', 'figure'),
    [Input('data-store', 'data'),
     Input('vessel-dropdown', 'value')]
)
def update_timeline(data, selected_vessel):
    """Update timeline plot"""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if selected_vessel:
        df = df[df['MMSI'] == selected_vessel]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['ensemble_score'],
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='#e74c3c')
    ))
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        hovermode='x unified',
        margin=dict(l=40, r=20, t=20, b=40)
    )
    
    return fig

@app.callback(
    Output('scores-plot', 'figure'),
    [Input('data-store', 'data'),
     Input('vessel-dropdown', 'value')]
)
def update_scores(data, selected_vessel):
    """Update model scores comparison"""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    if selected_vessel:
        df = df[df['MMSI'] == selected_vessel]
    
    # Sample data for visualization
    df_sample = df.sample(min(100, len(df)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sample.index,
        y=df_sample['supervised_score'],
        mode='lines',
        name='Supervised',
        line=dict(color='#3498db')
    ))
    fig.add_trace(go.Scatter(
        x=df_sample.index,
        y=df_sample['unsupervised_score'],
        mode='lines',
        name='Unsupervised',
        line=dict(color='#2ecc71')
    ))
    fig.add_trace(go.Scatter(
        x=df_sample.index,
        y=df_sample['ensemble_score'],
        mode='lines',
        name='Ensemble',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Sample Index",
        yaxis_title="Score",
        hovermode='x unified',
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
    )
    
    return fig

def main():
    """Run dashboard"""
    host = config.get('dashboard', 'host', default='0.0.0.0')
    port = config.get('dashboard', 'port', default=8050)
    
    logger.info(f"Starting dashboard at http://{host}:{port}")
    app.run_server(debug=True, host=host, port=port)

if __name__ == '__main__':
    main()
