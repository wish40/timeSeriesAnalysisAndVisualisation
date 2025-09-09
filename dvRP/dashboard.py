import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output

# --- 1. Load and Prepare Data ---
try:
    df_reliance = pd.read_excel('./DV_ResearchPaper.xlsx', sheet_name="Final Data")
    df_nifty = pd.read_csv('nifty50_data.csv', skiprows=2) # Skip the first two rows

    # Process df_reliance
    df_reliance['date'] = pd.to_datetime(df_reliance['date'])
    df_reliance['close'] = pd.to_numeric(df_reliance['close'], errors='coerce')
    df_reliance['MA20'] = df_reliance['close'].rolling(window=20).mean()
    df_reliance['MA50'] = df_reliance['close'].rolling(window=50).mean()
    df_reliance['BB_Mid'] = df_reliance['close'].rolling(window=20).mean()
    df_reliance['BB_Upper'] = df_reliance['BB_Mid'] + 2 * df_reliance['close'].rolling(window=20).std()
    df_reliance['BB_Lower'] = df_reliance['BB_Mid'] - 2 * df_reliance['close'].rolling(window=20).std()
    delta_reliance = df_reliance['close'].diff()
    gain_reliance = (delta_reliance.where(delta_reliance > 0, 0)).rolling(window=14).mean()
    loss_reliance = (-delta_reliance.where(delta_reliance < 0, 0)).rolling(window=14).mean()
    rs_reliance = gain_reliance / loss_reliance
    df_reliance['RSI'] = 100 - (100 / (1 + rs_reliance))


    # Process df_nifty - using 'Date' as the date column and 'Unnamed: 1' as close
    df_nifty['date'] = pd.to_datetime(df_nifty['Date']) # Use 'Date' column
    df_nifty['close'] = pd.to_numeric(df_nifty['Unnamed: 1'], errors='coerce') # Use 'Unnamed: 1' as close price column
    # Calculate moving averages for Nifty (optional, remove if not needed for comparison)
    df_nifty['MA20'] = df_nifty['close'].rolling(window=20).mean()
    df_nifty['MA50'] = df_nifty['close'].rolling(window=50).mean()
    # Calculate Bollinger Bands for Nifty (optional)
    df_nifty['BB_Mid'] = df_nifty['close'].rolling(window=20).mean()
    df_nifty['BB_Upper'] = df_nifty['BB_Mid'] + 2 * df_nifty['close'].rolling(window=20).std()
    df_nifty['BB_Lower'] = df_nifty['BB_Mid'] - 2 * df_nifty['close'].rolling(window=20).std()
    # Calculate RSI for Nifty (optional)
    delta_nifty = df_nifty['close'].diff()
    gain_nifty = (delta_nifty.where(delta_nifty > 0, 0)).rolling(window=14).mean()
    loss_nifty = (-delta_nifty.where(delta_nifty < 0, 0)).rolling(window=14).mean()
    rs_nifty = gain_nifty / loss_nifty
    df_nifty['RSI'] = 100 - (100 / (1 + rs_nifty))


except FileNotFoundError as e:
    print(f"Error loading data file: {e}. Please ensure both CSV files are present.")
    # Create dummy dataframes to allow app to launch
    df_reliance = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'Open': [0], 'High': [0], 'Low': [0], 'close': [0], 'Volume': [0], 'MA20': [0], 'MA50': [0], 'BB_Upper': [0], 'BB_Lower': [0], 'RSI': [50]})
    df_nifty = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'close': [0]})

# Merge data for comparison
df_comparison = pd.merge(df_reliance[['date', 'close']], df_nifty[['date', 'close']], on='date', suffixes=('_RELIANCE', '_NIFTY'))


# --- 2. Initialize the Dash App ---
# The 'assets_folder' is automatically detected by Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'Professional Financial Dashboard'
server = app.server

# --- 3. Define App Layout ---
app.layout = html.Div(className='app-container', children=[
    # Header
    html.Div(className='header', children=[
        html.H1('Reliance Industries Financial Dashboard'),
        html.P(f"Last Data Point: {df_reliance['date'].max().strftime('%B %d, %Y')}")
    ]),

    # KPI Row
    html.Div(id='kpi-cards-row', className='row kpi-row'),

    # Main content with Controls and Tabs
    html.Div(className='row main-content', children=[
        # Control Panel
        html.Div(className='three columns control-panel', children=[
            html.H4("Analysis Controls"),
            html.Label("Date Range Selector"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=df_reliance['date'].min(),
                max_date_allowed=df_reliance['date'].max(),
                start_date=df_reliance['date'].max() - pd.DateOffset(years=1),
                end_date=df_reliance['date'].max(),
                className='date-picker'
            ),
            html.Div(className='control-section', children=[
                html.Label("Chart Type"),
                dcc.RadioItems(
                    id='chart-type-radio',
                    options=[{'label': 'Candlestick', 'value': 'candlestick'}, {'label': 'Line', 'value': 'line'}],
                    value='candlestick',
                    labelClassName='radio-label'
                ),
            ]),
            html.Div(className='control-section', children=[
                html.Label("Technical Indicators"),
                dcc.Checklist(
                    id='indicator-checklist',
                    options=[
                        {'label': 'Moving Averages (20, 50)', 'value': 'MA'},
                        {'label': 'Bollinger Bands', 'value': 'BB'},
                        {'label': 'RSI Plot', 'value': 'RSI'}
                    ],
                    value=['MA', 'RSI'],
                    labelClassName='checklist-label'
                ),
            ]),
        ]),

        # Tabs for different views
        html.Div(className='nine columns chart-area', children=[
            dcc.Tabs(id='analysis-tabs', value='tab-overview', children=[
                dcc.Tab(label='Price Overview & Technicals', value='tab-overview', children=[
                    dcc.Graph(id='main-price-chart', config={'displayModeBar': False}),
                    dcc.Graph(id='volume-chart', config={'displayModeBar': False})
                ]),
                dcc.Tab(label='Performance Comparison', value='tab-comparison', children=[
                    dcc.Graph(id='comparison-chart', config={'displayModeBar': False})
                ]),
                dcc.Tab(label='Raw Data Table', value='tab-data', children=[
                    html.Div(id='data-table-container', className='data-table')
                ])
            ])
        ])
    ])
])

# --- 4. Callbacks for Interactivity ---
@app.callback(
    [Output('kpi-cards-row', 'children'),
     Output('main-price-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('volume-chart', 'style'),
     Output('comparison-chart', 'figure'),
     Output('data-table-container', 'children')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('chart-type-radio', 'value'),
     Input('indicator-checklist', 'value')]
)
def update_dashboard(start_date, end_date, chart_type, indicators):
    # Filter all dataframes based on the selected date range
    dff_reliance = df_reliance[(df_reliance['date'] >= start_date) & (df_reliance['date'] <= end_date)].copy()
    dff_comp = df_comparison[(df_comparison['date'] >= start_date) & (df_comparison['date'] <= end_date)].copy()

    # --- 1. KPI Calculations ---
    latest_close = dff_reliance['close'].iloc[-1]
    prev_close = dff_reliance['close'].iloc[-2]
    change = latest_close - prev_close
    percent_change = (change / prev_close) * 100
    range_high = dff_reliance['high'].max() # Corrected column name to lowercase 'high'
    range_low = dff_reliance['low'].min() # Corrected column name to lowercase 'low'


    kpi_values = {
        "Current Price": (f"₹{latest_close:,.2f}", ""),
        "Price Change": (f"{change:+.2f} ({percent_change:+.2f}%)", "positive" if change >= 0 else "negative"),
        "Period High": (f"₹{range_high:,.2f}", ""),
        "Period Low": (f"₹{range_low:,.2f}", "")
    }

    kpi_cards = []
    for title, (value, color_class) in kpi_values.items():
        kpi_cards.append(
            html.Div(className='three columns', children=html.Div(className='kpi-card', children=[
                html.H5(title),
                html.H3(value, className=color_class)
            ]))
        )

    # --- 2. Main Price Chart ---
    show_rsi = 'RSI' in indicators
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3] if show_rsi else [1, 0])

    # Add Price Trace (Candlestick or Line)
    if chart_type == 'candlestick':
        fig_price.add_trace(go.Candlestick(x=dff_reliance['date'], open=dff_reliance['open'], high=dff_reliance['high'], low=dff_reliance['low'], close=dff_reliance['close'], name='Price'), row=1, col=1) # Corrected column names to lowercase
    else:
        fig_price.add_trace(go.Scatter(x=dff_reliance['date'], y=dff_reliance['close'], mode='lines', name='Close Price', line={'color': '#007bff'}), row=1, col=1)

    # Add Technical Indicators
    if 'MA' in indicators:
        fig_price.add_trace(go.Scatter(x=dff_reliance['date'], y=dff_reliance['MA20'], mode='lines', name='20-Day MA', line={'color': '#ff7f0e', 'width': 1.5}), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=dff_reliance['date'], y=dff_reliance['MA50'], mode='lines', name='50-Day MA', line={'color': '#9467bd', 'width': 1.5}), row=1, col=1)
    if 'BB' in indicators:
        fig_price.add_trace(go.Scatter(x=dff_reliance['date'], y=dff_reliance['BB_Upper'], mode='lines', name='Upper Band', line={'color': '#d3d3d3', 'width': 1, 'dash': 'dash'}), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=dff_reliance['date'], y=dff_reliance['BB_Lower'], mode='lines', name='Lower Band', line={'color': '#d3d3d3', 'width': 1, 'dash': 'dash'}, fill='tonexty', fillcolor='rgba(211,211,211,0.1)'), row=1, col=1)

    # Add RSI Plot if selected
    if show_rsi:
        fig_price.add_trace(go.Scatter(x=dff_reliance['date'], y=dff_reliance['RSI'], mode='lines', name='RSI', line={'color': '#17becf'}), row=2, col=1)
        fig_price.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.2, row=2, col=1)
        fig_price.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.2, row=2, col=1)
        fig_price.update_yaxes(title_text="RSI", row=2, col=1)

    fig_price.update_layout(title_text="Price and Technical Indicators", template='plotly_white', xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_price.update_yaxes(title_text="Price (INR)", row=1, col=1)

    # --- 3. Volume Chart ---
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=dff_reliance['date'], y=dff_reliance['volume'], name='Volume', marker_color='#adb5bd')) # Corrected column name to lowercase 'volume'
    fig_volume.update_layout(title_text="Trading Volume", template='plotly_white', xaxis_title='date', yaxis_title='Volume')

    # --- 4. Comparison Chart ---
    # Normalize data to compare performance
    dff_comp['RELIANCE_Norm'] = (dff_comp['close_RELIANCE'] / dff_comp['close_RELIANCE'].iloc[0]) * 100 # Corrected column name to lowercase 'close'
    dff_comp['NIFTY_Norm'] = (dff_comp['close_NIFTY'] / dff_comp['close_NIFTY'].iloc[0]) * 100 # Corrected column name to lowercase 'close'

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=dff_comp['date'], y=dff_comp['RELIANCE_Norm'], mode='lines', name='Reliance', line={'color': '#007bff'}))
    fig_comp.add_trace(go.Scatter(x=dff_comp['date'], y=dff_comp['NIFTY_Norm'], mode='lines', name='NIFTY 50', line={'color': '#ff7f0e'}))
    fig_comp.update_layout(title_text="Performance vs. NIFTY 50 (Normalized)", template='plotly_white', xaxis_title='date', yaxis_title='Normalized Price (Base 100)')

    # --- 5. Data Table ---
    dff_reliance_display = dff_reliance[['date', 'open', 'high', 'low', 'close', 'volume']].copy() # Corrected column names to lowercase
    dff_reliance_display['date'] = dff_reliance_display['date'].dt.strftime('%Y-%m-%d')
    for col in ['open', 'high', 'low', 'close']: # Corrected column names to lowercase
        dff_reliance_display[col] = dff_reliance_display[col].round(2)

    data_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in dff_reliance_display.columns],
        data=dff_reliance_display.to_dict('records'),
        page_size=10,
        sort_action="native",
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
    )

    volume_style = {'display': 'block'}
    if chart_type == 'candlestick':
        volume_style['height'] = '250px'

    return kpi_cards, fig_price, fig_volume, volume_style, fig_comp, data_table

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)