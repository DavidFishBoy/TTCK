# src/monitoring/dashboard.py

import streamlit as st
import pandas as pd
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.visualization.visualizer import CryptoVisualizer

class MonitoringDashboard:
    """
    Comprehensive Streamlit dashboard for cryptocurrency prediction monitoring.
    Features:
    - Historical Data visualization from raw/train
    - Training Results and metrics
    - Future Predictions visualization
    """

    def __init__(self, data_dir: str = "data/raw/train", results_dir: str = "results"):
        st.set_page_config(
            page_title="Crypto Prediction Monitor", 
            page_icon="📈", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.logger = self._setup_logger()
        self.visualizer = CryptoVisualizer()
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.coins = [
            "bitcoin", 
            "ethereum", 
            "litecoin",
            "binancecoin",
            "cardano",
            "solana",
            "pancakeswap",
            "axieinfinity",
            "thesandbox"
        ]

    def _setup_logger(self):
        logger = logging.getLogger("MonitoringDashboard")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_historical_data(self, coin: str) -> pd.DataFrame:
        """Load historical data from raw/train directory."""
        csv_files = list(self.data_dir.glob(f"{coin}_binance_*.csv"))
        if not csv_files:
            self.logger.warning(f"No historical data found for {coin}")
            return pd.DataFrame()
        
        # Load the most recent file
        latest_file = sorted(csv_files)[-1]
        try:
            df = pd.read_csv(latest_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error loading {latest_file}: {e}")
            return pd.DataFrame()

    def load_results(self, coin: str) -> dict:
        """Load training results for a specific coin."""
        result_files = list(self.results_dir.glob(f"{coin}_results_*.json"))
        if not result_files:
            return {}
        
        latest_file = sorted(result_files)[-1]
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return {}

    def load_predictions(self, coin: str) -> dict:
        """Load future predictions for a specific coin."""
        pred_file = self.results_dir / "predictions" / f"{coin}_future_predictions.json"
        if not pred_file.exists():
            return {}
        
        try:
            with open(pred_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading predictions: {e}")
            return {}

    def plot_historical_data(self, df: pd.DataFrame, coin: str):
        """Plot historical OHLCV data."""
        if df.empty:
            st.warning(f"No historical data available for {coin}")
            return

        # 1. Main Candlestick chart with Volume
        st.subheader("📊 Price & Volume Overview")
        fig_main = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{coin.capitalize()} Price (OHLC)', 'Volume'),
            row_heights=[0.7, 0.3]
        )

        # Candlestick
        fig_main.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Volume bars
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                  for i in range(len(df))]
        fig_main.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )

        fig_main.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig_main, use_container_width=True)

        # 2. Individual OHLC Line Charts
        st.subheader("📈 OHLC Trends")
        fig_ohlc = go.Figure()
        
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['open'], 
            name='Open', mode='lines',
            line=dict(color='blue', width=1)
        ))
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['high'], 
            name='High', mode='lines',
            line=dict(color='green', width=1)
        ))
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['low'], 
            name='Low', mode='lines',
            line=dict(color='red', width=1)
        ))
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['close'], 
            name='Close', mode='lines',
            line=dict(color='black', width=2)
        ))
        
        fig_ohlc.update_layout(
            title=f"{coin.capitalize()} - OHLC Price Lines",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ohlc, use_container_width=True)

        # 3. Price Range (High-Low) Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📏 Daily Price Range")
            fig_range = go.Figure()
            
            # Calculate daily range
            df['range'] = df['high'] - df['low']
            df['range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
            
            fig_range.add_trace(go.Bar(
                x=df.index, 
                y=df['range'],
                name='Price Range',
                marker_color='purple'
            ))
            
            fig_range.update_layout(
                title="Daily High-Low Range",
                xaxis_title="Date",
                yaxis_title="Range (USDT)",
                height=350
            )
            
            st.plotly_chart(fig_range, use_container_width=True)
        
        with col2:
            st.subheader("📊 Volume Distribution")
            fig_vol_dist = go.Figure()
            
            fig_vol_dist.add_trace(go.Histogram(
                x=df['volume'],
                nbinsx=50,
                name='Volume Distribution',
                marker_color='lightblue'
            ))
            
            fig_vol_dist.update_layout(
                title="Volume Distribution",
                xaxis_title="Volume",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig_vol_dist, use_container_width=True)

        # 4. Price Statistics Cards
        st.subheader("💹 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_close = go.Figure()
            fig_close.add_trace(go.Indicator(
                mode="number+delta",
                value=df['close'].iloc[-1],
                title={'text': "Close Price"},
                delta={'reference': df['close'].iloc[-2], 'relative': True},
                number={'prefix': "$"}
            ))
            fig_close.update_layout(height=200)
            st.plotly_chart(fig_close, use_container_width=True)

        with col2:
            fig_high = go.Figure()
            fig_high.add_trace(go.Indicator(
                mode="number+delta",
                value=df['high'].iloc[-1],
                title={'text': "High Price"},
                delta={'reference': df['high'].iloc[-2], 'relative': True},
                number={'prefix': "$"}
            ))
            fig_high.update_layout(height=200)
            st.plotly_chart(fig_high, use_container_width=True)
        
        with col3:
            fig_low = go.Figure()
            fig_low.add_trace(go.Indicator(
                mode="number+delta",
                value=df['low'].iloc[-1],
                title={'text': "Low Price"},
                delta={'reference': df['low'].iloc[-2], 'relative': True},
                number={'prefix': "$"}
            ))
            fig_low.update_layout(height=200)
            st.plotly_chart(fig_low, use_container_width=True)

        with col4:
            fig_vol_ind = go.Figure()
            fig_vol_ind.add_trace(go.Indicator(
                mode="number+delta",
                value=df['volume'].iloc[-1],
                title={'text': "Volume"},
                delta={'reference': df['volume'].mean(), 'relative': True}
            ))
            fig_vol_ind.update_layout(height=200)
            st.plotly_chart(fig_vol_ind, use_container_width=True)

        # 5. Box Plots for OHLC
        st.subheader("📦 OHLC Distribution")
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(y=df['open'], name='Open', marker_color='blue'))
        fig_box.add_trace(go.Box(y=df['high'], name='High', marker_color='green'))
        fig_box.add_trace(go.Box(y=df['low'], name='Low', marker_color='red'))
        fig_box.add_trace(go.Box(y=df['close'], name='Close', marker_color='black'))
        
        fig_box.update_layout(
            title=f"{coin.capitalize()} - OHLC Box Plot",
            yaxis_title="Price (USDT)",
            height=400
        )
        
        st.plotly_chart(fig_box, use_container_width=True)

        # 6. Volume Analysis
        st.subheader("📊 Volume Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume over time
            fig_vol_time = go.Figure()
            fig_vol_time.add_trace(go.Scatter(
                x=df.index,
                y=df['volume'],
                fill='tozeroy',
                name='Volume',
                line=dict(color='lightblue')
            ))
            # Add moving average
            df['vol_ma'] = df['volume'].rolling(window=7).mean()
            fig_vol_time.add_trace(go.Scatter(
                x=df.index,
                y=df['vol_ma'],
                name='7-Day MA',
                line=dict(color='darkblue', width=2)
            ))
            
            fig_vol_time.update_layout(
                title="Volume Over Time",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=350
            )
            st.plotly_chart(fig_vol_time, use_container_width=True)
        
        with col2:
            # Price vs Volume correlation
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df['volume'],
                y=df['close'],
                mode='markers',
                name='Price vs Volume',
                marker=dict(
                    size=5,
                    color=df.index.astype('int64') / 10**9,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                )
            ))
            
            fig_corr.update_layout(
                title="Price vs Volume Correlation",
                xaxis_title="Volume",
                yaxis_title="Close Price (USDT)",
                height=350
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    def plot_training_results(self, results: dict, coin: str):
        """Plot training history and evaluation metrics."""
        if not results:
            st.warning(f"No training results available for {coin}")
            return

        # Metrics
        evaluation = results.get('evaluation', {})
        if evaluation:
            st.subheader("📊 Model Performance Metrics")
            cols = st.columns(len(evaluation))
            for idx, (metric, value) in enumerate(evaluation.items()):
                with cols[idx]:
                    st.metric(
                        label=metric.upper().replace('_', ' '),
                        value=f"{value:.4f}"
                    )

        # Training history
        history = results.get('history', {})
        if history:
            st.subheader("📈 Training History")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss', 'MAE'),
                horizontal_spacing=0.1
            )

            # Loss
            if 'loss' in history:
                fig.add_trace(
                    go.Scatter(y=history['loss'], name='Train Loss', mode='lines'),
                    row=1, col=1
                )
            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'),
                    row=1, col=1
                )

            # MAE
            if 'mae' in history:
                fig.add_trace(
                    go.Scatter(y=history['mae'], name='Train MAE', mode='lines'),
                    row=1, col=2
                )
            if 'val_mae' in history:
                fig.add_trace(
                    go.Scatter(y=history['val_mae'], name='Val MAE', mode='lines'),
                    row=1, col=2
                )

            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_layout(height=400, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)

        # Predictions vs Actual (if available)
        if 'predictions' in results and 'actual' in results:
            st.subheader("🎯 Predictions vs Actual")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['actual'],
                name='Actual',
                mode='lines',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=results['predictions'],
                name='Predicted',
                mode='lines',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{coin.capitalize()} - Test Set Predictions",
                xaxis_title="Time Step",
                yaxis_title="Price (USDT)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def plot_future_predictions(self, predictions: dict, coin: str):
        """Plot future price predictions."""
        if not predictions:
            st.warning(f"No future predictions available for {coin}")
            return

        st.subheader("🔮 Future Price Predictions")
        
        # Display prediction metadata
        if 'prediction_generated_at' in predictions:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📅 Generated: {predictions['prediction_generated_at']}")
            with col2:
                st.info(f"📊 Forecast Horizon: {predictions.get('forecast_horizon_days', 'N/A')} days")
        
        # Handle different prediction data formats
        pred_data = predictions.get('predictions', [])
        timestamps = predictions.get('timestamps', [])
        
        if not pred_data:
            st.warning("No prediction data found")
            return

        # Extract price values if pred_data contains dictionaries
        if isinstance(pred_data, list) and len(pred_data) > 0:
            if isinstance(pred_data[0], dict):
                # If predictions are in dict format like {'expected_price': value, 'day': ...}
                prices = [p.get('expected_price', p.get('price', p.get('predicted_price', 0))) for p in pred_data]
                if not timestamps:
                    # Try to get timestamps or day numbers
                    if 'timestamp' in pred_data[0]:
                        timestamps = [p['timestamp'] for p in pred_data]
                    elif 'day' in pred_data[0]:
                        timestamps = [f"Day {p['day']}" for p in pred_data]
            else:
                # If predictions are already numbers
                prices = pred_data
        else:
            st.warning("Invalid prediction data format")
            return

        # Create prediction chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps if timestamps else list(range(len(prices))),
            y=prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{coin.capitalize()} - Future Price Forecast",
            xaxis_title="Date/Time",
            yaxis_title="Price (USDT)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Prediction statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Min", f"${min(prices):.2f}")
        with col2:
            st.metric("Predicted Max", f"${max(prices):.2f}")
        with col3:
            avg_price = sum(prices) / len(prices)
            st.metric("Predicted Avg", f"${avg_price:.2f}")
        
        # Show prediction trend
        if len(prices) > 1 and prices[0] != 0:
            trend = ((prices[-1] - prices[0]) / prices[0]) * 100
            trend_text = "📈 Upward" if trend > 0 else "📉 Downward"
            st.info(f"Overall Trend: {trend_text} ({trend:+.2f}%)")
        elif len(prices) > 1:
            change = prices[-1] - prices[0]
            trend_text = "📈 Upward" if change > 0 else "📉 Downward" if change < 0 else "➡️ Flat"
            st.info(f"Overall Trend: {trend_text} (Change: ${change:.2f})")
        
        # Display explanation if available
        if 'explanation' in predictions:
            with st.expander("ℹ️ About These Predictions"):
                st.write(predictions['explanation'])
    
    def show(self):
        """Main dashboard display."""
        st.title("🚀 Cryptocurrency Prediction Dashboard")
        st.markdown("---")

        # Sidebar for coin selection
        st.sidebar.title("Settings")
        selected_coin = st.sidebar.selectbox(
            "Select Cryptocurrency",
            self.coins,
            format_func=lambda x: x.capitalize()
        )

        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Historical Data",
            "🎓 Training Results",
            "🔮 Future Predictions"
        ])

        with tab1:
            st.header(f"Historical Data - {selected_coin.capitalize()}")
            hist_data = self.load_historical_data(selected_coin)
            
            if not hist_data.empty:
                # Display summary statistics
                st.subheader("📈 Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(hist_data))
                with col2:
                    st.metric("Date Range", f"{(hist_data.index[-1] - hist_data.index[0]).days} days")
                with col3:
                    st.metric("Avg Price", f"${hist_data['close'].mean():.2f}")
                with col4:
                    st.metric("Price Volatility", f"{hist_data['close'].std():.2f}")
                
                st.markdown("---")
                self.plot_historical_data(hist_data, selected_coin)
                
                # Display raw data table
                with st.expander("📋 View Raw Data"):
                    st.dataframe(hist_data.tail(50))

        with tab2:
            st.header(f"Training Results - {selected_coin.capitalize()}")
            results = self.load_results(selected_coin)
            self.plot_training_results(results, selected_coin)

        with tab3:
            st.header(f"Future Predictions - {selected_coin.capitalize()}")
            predictions = self.load_predictions(selected_coin)
            self.plot_future_predictions(predictions, selected_coin)

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Crypto Prediction Dashboard | Powered by Deep Learning</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    dashboard = MonitoringDashboard()
    dashboard.show()


if __name__ == "__main__":
    main()