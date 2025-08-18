import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
import numpy as np
import pytz
from dotenv import load_dotenv

# Local imports
from btm_utils import TradingConfig, compute_noise_bands, compute_daily_ohlcv


class BTMAlertSystem:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.tz = pytz.timezone(config.timezone)
        load_dotenv()
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_address = os.getenv("ALERT_EMAIL_ADDRESS")
        self.email_password = os.getenv("ALERT_EMAIL_PASSWORD")
        self.recipient_email = os.getenv("RECIPIENT_EMAIL")
        
        if not all([self.email_address, self.email_password, self.recipient_email]):
            raise ValueError("Missing email configuration. Set ALERT_EMAIL_ADDRESS, ALERT_EMAIL_PASSWORD, and RECIPIENT_EMAIL in .env")
        
        # Trading data storage
        self.daily_trades = []
        self.opening_aum = 0.0
        self.closing_aum = 0.0
        self.daily_volatility = 0.0
        self.daily_leverage = 1.0
        self.in_play_tickers = {"long_ticker": "SPY", "short_ticker": "SPDN"}
        self.noise_bands = None
        self.historical_data = None
        
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Add a trade to the daily tracking."""
        self.daily_trades.append(trade_data)
    
    def set_daily_metrics(self, volatility: float, leverage: float, tickers: Dict[str, str]) -> None:
        """Set the daily pre-calculated metrics."""
        self.daily_volatility = volatility
        self.daily_leverage = leverage
        self.in_play_tickers = tickers
    
    def set_market_data(self, historical_data: pd.DataFrame, noise_bands: pd.DataFrame) -> None:
        """Set the market data for plotting."""
        self.historical_data = historical_data
        self.noise_bands = noise_bands
    
    def set_account_values(self, opening_aum: float, closing_aum: float) -> None:
        """Set the account values for the day."""
        self.opening_aum = opening_aum
        self.closing_aum = closing_aum
    
    def create_morning_alert(self) -> str:
        """Create the morning alert email content."""
        today = datetime.now(self.tz).strftime("%Y-%m-%d")
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007bff; }}
                .ticker {{ font-weight: bold; color: #28a745; }}
                .warning {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚀 BTM Trading Strategy - Morning Setup Alert</h2>
                <p><strong>Date:</strong> {today}</p>
                <p><strong>Session:</strong> {self.config.session.upper()}</p>
            </div>
            
            <div class="metric">
                <h3>📊 Daily Pre-Calculated Metrics</h3>
                <p><strong>Daily Volatility:</strong> {self.daily_volatility:.4f} ({self.daily_volatility*100:.2f}%)</p>
                <p><strong>Daily Leverage:</strong> {self.daily_leverage:.2f}x</p>
            </div>
            
            <div class="metric">
                <h3>🎯 Trading Tickers for Today</h3>
                <p><strong>Long Position Ticker:</strong> <span class="ticker">{self.in_play_tickers['long_ticker']}</span></p>
                <p><strong>Short Position Ticker:</strong> <span class="ticker">{self.in_play_tickers['short_ticker']}</span></p>
            </div>
            
            <div class="metric">
                <h3>⚙️ Strategy Configuration</h3>
                <p><strong>Base Symbol:</strong> {self.config.symbol}</p>
                <p><strong>Leverage Cap:</strong> {self.config.leverage_cap}x</p>
                <p><strong>Target Daily Volatility:</strong> {self.config.target_daily_volatility*100:.1f}%</p>
                <p><strong>Lookback Days:</strong> {self.config.lookback_days}</p>
                <p><strong>Volatility Multiplier:</strong> {self.config.volatility_multiplier}</p>
            </div>
            
            <p><em>Noise bands chart is attached to this email.</em></p>
            
            <div class="warning">
                <p><strong>⚠️ Trading will begin at {self.config.start_trading_time} ET</strong></p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def create_evening_digest(self) -> str:
        """Create the evening digest email content."""
        today = datetime.now(self.tz).strftime("%Y-%m-%d")
        
        # Calculate daily PnL
        daily_pnl = self.closing_aum - self.opening_aum
        daily_pnl_pct = (daily_pnl / self.opening_aum * 100) if self.opening_aum > 0 else 0
        
        # Format trades
        trades_html = ""
        if self.daily_trades:
            for i, trade in enumerate(self.daily_trades, 1):
                trade_time = trade.get('timestamp', 'Unknown').strftime('%H:%M:%S') if hasattr(trade.get('timestamp', ''), 'strftime') else str(trade.get('timestamp', 'Unknown'))
                trades_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{trade_time}</td>
                    <td>{trade.get('action', 'Unknown')}</td>
                    <td>{trade.get('ticker', 'Unknown')}</td>
                    <td>{trade.get('shares', 0)}</td>
                    <td>${trade.get('price', 0):.2f}</td>
                    <td>${trade.get('value', 0):,.2f}</td>
                </tr>
                """
        else:
            trades_html = "<tr><td colspan='7'>No trades executed today</td></tr>"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007bff; }}
                .pnl-positive {{ color: #28a745; font-weight: bold; }}
                .pnl-negative {{ color: #dc3545; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📈 BTM Trading Strategy - Evening Digest</h2>
                <p><strong>Date:</strong> {today}</p>
                <p><strong>Session:</strong> {self.config.session.upper()}</p>
            </div>
            
            <div class="metric">
                <h3>💰 Account Performance</h3>
                <p><strong>Opening AUM:</strong> ${self.opening_aum:,.2f}</p>
                <p><strong>Closing AUM:</strong> ${self.closing_aum:,.2f}</p>
                <p><strong>Daily P&L:</strong> 
                    <span class="{'pnl-positive' if daily_pnl >= 0 else 'pnl-negative'}">
                        ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)
                    </span>
                </p>
            </div>
            
            <div class="metric">
                <h3>📊 Daily Metrics</h3>
                <p><strong>Daily Volatility:</strong> {self.daily_volatility:.4f} ({self.daily_volatility*100:.2f}%)</p>
                <p><strong>Daily Leverage:</strong> {self.daily_leverage:.2f}x</p>
                <p><strong>Number of Trades:</strong> {len(self.daily_trades)}</p>
            </div>
            
            <div class="metric">
                <h3>🔄 Trading Activity</h3>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Time</th>
                            <th>Action</th>
                            <th>Ticker</th>
                            <th>Shares</th>
                            <th>Price</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades_html}
                    </tbody>
                </table>
            </div>
            
            <p><em>Detailed noise bands chart with trade annotations is attached to this email.</em></p>
        </body>
        </html>
        """
        
        return html_content
    
    def create_noise_bands_plot(self, include_trades: bool = False) -> str:
        """Create a noise bands plot and save it to a file."""
        if self.historical_data is None or self.noise_bands is None:
            return None
        
        # Get data for today's plot ... based on noise bands as of YESTERDAY's data
        last_trading_day = self.noise_bands.iloc[-1].name.date()
        last_trading_day_mask = self.historical_data.index.date == last_trading_day
        last_trading_day_data = self.historical_data[last_trading_day_mask]
        last_trading_day_bands = self.noise_bands[last_trading_day_mask]
        
        if len(last_trading_day_data) == 0:
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot price
        ax.plot(last_trading_day_data.index, last_trading_day_data['close'], label='SPY Price', color='black', linewidth=2)
        
        # Plot noise bands
        ax.plot(last_trading_day_bands.index, last_trading_day_bands['UB'], label='Upper Band', color='red', linestyle='--', alpha=0.7)
        ax.plot(last_trading_day_bands.index, last_trading_day_bands['LB'], label='Lower Band', color='red', linestyle='--', alpha=0.7)
        
        # Fill the bands area
        ax.fill_between(last_trading_day_bands.index, last_trading_day_bands['LB'], last_trading_day_bands['UB'], 
                       alpha=0.1, color='red', label='Noise Bands')
        
        # Add trade annotations if requested
        if include_trades and self.daily_trades:
            for trade in self.daily_trades:
                trade_time = trade.get('timestamp')
                if trade_time and hasattr(trade_time, 'tz_localize'):
                    if trade_time.tz is None:
                        trade_time = trade_time.tz_localize(self.tz)
                    elif trade_time.tz != self.tz:
                        trade_time = trade_time.tz_convert(self.tz)
                    
                    # Find closest time in today's data
                    time_diff = abs(last_trading_day_data.index - trade_time)
                    if len(time_diff) > 0:
                        closest_idx = time_diff.argmin()
                        closest_time = last_trading_day_data.index[closest_idx]
                        price_at_time = last_trading_day_data.loc[closest_time, 'close']
                        
                        # Plot trade marker
                        action = trade.get('action', 'Unknown')
                        if 'open' in action.lower():
                            marker = '^'
                            color = 'green'
                            label = 'Position Opened'
                        elif 'close' in action.lower():
                            marker = 'v'
                            color = 'red'
                            label = 'Position Closed'
                        else:
                            marker = 'o'
                            color = 'blue'
                            label = 'Trade'
                        
                        ax.scatter(closest_time, price_at_time, marker=marker, s=100, 
                                 color=color, zorder=5, label=label)
                        
                        # Add annotation
                        ax.annotate(f"{trade.get('ticker', 'Unknown')}\n{trade.get('shares', 0)} shares", 
                                  (closest_time, price_at_time), 
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                  fontsize=8)
        
        # Format the plot
        ax.set_title(f'BTM Noise Bands - {datetime.now().date().strftime("%Y-%m-%d")}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('SPY Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.xticks(rotation=45)
        
        # Save the plot
        plot_filename = os.path.join(os.getcwd(), f"noise_bands_{datetime.now().date().strftime('%Y%m%d')}.png")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename
    
    def send_email(self, subject: str, html_content: str, attachment_path: Optional[str] = None) -> bool:
        """Send an email with optional attachment."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_address
            msg['To'] = self.recipient_email
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as f:
                    attachment = MIMEImage(f.read())
                    attachment.add_header('Content-Disposition', 'attachment', 
                                        filename=os.path.basename(attachment_path))
                    msg.attach(attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            
            print(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_morning_alert(self) -> bool:
        """Send the morning alert email."""
        subject = f"BTM Trading Alert - Morning Setup - {datetime.now(self.tz).strftime('%Y-%m-%d')}"
        html_content = self.create_morning_alert()
        
        # Create and attach noise bands plot
        plot_filename = self.create_noise_bands_plot(include_trades=False)
        print('Saving plot to', plot_filename)
        
        success = self.send_email(subject, html_content, plot_filename)
        
        # Clean up plot file
        if plot_filename and os.path.exists(plot_filename):
            os.remove(plot_filename)
        
        return success
    
    def send_evening_digest(self) -> bool:
        """Send the evening digest email."""
        subject = f"BTM Trading Digest - {datetime.now(self.tz).strftime('%Y-%m-%d')}"
        html_content = self.create_evening_digest()
        
        # Create and attach noise bands plot with trade annotations
        plot_filename = self.create_noise_bands_plot(include_trades=True)
        
        success = self.send_email(subject, html_content, plot_filename)
        
        # Clean up plot file
        if plot_filename and os.path.exists(plot_filename):
            os.remove(plot_filename)
        
        return success


def test_email_configuration():
    """Test the email configuration."""
    try:
        config = TradingConfig()
        alert_system = BTMAlertSystem(config)
        
        # Test with sample data
        alert_system.set_daily_metrics(0.025, 2.5, {"long_ticker": "SPUU", "short_ticker": "SDS"})
        alert_system.set_account_values(100000.0, 101500.0)
        
        # Add sample trade
        alert_system.add_trade({
            'timestamp': datetime.now(),
            'action': 'Open Long',
            'ticker': 'SPUU',
            'shares': 100,
            'price': 45.50,
            'value': 4550.0
        })
        
        print("Email configuration test successful!")
        return True
        
    except Exception as e:
        print(f"Email configuration test failed: {e}")
        return False


if __name__ == "__main__":
    test_email_configuration()
