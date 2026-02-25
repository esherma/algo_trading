"""
Email reporting via generic SMTP.

Reads connection details from environment variables:
  SMTP_SERVER    – hostname (default: smtp.gmail.com)
  SMTP_PORT      – port    (default: 587)
  SMTP_USERNAME  – login / from address
  SMTP_PASSWORD  – password or app-specific password
  RECIPIENT_EMAIL – where to send the morning chart

Usage::

    from btm.email_report import send_morning_email
    png_bytes = chart.plot_morning_bands(...)
    send_morning_email(png_bytes, date_str="2024-01-15", symbol="SPY")
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv


def _smtp_cfg() -> dict:
    load_dotenv()
    return {
        "server":    os.getenv("SMTP_SERVER",   "smtp.gmail.com"),
        "port":      int(os.getenv("SMTP_PORT", "587")),
        "username":  os.getenv("SMTP_USERNAME", ""),
        "password":  os.getenv("SMTP_PASSWORD", ""),
        "recipient": os.getenv("RECIPIENT_EMAIL", ""),
    }


def send_morning_email(
    chart_png: bytes,
    date_str:  str,
    symbol:    str,
    leverage:  float,
    long_etf:  str,
    short_etf: str,
    daily_vol_pct: float,
) -> None:
    """
    Send the morning noise-band chart as an email attachment.

    Raises RuntimeError if SMTP credentials are missing.
    """
    cfg = _smtp_cfg()
    if not cfg["username"] or not cfg["password"]:
        raise RuntimeError(
            "SMTP credentials missing. Set SMTP_USERNAME and SMTP_PASSWORD in .env"
        )
    if not cfg["recipient"]:
        raise RuntimeError(
            "RECIPIENT_EMAIL not set in .env"
        )

    subject = f"[BTM] Trading Plan  {symbol}  {date_str}"

    body_html = f"""\
<html><body style="font-family: sans-serif; color: #333;">
<h2 style="margin-bottom:4px;">BTM Trading Plan — {symbol} — {date_str}</h2>
<table style="border-collapse:collapse; margin-bottom:12px;">
  <tr>
    <td style="padding:4px 12px 4px 0;"><b>Leverage</b></td>
    <td>{leverage:.2f}×</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;"><b>Long ETF</b></td>
    <td>{long_etf}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;"><b>Short ETF</b></td>
    <td>{short_etf}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;"><b>σ_SPY (14d)</b></td>
    <td>{daily_vol_pct:.2f}%</td>
  </tr>
</table>
<p>Decision times: <b>10:00, 10:30, … 15:30</b>.  Positions close at <b>15:50</b>.</p>
<p>See attached chart for today's noise bands.</p>
</body></html>
"""

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = cfg["username"]
    msg["To"]      = cfg["recipient"]

    # HTML body
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body_html, "html"))
    msg.attach(alt)

    # Chart as inline attachment
    img = MIMEImage(chart_png, name=f"btm_bands_{date_str}.png")
    img.add_header("Content-Disposition", "attachment",
                   filename=f"btm_bands_{date_str}.png")
    msg.attach(img)

    ctx = ssl.create_default_context()
    with smtplib.SMTP(cfg["server"], cfg["port"]) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ctx)
        smtp.login(cfg["username"], cfg["password"])
        smtp.sendmail(cfg["username"], cfg["recipient"], msg.as_string())

    print(f"Morning email sent to {cfg['recipient']}")
