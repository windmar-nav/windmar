"""
PDF report generator for voyage reports.

Uses fpdf2 (pure Python, no system dependencies) to generate
professional maritime voyage reports.
"""

import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fpdf import FPDF, XPos, YPos

from api.models import Voyage, VoyageLeg

logger = logging.getLogger(__name__)


def _safe(text: str) -> str:
    """Sanitize text for Latin-1 PDF fonts (replace unsupported Unicode chars)."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00b0": "deg", # degree sign
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Strip any remaining non-latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


class VoyagePDF(FPDF):
    """Custom PDF class for voyage reports."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "WINDMAR - Voyage Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, _safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def kv_row(self, key: str, value: str, key_width: int = 55):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(60, 60, 60)
        self.cell(key_width, 6, _safe(key), new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, _safe(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_voyage_pdf(
    voyage: Voyage,
    noon_reports: List[dict],
    departure_report: dict,
    arrival_report: dict,
) -> bytes:
    """Generate a complete PDF report for a voyage.

    Args:
        voyage: Persisted voyage with legs loaded.
        noon_reports: List of noon report dicts (from noon_reports module).
        departure_report: Departure report dict (from templates module).
        arrival_report: Arrival report dict (from templates module).

    Returns:
        PDF file contents as bytes.
    """
    pdf = VoyagePDF()
    pdf.alias_nb_pages()

    _add_cover_page(pdf, voyage, departure_report, arrival_report)
    _add_voyage_summary(pdf, voyage, arrival_report)
    _add_leg_details(pdf, voyage)
    if noon_reports:
        _add_noon_reports(pdf, noon_reports)
    _add_weather_summary(pdf, arrival_report)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _fmt_num(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _add_cover_page(pdf: VoyagePDF, voyage: Voyage, dep: dict, arr: dict):
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 50, 100)
    pdf.ln(20)
    pdf.cell(0, 12, "Voyage Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # Voyage name
    if voyage.name:
        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 10, _safe(voyage.name), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(10)

    # Key info box
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)

    info_lines = [
        ("Vessel", dep.get("vessel_name") or "Default Vessel"),
        ("Departure", f"{voyage.departure_port or 'N/A'} - {_fmt_dt(voyage.departure_time)}"),
        ("Arrival", f"{voyage.arrival_port or 'N/A'} - {_fmt_dt(voyage.arrival_time)}"),
        ("Distance", f"{voyage.total_distance_nm:.1f} NM"),
        ("Duration", f"{voyage.total_time_hours:.1f} hours ({voyage.total_time_hours / 24:.1f} days)"),
        ("Total Fuel", f"{voyage.total_fuel_mt:.2f} MT"),
        ("Avg SOG", f"{_fmt_num(voyage.avg_sog_kts)} kts"),
        ("Condition", "Laden" if voyage.is_laden else "Ballast"),
    ]

    for key, value in info_lines:
        pdf.kv_row(key, value, 40)

    # CII estimate
    cii = voyage.cii_estimate
    if cii:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 8, "CII Estimate", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.kv_row("Rating", str(cii.get("rating", "N/A")), 40)
        pdf.kv_row("Attained CII", _fmt_num(cii.get("attained_cii"), 4), 40)
        pdf.kv_row("Required CII", _fmt_num(cii.get("required_cii"), 4), 40)


def _add_voyage_summary(pdf: VoyagePDF, voyage: Voyage, arr: dict):
    pdf.add_page()
    pdf.section_title("Voyage Summary")

    pdf.kv_row("Route", voyage.name or "Unnamed Route")
    pdf.kv_row("Departure", f"{voyage.departure_port or 'N/A'} at {_fmt_dt(voyage.departure_time)}")
    pdf.kv_row("Arrival", f"{voyage.arrival_port or 'N/A'} at {_fmt_dt(voyage.arrival_time)}")
    pdf.kv_row("Total Distance", f"{voyage.total_distance_nm:.1f} NM")
    pdf.kv_row("Total Time", f"{voyage.total_time_hours:.1f} hours")
    pdf.kv_row("Total Fuel", f"{voyage.total_fuel_mt:.2f} MT")
    pdf.kv_row("Avg SOG", f"{_fmt_num(voyage.avg_sog_kts)} kts")
    pdf.kv_row("Avg STW", f"{_fmt_num(voyage.avg_stw_kts)} kts")
    pdf.kv_row("Calm Speed", f"{voyage.calm_speed_kts:.1f} kts")
    pdf.kv_row("Condition", "Laden" if voyage.is_laden else "Ballast")
    pdf.kv_row("Number of Legs", str(len(voyage.legs)))


def _add_leg_details(pdf: VoyagePDF, voyage: Voyage):
    pdf.add_page()
    pdf.section_title("Leg Details")

    legs = sorted(voyage.legs, key=lambda l: l.leg_index)

    # Table header
    headers = ["#", "From", "To", "Dist(NM)", "SOG(kts)", "Fuel(MT)", "Time(h)", "Wind(kts)", "Wave(m)"]
    widths = [8, 30, 30, 20, 20, 20, 18, 22, 22]

    def _draw_header():
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(230, 235, 245)
        for i, h in enumerate(headers):
            pdf.cell(widths[i], 6, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(0, 0, 0)

    _draw_header()

    for leg in legs:
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.section_title("Leg Details (continued)")
            _draw_header()

        from_label = _safe((leg.from_name or "")[:12] or f"{leg.from_lat:.2f},{leg.from_lon:.2f}")
        to_label = _safe((leg.to_name or "")[:12] or f"{leg.to_lat:.2f},{leg.to_lon:.2f}")

        values = [
            str(leg.leg_index),
            from_label[:12],
            to_label[:12],
            _fmt_num(leg.distance_nm, 1),
            _fmt_num(leg.sog_kts, 1),
            _fmt_num(leg.fuel_mt, 2),
            _fmt_num(leg.time_hours, 1),
            _fmt_num(leg.wind_speed_kts, 1),
            _fmt_num(leg.wave_height_m, 1),
        ]

        for i, v in enumerate(values):
            pdf.cell(widths[i], 5, v, border=1, align="C")
        pdf.ln()


def _add_noon_reports(pdf: VoyagePDF, noon_reports: List[dict]):
    pdf.add_page()
    pdf.section_title("Noon Reports (24h intervals)")

    headers = ["#", "Date/Time", "Lat", "Lon", "SOG", "Dist(NM)", "Fuel(MT)", "Wind", "Wave"]
    widths = [8, 35, 18, 18, 16, 22, 22, 22, 22]

    def _draw_header():
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(230, 235, 245)
        for i, h in enumerate(headers):
            pdf.cell(widths[i], 6, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)

    _draw_header()

    for nr in noon_reports:
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.section_title("Noon Reports (continued)")
            _draw_header()

        ts = nr.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime) else str(ts)[:16]

        values = [
            str(nr.get("report_number", "")),
            ts_str,
            _fmt_num(nr.get("lat"), 2),
            _fmt_num(nr.get("lon"), 2),
            _fmt_num(nr.get("sog_kts"), 1),
            _fmt_num(nr.get("cumulative_distance_nm"), 1),
            _fmt_num(nr.get("cumulative_fuel_mt"), 2),
            _fmt_num(nr.get("wind_speed_kts"), 1),
            _fmt_num(nr.get("wave_height_m"), 1),
        ]

        for i, v in enumerate(values):
            pdf.cell(widths[i], 5, v, border=1, align="C")
        pdf.ln()


def _add_weather_summary(pdf: VoyagePDF, arrival_report: dict):
    ws = arrival_report.get("weather_summary")
    if not ws:
        return

    pdf.add_page()
    pdf.section_title("Weather Summary")

    for param_key, label in [
        ("wind_speed_kts", "Wind Speed (kts)"),
        ("wave_height_m", "Wave Height (m)"),
        ("current_speed_ms", "Current Speed (m/s)"),
    ]:
        data = ws.get(param_key)
        if data:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, _safe(label), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.kv_row("  Min", str(data.get("min", "N/A")))
            pdf.kv_row("  Max", str(data.get("max", "N/A")))
            pdf.kv_row("  Avg", str(data.get("avg", "N/A")))
            pdf.ln(2)
