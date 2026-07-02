"""
Generate PDF report using fpdf2 (no system deps needed).
Usage: python scripts/generate_pdf.py
"""
from fpdf import FPDF
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md_path = os.path.join(BASE, 'docs', 'report.md')
pdf_path = os.path.join(BASE, 'docs', 'AQI_PROJECT_REPORT.pdf')

with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=20)

pdf.add_font('DejaVu', '', os.path.join(BASE, 'docs', 'DejaVuSans.ttf'), uni=True)
pdf.add_font('DejaVu', 'B', os.path.join(BASE, 'docs', 'DejaVuSans-Bold.ttf'), uni=True)
pdf.add_font('DejaVu', 'I', os.path.join(BASE, 'docs', 'DejaVuSans-Oblique.ttf'), uni=True)
# Fallback: use built-in Courier if DejaVu not available
