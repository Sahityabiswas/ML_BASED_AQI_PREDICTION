import markdown
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md_path = os.path.join(BASE, 'docs', 'report.md')
html_path = os.path.join(BASE, 'docs', 'AQI_PROJECT_REPORT.html')

with open(md_path, 'r', encoding='utf-8') as f:
    md_text = f.read()

html = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite'])

full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AQI Project Report</title>
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 2em; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 0.3em; }}
  h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; margin-top: 1.5em; }}
  h3 {{ color: #34495e; margin-top: 1.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #2980b9; color: white; }}
  tr:nth-child(even) {{ background: #f2f6f9; }}
  code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #f8f8f8; border: 1px solid #ddd; padding: 1em; border-radius: 5px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 2em 0; }}
  .header {{ text-align: center; margin-bottom: 2em; }}
  .header h1 {{ border-bottom: none; }}
</style>
</head>
<body>
<div class="header">
  <h1>AQI Prediction — Full ML Pipeline Report</h1>
  <p>ML Engineering Project | July 2026</p>
</div>
{html}
</body>
</html>'''

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f'Report HTML saved: {html_path}')
