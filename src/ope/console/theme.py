"""Console look: GFS Didot for headings and display numbers, Times New Roman Bold for labels
and data. Calm palette, no glow. Text colours meet WCAG AA on the page background."""

INK = "#1f2328"  # body text, 15.8:1 on PAPER
MUTED = "#4a4f57"  # secondary text, 8.2:1 on PAPER
PAPER = "#fbfaf7"
CARD = "#ffffff"
RULE = "#d9d6cf"
ACCENT = "#2f5d50"  # deep green, 7.1:1 on PAPER
SERIES = ["#2f5d50", "#9a5b2e", "#4b5d8a", "#7a7a7a"]  # retain, cross-sell, up-sell, none

DISPLAY = "'GFS Didot', Didot, 'Bodoni 72', Georgia, serif"
DATA = "'Times New Roman', Times, serif"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');
body {{ background:{PAPER}; color:{INK}; font-family:{DATA}; font-weight:700; margin:0; }}
h1, h2, h3, .display {{ font-family:{DISPLAY}; font-weight:400; color:{INK}; letter-spacing:0; }}
h1 {{ font-size:34px; margin:0; }}
h2 {{ font-size:24px; margin:8px 0 12px; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:24px 16px 48px; }}
.sub {{ color:{MUTED}; font-size:15px; margin-top:4px; }}
.card {{ background:{CARD}; border:1px solid {RULE}; border-radius:6px; padding:16px 18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.kpi .value {{ font-family:{DISPLAY}; font-size:34px; font-weight:400; }}
.kpi .label {{ color:{MUTED}; font-size:14px; }}
.note {{ color:{MUTED}; font-size:13px; }}
label, .label {{ font-family:{DATA}; font-weight:700; color:{INK}; font-size:14px; }}
table {{ border-collapse:collapse; width:100%; font-family:{DATA}; font-weight:700; font-size:14px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid {RULE}; }}
th {{ color:{MUTED}; }}
.tab {{ font-family:{DATA} !important; font-weight:700 !important; }}
input {{ font-family:{DATA}; font-weight:700; font-size:15px; padding:6px 8px; border:1px solid {RULE}; border-radius:4px; }}
button {{ font-family:{DATA}; font-weight:700; background:{ACCENT}; color:#fff; border:0; border-radius:4px; padding:8px 14px; cursor:pointer; }}
@media (max-width:640px) {{ h1 {{ font-size:26px; }} .kpi .value {{ font-size:26px; }} }}
"""


def figure_layout(**kw):
    base = dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family=DATA, color=INK, size=13),
        margin=dict(l=56, r=16, t=36, b=48),
        xaxis=dict(gridcolor="#eceae4", zeroline=False, linecolor=RULE),
        yaxis=dict(gridcolor="#eceae4", zeroline=False, linecolor=RULE),
        legend=dict(orientation="h", y=-0.2),
        title_font=dict(family=DISPLAY, size=18),
    )
    base.update(kw)
    return base
