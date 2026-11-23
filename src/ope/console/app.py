"""CVM console (Plotly Dash). Talks to the API only.

    OPE_API_URL=http://localhost:8000 OPE_API_KEY=local-dev-key python -m ope.console.app
"""

from __future__ import annotations

import os
from urllib.parse import parse_qs

import httpx
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

from ope import config
from ope.console import theme

API = os.environ.get("OPE_API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ.get("OPE_API_KEY", "local-dev-key")}
ACTION_LABEL = {"retain": "Retain", "crosssell": "Cross-sell", "upsell": "Up-sell", "none": "No contact"}
CHANNEL_LABEL = {"call_centre": "Call centre", "sms": "SMS", "app_push": "App push"}


def api(method: str, path: str, **kw):
    r = httpx.request(method, f"{API}{path}", headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def money(x: float) -> str:
    return f"{x:,.0f} EGP"


def kpi(label: str, value: str) -> html.Div:
    return html.Div([html.Div(value, className="value"), html.Div(label, className="label")], className="card kpi")


biz = config.load("business")
app = Dash(__name__, title="OPE CVM console")
app.index_string = app.index_string.replace("</head>", f"<style>{theme.CSS}</style></head>")


def slider(id_, label, lo, hi, step, value, fmt="{:,.0f}"):
    marks = {lo: fmt.format(lo), hi: fmt.format(hi)}
    return html.Div([
        html.Label(label),
        dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=marks, tooltip={"always_visible": False, "placement": "bottom"}),
    ], style={"marginBottom": "10px"})


simulator = html.Div([
    html.Div([
        html.Div([
            html.H2("Campaign settings"),
            slider("budget", "Weekly budget (EGP)", 0, 200_000, 5_000, biz["budget"]),
            slider("cost-retain", "Retention offer cost per contact", 0, 60, 1, biz["actions"]["retain"]["cost"]),
            slider("cost-crosssell", "Cross-sell SMS cost per contact", 0, 20, 0.5, biz["actions"]["crosssell"]["cost"], "{:,.1f}"),
            slider("cost-upsell", "Up-sell push cost per contact", 0, 20, 0.5, biz["actions"]["upsell"]["cost"], "{:,.1f}"),
            slider("cap-call", "Call-centre capacity (contacts)", 0, 6000, 100, biz["channel_capacity"]["call_centre"]),
            html.P("All money values and response rates are illustrative assumptions (configs/business.yaml).", className="note"),
        ], className="card"),
        html.Div([
            html.Div(id="sim-kpis", className="grid"),
            dcc.Graph(id="profit-curve", config={"displayModeBar": False}),
        ]),
    ], style={"display": "grid", "gridTemplateColumns": "minmax(260px, 360px) 1fr", "gap": "16px"}, className="stack-mobile"),
    html.Div([
        dcc.Graph(id="per-action", config={"displayModeBar": False}, style={"flex": 1, "minWidth": "280px"}),
        dcc.Graph(id="per-channel", config={"displayModeBar": False}, style={"flex": 1, "minWidth": "280px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginTop": "16px"}),
])

segments = html.Div([
    html.Div([
        html.Label("Group subscribers by"),
        dcc.Dropdown(id="group-by", value="risk_band", clearable=False, options=[
            {"label": "Churn risk band", "value": "risk_band"},
            {"label": "Planned action", "value": "action"},
            {"label": "Var218 (top categorical)", "value": "Var218"},
            {"label": "Var226", "value": "Var226"},
        ], style={"maxWidth": "320px"}),
    ], className="card"),
    dcc.Graph(id="segment-chart", config={"displayModeBar": False}),
    html.Div(id="segment-table", className="card"),
    html.Div(id="lift-table", className="card", style={"marginTop": "16px"}),
])

card = html.Div([
    html.Div([
        html.Label("Subscriber id"),
        html.Div([dcc.Input(id="sid", type="text", placeholder="16-character id", style={"width": "260px"}), html.Button("Look up", id="lookup")], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ], className="card"),
    html.Div(id="card-body", style={"marginTop": "16px"}),
])

health = html.Div(id="health-body")

app.layout = html.Div([
    html.Div([
        html.H1("Propensity & Next Best Action"),
        html.Div("Weekly retention, cross-sell and up-sell plan. Data: Orange KDD Cup 2009 (public research data; not affiliated with Orange).", className="sub"),
    ], style={"marginBottom": "18px"}),
    dcc.Tabs([
        dcc.Tab(simulator, label="Campaign simulator", className="tab", selected_className="tab", value="simulator"),
        dcc.Tab(segments, label="Segments", className="tab", selected_className="tab", value="segments"),
        dcc.Tab(card, label="Subscriber card", className="tab", selected_className="tab", value="card"),
        dcc.Tab(health, label="Model health", className="tab", selected_className="tab", value="health"),
    ], id="tabs", value="simulator"),
    dcc.Location(id="url"),
], className="wrap")


def _query(search: str | None) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}


@app.callback(Output("tabs", "value"), Output("sid", "value"), Input("url", "search"))
def deep_link(search):
    """Shareable links: ?tab=segments, ?tab=card&sid=<subscriber id>, ?tab=health."""
    q = _query(search)
    tab = q.get("tab", "simulator")
    return (tab if tab in ("simulator", "segments", "card", "health") else "simulator"), q.get("sid")


def _sim_body(budget, c_ret, c_cross, c_up, cap_call):
    caps = dict(biz["channel_capacity"], call_centre=int(cap_call))
    return {
        "budget": float(budget),
        "channel_capacity": caps,
        "actions": {"retain": {"cost": c_ret}, "crosssell": {"cost": c_cross}, "upsell": {"cost": c_up}},
    }


@app.callback(
    Output("sim-kpis", "children"), Output("profit-curve", "figure"), Output("per-action", "figure"), Output("per-channel", "figure"),
    Input("budget", "value"), Input("cost-retain", "value"), Input("cost-crosssell", "value"), Input("cost-upsell", "value"), Input("cap-call", "value"),
)
def run_simulation(budget, c_ret, c_cross, c_up, cap_call):
    body = _sim_body(budget, c_ret, c_cross, c_up, cap_call)
    s = api("POST", "/v1/campaigns/simulate", json=body)
    curve = api("POST", "/v1/campaigns/profit-curve", json=body, params={"points": 16})["points"]
    kpis = [
        kpi("Expected value (assumption-based)", money(s["expected_value"])),
        kpi("Spend", money(s["spend"])),
        kpi("Contacts", f"{s['contacts']:,}"),
        kpi("Value per EGP spent", f"{s['expected_value'] / s['spend']:.2f}" if s["spend"] else "-"),
    ]
    fig = go.Figure(go.Scatter(x=[p["budget"] for p in curve], y=[p["expected_value"] for p in curve], mode="lines+markers", line=dict(color=theme.ACCENT, width=2), name="expected value"))
    fig.add_vline(x=budget, line=dict(color=theme.MUTED, dash="dot"))
    fig.update_layout(**theme.figure_layout(title="Expected value against budget", xaxis_title="budget (EGP)", yaxis_title="expected value (EGP)", height=330))
    acts = s["per_action"]
    names = [a for a in ("retain", "crosssell", "upsell") if a in acts]
    f2 = go.Figure(go.Bar(x=[ACTION_LABEL[a] for a in names], y=[acts[a]["n"] for a in names], marker_color=theme.SERIES[: len(names)], text=[f"{acts[a]['n']:,}" for a in names], textposition="outside"))
    f2.update_layout(**theme.figure_layout(title="Contacts per action", height=300))
    ch = s["per_channel"]
    caps = body["channel_capacity"]
    f3 = go.Figure([
        go.Bar(name="planned", x=[CHANNEL_LABEL[c] for c in caps], y=[ch.get(c, 0) for c in caps], marker_color=theme.ACCENT),
        go.Bar(name="capacity", x=[CHANNEL_LABEL[c] for c in caps], y=list(caps.values()), marker_color="#c9c5bb"),
    ])
    f3.update_layout(**theme.figure_layout(title="Channel load", barmode="group", height=300))
    return kpis, fig, f2, f3


@app.callback(Output("segment-chart", "figure"), Output("segment-table", "children"), Output("lift-table", "children"), Input("group-by", "value"))
def show_segments(group_by):
    d = api("GET", "/v1/segments/summary", params={"group_by": group_by})
    segs = d["segments"]
    fig = go.Figure()
    for col, name, color in (("mean_p_churn", "churn", theme.SERIES[0]), ("mean_p_appetency", "appetency", theme.SERIES[1]), ("mean_p_upselling", "up-sell", theme.SERIES[2])):
        fig.add_bar(name=name, x=[s["segment"] for s in segs], y=[s[col] for s in segs], marker_color=color)
    fig.update_layout(**theme.figure_layout(title="Mean calibrated probability per segment", barmode="group", yaxis_tickformat=".0%", height=360))
    head = html.Tr([html.Th(h) for h in ("Segment", "Subscribers", "Churn", "Appetency", "Up-sell")])
    rows = [html.Tr([html.Td(s["segment"]), html.Td(f"{s['subscribers']:,}"), html.Td(f"{s['mean_p_churn']:.1%}"), html.Td(f"{s['mean_p_appetency']:.1%}"), html.Td(f"{s['mean_p_upselling']:.1%}")]) for s in segs]
    lift = [html.H3("Test-split lift (top decile)")]
    try:
        import json

        ev = json.loads((config.path("reports") / "evaluation.json").read_text())
        lift.append(html.Table([html.Tr([html.Th(h) for h in ("Target", "Lift d1", "Capture top 30%", "Precision top 5%")])] + [
            html.Tr([html.Td(t), html.Td(f"{e['lift']['lift_d1']:.2f}"), html.Td(f"{e['lift']['capture_d3']:.1%}"), html.Td(f"{e['precision_top5']:.1%}")]) for t, e in ev.items()
        ]))
    except FileNotFoundError:
        lift.append(html.P("Evaluation report not generated yet.", className="note"))
    return fig, html.Table([head, *rows]), lift


@app.callback(Output("card-body", "children"), Input("lookup", "n_clicks"), Input("sid", "n_submit"), Input("url", "search"), State("sid", "value"))
def subscriber_card(_, __, search, sid):
    if callback_context.triggered_id == "url":
        sid = _query(search).get("sid")
        if not sid:
            return []
    if not sid:
        return html.P("Enter a subscriber id.", className="note")
    try:
        sc = api("GET", f"/v1/subscribers/{sid.strip()}/scores")
        nba = api("GET", f"/v1/subscribers/{sid.strip()}/next-best-action")
    except httpx.HTTPStatusError as e:
        return html.P(f"Not found ({e.response.status_code}).", className="note")
    p = sc["scores"]
    reasons = nba["reasons"]
    return html.Div([
        html.Div([
            kpi("Churn probability", f"{p['p_churn']:.1%}"),
            kpi("Appetency probability", f"{p['p_appetency']:.1%}"),
            kpi("Up-sell probability", f"{p['p_upselling']:.1%}"),
            kpi(f"Next best action ({CHANNEL_LABEL.get(nba['channel'], 'none')})", ACTION_LABEL[nba["action"]]),
        ], className="grid"),
        html.Div([
            html.H3("Why (top reasons, SHAP)"),
            html.Table([html.Tr([html.Th(h) for h in ("Feature", "Signal", "Value", "Impact (log-odds)")])] + [
                html.Tr([html.Td(r["feature"]), html.Td(r["signal"]), html.Td("missing" if r["value"] is None else str(r["value"])), html.Td(f"+{r['shap']:.3f}")]) for r in reasons
            ]) if reasons else html.P("Reasons are computed only for subscribers selected for an action.", className="note"),
            html.P(f"Expected value {money(nba['expected_value'])}, cost {money(nba['cost'])}. Snapshot {sc['snapshot_date']}. Models: {', '.join(sc['model_version'].values())}.", className="note"),
        ], className="card", style={"marginTop": "16px"}),
    ])


@app.callback(Output("health-body", "children"), Input("tabs", "value"))
def model_health(tab):
    if tab != "health":
        return []
    h = api("GET", "/v1/model/health")
    rows = [html.Tr([html.Th(x) for x in ("Target", "Model", "CV AUC", "Test AUC (95% CI)", "Test PR-AUC", "ECE (calibrated)", "Calibration", "Gate")])]
    for t, m in h.items():
        ci = m.get("test_auc_ci95") or [None, None]
        rows.append(html.Tr([
            html.Td(t), html.Td(m["candidate"] or "-"), html.Td(f"{m['cv_auc']:.4f}" if m["cv_auc"] else "-"),
            html.Td(f"{m['test_auc']:.4f} ({ci[0]:.3f}-{ci[1]:.3f})" if m["test_auc"] else "-"),
            html.Td(f"{m['test_pr_auc']:.4f}" if m["test_pr_auc"] else "-"),
            html.Td(f"{m['test_ece_calibrated']:.4f}" if m["test_ece_calibrated"] is not None else "-"),
            html.Td(m["calibration_method"] or "-"), html.Td("passed" if m["gate_passed"] else "-"),
        ]))
    parts = [html.Div([html.H2("Champion models"), html.Table(rows)], className="card")]
    try:
        d = api("GET", "/v1/monitoring/drift")
        est = [html.Tr([html.Th(x) for x in ("Target", "Reference AUC", "Estimated AUC (CBPE)", "Alert chunks")])] + [
            html.Tr([html.Td(t), html.Td(f"{c['reference_auc']:.4f}"), html.Td(f"{c['estimated_auc']:.4f}"), html.Td(str(c["alert_chunks"]))]) for t, c in d["cbpe"].items()
        ]
        psi = go.Figure(go.Bar(x=list(d["top_psi"].values()), y=list(d["top_psi"].keys()), orientation="h", marker_color=[theme.SERIES[1] if v > 0.2 else theme.ACCENT for v in d["top_psi"].values()]))
        psi.add_vline(x=0.2, line=dict(color=theme.MUTED, dash="dot"))
        psi.update_layout(**theme.figure_layout(title="PSI, top features and scores (0.2 = retrain line)", height=360, yaxis=dict(autorange="reversed")))
        label = d["label"] + (" (SYNTHETIC drift demo)" if d.get("synthetic") else "")
        decision = "Retrain requested: " + "; ".join(d["decision"]["reasons"]) if d["decision"]["retrain"] else "No retrain trigger."
        parts.append(html.Div([html.H2(f"Monitoring: {label}"), html.P(decision, className="note"), html.Table(est), dcc.Graph(figure=psi, config={"displayModeBar": False})], className="card", style={"marginTop": "16px"}))
    except httpx.HTTPStatusError:
        parts.append(html.P("No monitoring run yet.", className="note"))
    return parts


server = app.server

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=False)
