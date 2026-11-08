"""Dashboard del Proyecto Integrador de Deep Learning: predicción de clics con MLP.

Lee únicamente los resultados ya calculados en ../results (no entrena nada) y los presenta en
seis secciones: Contexto, Datos, Modelos, Escalabilidad, Interpretabilidad y Conclusiones.

Ejecutar en local:   python dashboard/app.py          -> http://127.0.0.1:8050
Producción (Render): gunicorn --chdir dashboard app:server
"""
from __future__ import annotations

import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html
from flask import abort, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = RES / "figures"
BOOK = "https://aduartech81.github.io/deep-learning-01-ctr-mlp/"
REPO = "https://github.com/aduartech81/deep-learning-01-ctr-mlp"

# ----------------------------------------------------------------------------- estilo
# Paleta categórica validada (la misma de los notebooks); el color sigue a la entidad.
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7", "#ffffff"
FW_COLOR = {"scikit-learn": S[0], "PySpark": S[1]}
MODEL_COLOR = {"MLP": S[0], "Lineal (SGD/LR)": S[1], "Árboles (HGB/GBT)": S[2]}
ENC_DASH = {"hash": "solid", "target": "dash", "freq": "dot"}
ENC_NAME = {"hash": "hashing", "target": "target encoding", "freq": "frecuencia"}
MODEL_NAME = {"mlp": "MLP", "sgd": "Lineal (SGD/LR)", "lr": "Lineal (SGD/LR)",
              "hgb": "Árboles (HGB/GBT)", "gbt": "Árboles (HGB/GBT)"}
FW_NAME = {"sklearn": "scikit-learn", "spark": "PySpark"}


def base_layout(fig: go.Figure, title: str | None = None, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, x=0, font=dict(size=15, color=INK)) if title else None,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK2),
        margin=dict(l=10, r=10, t=48 if title else 16, b=10), paper_bgcolor=SURF, plot_bgcolor=SURF,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)), separators=",.", **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=BASE, zeroline=False, ticks="outside", tickcolor=BASE)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASE, zeroline=False)
    return fig


N_TICKS = dict(tickvals=[1e5, 3e5, 1e6, 3e6, 1e7, 3e7], ticktext=["100 mil", "300 mil", "1 M", "3 M", "10 M", "30 M"])


def fmt(x: float, d: int = 3) -> str:
    """Número con coma decimal (es-CO)."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def graph(fig: go.Figure) -> dcc.Graph:
    return dcc.Graph(figure=fig, style={"height": f"{fig.layout.height or 380}px"}, config={"displayModeBar": False})


# ----------------------------------------------------------------------------- datos
def read_jsonl(p: Path) -> pd.DataFrame:
    return pd.read_json(p, lines=True) if p.exists() else pd.DataFrame()


final = pd.read_csv(RES / "tabla_final.csv")
final["entorno"] = final.framework.map(FW_NAME)
final["familia"] = final.model.map(MODEL_NAME)
final["codificación"] = final.encoding.map(ENC_NAME)
final["modelo"] = final.model.str.upper()
tests = pd.read_csv(RES / "pruebas_hipotesis.csv")
boot = pd.read_csv(RES / "bootstrap_ic.csv")
bench = read_jsonl(RES / "bench.jsonl")
if "fallo" in bench:
    bench = bench[bench["fallo"].isna()]
amd = pd.read_csv(RES / "amdahl.csv")
etapas = pd.read_csv(RES / "bench_etapas.csv")
prep = pd.read_csv(RES / "prep_codificaciones.csv")
cardin = pd.read_csv(RES / "eda_cardinalidad.csv")
psi = pd.read_csv(RES / "eda_psi.csv")
asoc = pd.read_csv(RES / "eda_asociacion.csv")
coh = pd.read_csv(RES / "coherencia_global.csv")
lime = pd.read_csv(RES / "lime_casos.csv")
lvs = pd.read_csv(RES / "lime_vs_shap.csv")
abl = pd.read_csv(RES / "tabla_ablaciones_sklearn.csv")
manif = json.loads((RES / "manifiesto_datos.json").read_text())
ficha = dict(pd.read_csv(RES / "ficha_datos.csv").values)

N_TOTAL = sum(manif["filas_por_dia"].values())
CTR = float(str(ficha.get("Tasa base de clics (global)", "0.17")).replace(",", "."))
best = final.sort_values("roc_auc_media", ascending=False).iloc[0]


def pick(fw, model, enc, n):
    r = final[(final.framework == fw) & (final.model == model) & (final.encoding == enc) & (final.n == n)]
    return r.iloc[0] if len(r) else None


hgb10 = pick("sklearn", "hgb", "target", 10_000_000)
t_mlp_hgb = tests[(tests.A == "sk MLP hash") & (tests.B == "sk HGB target")].iloc[0]
t_same = tests[tests.A.str.contains("mismo n")].iloc[0]
amd_spark = amd[amd.framework == "spark"].iloc[0]

# ----------------------------------------------------------------------------- componentes
def kpi(value: str, label: str, accent: bool = False):
    return html.Div([html.Div(value, className="kpi-value" + (" accent" if accent else "")),
                     html.Div(label, className="kpi-label")], className="kpi")


def section_head(eyebrow: str, title: str, lead: str):
    return html.Div([html.Div(eyebrow, className="eyebrow"), html.H1(title), html.P(lead, className="lead")],
                    className="section-head")


def card(title: str, body, note: str | None = None):
    return html.Div([html.H3(title), body] + ([html.P(note, className="note")] if note else []), className="viz-card")


def figure_img(name: str, caption: str):
    return html.Figure([html.Img(src=f"/figs/{name}", alt=caption),
                        html.Figcaption(caption)], className="static-fig")


def table(df: pd.DataFrame, page_size: int = 12):
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns], page_size=page_size,
        sort_action="native", style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": 600, "backgroundColor": "#f6f6f3", "color": INK, "border": "none",
                      "fontFamily": "Inter, system-ui, sans-serif", "fontSize": 12.5},
        style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": 13, "padding": "7px 10px",
                    "color": INK2, "textAlign": "left", "borderBottom": f"1px solid {GRID}",
                    "whiteSpace": "normal", "height": "auto", "maxWidth": 340})


# ----------------------------------------------------------------------------- páginas
def page_contexto():
    days = pd.DataFrame({"día": [int(d) for d in manif["filas_por_dia"]], "filas": list(manif["filas_por_dia"].values())})
    days["partición"] = np.select([days.día <= 28, days.día == 29], ["Entrenamiento (21–28)", "Validación (29)"], "Prueba (30)")
    fig = go.Figure()
    for part, col in zip(["Entrenamiento (21–28)", "Validación (29)", "Prueba (30)"], [S[0], S[3], S[1]]):
        d = days[days["partición"] == part]
        fig.add_bar(x=[f"{x} oct" for x in d.día], y=d.filas / 1e6, name=part, marker_color=col,
                    hovertemplate="%{x}: %{y:.2f} M impresiones<extra>" + part + "</extra>")
    base_layout(fig, height=300, bargap=0.25)
    fig.update_yaxes(title="Millones de impresiones")
    fig.update_traces(marker_line_width=0)

    qs = [
        ("P1 · Escala", "¿Desde qué volumen PySpark supera a scikit-learn en tiempo sin perder calidad?",
         "En una sola máquina no hubo cruce: scikit-learn fue más rápido hasta 10 M de filas."),
        ("P2 · Codificación", "¿Cómo afecta la codificación de alta cardinalidad al desempeño y al costo?",
         "Hashing supera a frecuencia por +0,070 de AUC, con el doble de memoria de entrada."),
        ("P3 · ¿Vale el MLP?", "¿Mejora el MLP de forma significativa a modelos más simples?",
         "Sí, pero poco: +0,003 de AUC sobre HistGradientBoosting, ~8 veces más lento."),
        ("P4 · LIME", "¿Son las explicaciones de LIME estables y fieles?",
         "Estables en parte, poco fieles (R² ≈ 0,12): no bastan solas como evidencia."),
    ]
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Andrés Duarte Chaves · Deep Learning · Maestría en Ingeniería Biomédica, Uninorte", className="eyebrow"),
                html.H1("¿Hará clic? Predecir el clic en un anuncio móvil con redes neuronales", className="hero-title"),
                html.P(["Con ", html.B(f"{N_TOTAL/1e6:,.1f} millones".replace(".", ",")),
                        " de impresiones reales de Avazu (10 días de octubre de 2014) se entrena un perceptrón multicapa (MLP) "
                        "en scikit-learn y en PySpark, se compara contra modelos más simples y se pregunta cuánto se puede "
                        "confiar en sus explicaciones con LIME."], className="lead"),
                html.Div([html.Span(t, className="chip") for t in ["Clasificación binaria", "Datos tabulares masivos",
                                                                   "scikit-learn vs PySpark", "Interpretabilidad"]], className="chips"),
                html.Div([dbc.Button("Ver el resultado principal →", href="/modelos", className="btn-primary-soft"),
                          dbc.Button("Jupyter Book completo", href=BOOK, target="_blank", className="btn-ghost")], className="hero-actions"),
            ], className="hero-text"),
            html.Div(html.Img(src="/assets/hero.svg", alt="Ilustración: teléfono con un anuncio y una red neuronal"), className="hero-art"),
        ], className="hero"),
        html.Div([
            kpi(f"{N_TOTAL/1e6:,.1f} M".replace(".", ","), "impresiones (filas) en el conjunto completo"),
            kpi(f"{CTR*100:,.1f} %".replace(".", ","), "tasa base de clics: clase positiva minoritaria"),
            kpi(fmt(best.roc_auc_media), f"ROC AUC del mejor modelo en prueba (MLP + hashing, 10 M filas)", accent=True),
            kpi(f"{len(final)}", "configuraciones evaluadas (entorno × modelo × codificación × tamaño)"),
        ], className="kpi-row"),
        html.Div([
            card("Una partición temporal, sin fuga de información", graph(fig),
                 "Se entrena con el pasado (días 21–28), se ajusta con el día 29 y se mide una sola vez en el día 30. "
                 "Vocabularios, escaladores y codificadores se ajustan solo con entrenamiento."),
        ]),
        html.H2("Cuatro preguntas, cuatro respuestas", className="sub"),
        html.Div([html.Div([html.Div(t, className="q-tag"), html.P(q, className="q-text"), html.P(a, className="q-answer")],
                           className="q-card") for t, q, a in qs], className="q-grid"),
    ])


def page_datos():
    cc = card_cardinalidad()
    a = asoc.sort_values("V_cramer_vs_click").tail(15)
    fa = go.Figure(go.Bar(x=a.V_cramer_vs_click, y=a.variable, orientation="h", marker_color=S[0],
                          customdata=a.info_mutua, hovertemplate="%{y}<br>V de Cramér %{x:.3f}<br>Información mutua %{customdata:.4f}<extra></extra>"))
    base_layout(fa, height=420); fa.update_xaxes(title="V de Cramér con el clic")
    p = psi.sort_values("PSI").tail(12)
    fp = go.Figure(go.Bar(x=p.PSI, y=p.variable, orientation="h",
                          marker_color=[S[7] if v > 0.25 else (S[3] if v > 0.1 else BASE) for v in p.PSI],
                          customdata=np.c_[p.pct_train_en_RARE, p.pct_prueba_en_RARE],
                          hovertemplate="%{y}: PSI %{x:.2f}<br>categorías raras/nuevas: %{customdata[0]:.1f} % en entrenamiento → %{customdata[1]:.1f} % en prueba<extra></extra>"))
    base_layout(fp, height=420); fp.update_xaxes(title="PSI entrenamiento → prueba (escala log)", type="log", tickvals=[0.1, 0.25, 0.5, 1, 2, 5, 10], ticktext=["0,1", "0,25", "0,5", "1", "2", "5", "10"])
    return html.Div([
        section_head("Datos", "Pocas columnas, millones de categorías y un mercado que cambia de un día a otro",
                     "Las 24 variables son categóricas. Algunas tienen millones de valores distintos, y la distribución del día "
                     "de prueba no es la del entrenamiento: eso decide cómo codificar y cómo evaluar."),
        html.Div([cc, card("Qué variables se asocian con el clic", graph(fa),
                           "Las variables anonimizadas C14 y C17 y la identidad del sitio son las más asociadas al clic.")], className="grid-2"),
        html.Div([card("Deriva entre entrenamiento y prueba", graph(fp),
                       "PSI > 0,25 (rojo) indica un cambio fuerte. dow es trivial (el día 30 es otro día de la semana), pero C14 y C17 "
                       "traen más de un tercio de valores nuevos o raros el día 30: parte de la dificultad es temporal, no del modelo."),
                  card_codificaciones()], className="grid-2"),
        card("El clic según el día y la hora", figure_img("02_click_ctr_dia_hora.png", "Volumen y tasa de clics por día y hora (capítulo 2)")),
    ])


def card_codificaciones():
    d = prep.copy()
    d["etiqueta"] = d.codificacion.str.replace("hash 2^", "hashing 2^", regex=False).replace({"freq": "frecuencia", "target": "target encoding"})
    f = go.Figure(go.Bar(x=d.etiqueta, y=d.dim_entrada, marker_color=[S[0] if c.startswith("hash") else S[1] if c == "freq" else S[2] for c in d.codificacion],
                         customdata=np.c_[d.pct_tokens_colisionados.fillna(0), d.MB_matriz],
                         hovertemplate="%{x}<br>%{y:,} columnas de entrada<br>%{customdata[0]:.1f} % de valores colisionados"
                                       "<br>%{customdata[1]:.0f} MB por 100 mil filas<extra></extra>"))
    base_layout(f, height=420); f.update_yaxes(title="Columnas de entrada al MLP (escala log)", type="log", tickvals=[10, 100, 1000, 10000, 100000], ticktext=["10", "100", "1.000", "10.000", "100.000"])
    for x, y, c in zip(d.etiqueta, d.dim_entrada, d.pct_tokens_colisionados):
        if pd.notna(c):
            f.add_annotation(x=x, y=np.log10(y), text=f"{c:.0f} % colisiones".replace(".", ","), showarrow=False, yshift=14, font=dict(size=11, color=INK2))
    return card("Tres formas de codificar 24 columnas categóricas", graph(f),
                "El hashing reparte las categorías en 2^k columnas dispersas (más columnas = menos colisiones); frecuencia y target "
                "encoding comprimen cada variable a una sola columna numérica.")


def card_cardinalidad():
    cd = cardin.sort_values("categorias").tail(15)
    f = go.Figure(go.Bar(x=cd.categorias, y=cd.variable, orientation="h", marker_color=S[0],
                         customdata=cd.pct_vistas_una_vez,
                         hovertemplate="%{y}: %{x:,} categorías<br>%{customdata:.1f} % vistas una sola vez<extra></extra>"))
    base_layout(f, height=420); f.update_xaxes(title="Categorías distintas (escala log)", type="log")
    return card("Cardinalidad: de 7 a 5,6 millones de valores", graph(f),
                "device_ip y device_id hacen inviable el one-hot: por eso se comparan hashing, frecuencia y target encoding.")


def page_modelos():
    metrics = [("roc_auc_media", "ROC AUC (más alto es mejor)"), ("log_loss_media", "Log loss (más bajo es mejor)"),
               ("ece_media", "Error de calibración ECE (más bajo es mejor)"), ("train_s_media", "Tiempo de entrenamiento (s)")]
    controls = html.Div([
        html.Div([html.Label("Métrica"), dcc.Dropdown(id="m-metric", options=[{"label": l, "value": v} for v, l in metrics],
                                                      value="roc_auc_media", clearable=False)], className="ctl"),
        html.Div([html.Label("Entorno"), dcc.Checklist(id="m-fw", options=["scikit-learn", "PySpark"], value=["scikit-learn", "PySpark"],
                                                       inline=True, className="checks")], className="ctl"),
        html.Div([html.Label("Codificación"), dcc.Checklist(id="m-enc", options=[{"label": ENC_NAME[k], "value": k} for k in ["hash", "target", "freq"]],
                                                            value=["hash", "target", "freq"], inline=True, className="checks")], className="ctl"),
    ], className="controls")

    # bosque de IC por bootstrap (ROC AUC, prueba)
    b = boot[boot.metrica == "roc_auc"].sort_values("puntual")
    fb = go.Figure()
    fb.add_trace(go.Scatter(x=b.puntual, y=b.modelo, mode="markers", marker=dict(size=10, color=[S[1] if m.startswith("spark") else S[0] for m in b.modelo]),
                            error_x=dict(type="data", symmetric=False, array=b.ic_sup - b.puntual, arrayminus=b.puntual - b.ic_inf, color=INK2, thickness=1.5, width=5),
                            hovertemplate="%{y}<br>AUC %{x:.4f}<extra>IC 95 % por bootstrap</extra>", showlegend=False))
    base_layout(fb, height=60 + 34 * len(b)); fb.update_xaxes(title="ROC AUC en prueba (IC 95 %, bootstrap)")

    # pruebas pareadas
    t = tests.copy()
    t["ΔAUC (A − B)"] = t.dif_AUC.map(lambda v: f"{v:+.4f}".replace(".", ","))
    t["p (DeLong, Holm)"] = t.p_DeLong_Holm.map(lambda p: "< 1e-15" if p < 1e-15 else f"{p:.2g}")
    t["Lectura"] = [("significativa" if p < 0.05 else "no significativa") + (" pero pequeña (< 0,005)" if p < 0.05 and abs(d) < 0.005 else "")
                    for p, d in zip(t.p_DeLong_Holm, t.dif_AUC)]
    t = t[["A", "B", "ΔAUC (A − B)", "p (DeLong, Holm)", "Lectura"]]

    # calidad vs costo
    fc = go.Figure()
    for fam, col in MODEL_COLOR.items():
        d = final[final.familia == fam]
        fc.add_trace(go.Scatter(x=d.train_s_media, y=d.roc_auc_media, mode="markers", name=fam,
                                marker=dict(size=np.clip(np.log10(d.n) * 3.2 - 8, 8, 18), color=col, line=dict(color="white", width=2),
                                            symbol=["circle" if f == "sklearn" else "diamond" for f in d.framework]),
                                customdata=np.c_[d.entorno, d["codificación"], d.n],
                                hovertemplate="%{customdata[0]} · " + fam + " · %{customdata[1]}<br>%{customdata[2]:,} filas<br>"
                                              "AUC %{y:.4f} · %{x:,.0f} s<extra></extra>"))
    base_layout(fc, height=420); fc.update_xaxes(title="Tiempo de entrenamiento (s, escala log)", type="log", tickvals=[3, 10, 30, 100, 300, 1000], ticktext=["3", "10", "30", "100", "300", "1.000"]); fc.update_yaxes(title="ROC AUC en prueba")
    fc.add_annotation(text="● scikit-learn   ◆ PySpark · tamaño = filas", xref="paper", yref="paper", x=1, y=0, xanchor="right", yanchor="bottom",
                      showarrow=False, font=dict(size=11, color=MUTED))

    gain = best.roc_auc_media - hgb10.roc_auc_media
    return html.Div([
        section_head("Modelos", "El MLP gana, pero por un margen pequeño frente a los árboles",
                     f"Con 10 millones de filas y hashing, el MLP alcanza un AUC de {fmt(best.roc_auc_media)}. HistGradientBoosting con target "
                     f"encoding queda a {fmt(gain, 3)} y entrena unas {best.train_s_media / hgb10.train_s_media:,.0f} veces más rápido. "
                     "Con las mismas filas, scikit-learn y PySpark obtienen la misma calidad."),
        html.Div([
            kpi(fmt(best.roc_auc_media), "AUC del MLP (scikit-learn, hashing, 10 M)", accent=True),
            kpi(fmt(hgb10.roc_auc_media), "AUC de HistGradientBoosting (target encoding, 10 M)"),
            kpi(f"{t_mlp_hgb.dif_AUC:+.3f}".replace(".", ","), "diferencia MLP − HGB: significativa (DeLong + Holm) pero pequeña"),
            kpi(f"{t_same.dif_AUC:+.4f}".replace(".", ","), "scikit-learn − PySpark con las mismas 100 mil filas: el motor no cambia la calidad"),
        ], className="kpi-row"),
        card("Cómo cambia cada modelo al aumentar los datos", html.Div([controls, dcc.Graph(id="m-curve", style={"height": "440px"}, config={"displayModeBar": False})]),
             "Línea continua = hashing, discontinua = target encoding, punteada = frecuencia. Color = familia de modelo. "
             "El MLP de PySpark densifica la entrada: con hashing solo cupo en 100 mil filas."),
        html.Div([card("Calidad frente a costo", graph(fc),
                       "Arriba a la izquierda es mejor. Los árboles dan casi la misma calidad mucho antes; el MLP paga la última milésima con tiempo."),
                  card("Intervalos de confianza en el día de prueba", graph(fb),
                       "Con 4,2 millones de impresiones de prueba los intervalos son muy estrechos: casi todo es «significativo», por eso importa la magnitud.")],
                 className="grid-2"),
        card("Comparaciones pareadas sobre las mismas impresiones", table(t, 12),
             "Prueba de DeLong para AUC correlacionadas, con corrección de Holm por comparaciones múltiples."),
        html.H2("Explorar todas las configuraciones", className="sub"),
        table(final.assign(n=final.n.map(lambda v: f"{v/1e6:g} M" if v >= 1e6 else f"{v/1e3:g} mil"))[["entorno", "modelo", "codificación", "n", "n_semillas", "roc_auc_media", "log_loss_media", "ece_media", "train_s_media"]]
              .rename(columns={"n": "filas", "n_semillas": "semillas", "roc_auc_media": "ROC AUC", "log_loss_media": "log loss",
                               "ece_media": "ECE", "train_s_media": "entrenamiento (s)"}).round(4), 12),
    ])


def page_escala():
    cr = bench[bench.exp == "cruce"].groupby(["framework", "n"])[["train_s", "total_s"]].median().reset_index()
    fx = go.Figure()
    for fw in ["sklearn", "spark"]:
        d = cr[cr.framework == fw]
        for col, dash_, lab in [("total_s", "solid", "extremo a extremo"), ("train_s", "dot", "solo entrenamiento")]:
            fx.add_trace(go.Scatter(x=d.n, y=d[col], mode="lines+markers", name=f"{FW_NAME[fw]} · {lab}",
                                    line=dict(color=FW_COLOR[FW_NAME[fw]], width=2, dash=dash_), marker=dict(size=8),
                                    hovertemplate="%{x:,} filas: %{y:,.1f} s<extra>" + f"{FW_NAME[fw]} · {lab}" + "</extra>"))
    base_layout(fx, height=400); fx.update_xaxes(title="Filas de entrenamiento (escala log)", type="log", **N_TICKS); fx.update_yaxes(title="Segundos (escala log)", type="log", tickvals=[3, 10, 30, 100, 300], ticktext=["3", "10", "30", "100", "300"])

    st = bench[bench.exp == "fuerte"].groupby(["framework", "cores"])["train_s"].median().reset_index()
    fs = go.Figure()
    for fw in ["sklearn", "spark"]:
        d = st[st.framework == fw].sort_values("cores")
        sp = d.train_s.iloc[0] / d.train_s
        fs.add_trace(go.Scatter(x=d.cores, y=sp, mode="lines+markers", name=FW_NAME[fw], line=dict(color=FW_COLOR[FW_NAME[fw]], width=2),
                                marker=dict(size=9), hovertemplate="%{x} núcleos: aceleración %{y:.2f}×<extra>" + FW_NAME[fw] + "</extra>"))
    cores = np.linspace(1, 8, 50)
    s = float(amd_spark.fraccion_secuencial)
    fs.add_trace(go.Scatter(x=cores, y=1 / (s + (1 - s) / cores), mode="lines", name=f"Amdahl, s = {fmt(s, 2)}",
                            line=dict(color=MUTED, width=1.5, dash="dash"), hoverinfo="skip"))
    fs.add_trace(go.Scatter(x=[1, 8], y=[1, 8], mode="lines", name="ideal (lineal)", line=dict(color=BASE, width=1, dash="dot"), hoverinfo="skip"))
    base_layout(fs, height=400); fs.update_xaxes(title="Núcleos", tickvals=[1, 2, 4, 8]); fs.update_yaxes(title="Aceleración del entrenamiento (×)")

    e = etapas[(etapas.model == "mlp")].groupby(["framework", "n"])[["lectura_s", "prep_s", "train_s", "predict_s"]].median().reset_index()
    e["config"] = e.framework.map(FW_NAME) + " · " + e.n.map(lambda v: f"{v/1e6:.1f} M".replace(".0 M", " M").replace(".", ",") if v >= 1e6 else f"{v/1e3:g} mil")
    fe = go.Figure()
    for col, lab, colr in [("lectura_s", "lectura", S[6]), ("prep_s", "preprocesamiento", S[3]), ("train_s", "entrenamiento", S[0]), ("predict_s", "predicción", S[2])]:
        fe.add_bar(y=e.config, x=e[col], name=lab, orientation="h", marker_color=colr,
                   hovertemplate="%{y}<br>" + lab + ": %{x:,.1f} s<extra></extra>")
    base_layout(fe, height=60 + 38 * len(e), barmode="stack"); fe.update_xaxes(title="Segundos (mediana)"); fe.update_traces(marker_line_color="white", marker_line_width=1)

    sk1 = cr[(cr.framework == "sklearn") & (cr.n == 10_000_000)].total_s.iloc[0]
    sp1 = cr[(cr.framework == "spark") & (cr.n == 10_000_000)].total_s.iloc[0]
    return html.Div([
        section_head("Escalabilidad", "En una sola máquina, scikit-learn llega primero; Spark es el que escala",
                     "Se midió la misma carga (MLP con frecuencia) de 100 mil a 10 millones de filas, y luego con 1, 2, 4 y 8 núcleos. "
                     "Spark paga un costo fijo (arrancar la JVM, planificar, serializar) que en un solo equipo no alcanza a amortizar."),
        html.Div([
            kpi("sin cruce", "Spark no superó a scikit-learn en ningún volumen medido (hasta 10 M de filas)", accent=True),
            kpi(f"{sp1:,.0f} vs {sk1:,.0f} s", "extremo a extremo con 10 M de filas: PySpark vs scikit-learn"),
            kpi(f"{fmt(float(amd_spark.aceleracion_max_teorica), 1)}×", "aceleración máxima teórica de Spark (ley de Amdahl)"),
            kpi("≈ 1×", "el MLP de scikit-learn no se acelera con más núcleos"),
        ], className="kpi-row"),
        html.Div([card("Tiempo frente a volumen", graph(fx),
                       "Las curvas de entrenamiento se acercan en 10 M, pero Spark sigue pagando la lectura y el preprocesamiento."),
                  card("Escalamiento fuerte (1 M de filas, más núcleos)", graph(fs),
                       "Spark pasa de 124 s a 40 s con 8 núcleos; scikit-learn queda plano (su MLP usa un solo hilo).")], className="grid-2"),
        card("Dónde se va el tiempo", graph(fe), "Mediana por etapa en el benchmarking del MLP."),
    ])


def page_interpretabilidad():
    g = coh.copy()
    for c in ["LIME_|w|_medio", "SHAP_|phi|_medio", "caida_auc_media"]:
        g[c + "_n"] = g[c].clip(lower=0) / g[c].clip(lower=0).max()
    g = g.sort_values("caida_auc_media", ascending=True).tail(12)
    fg = go.Figure()
    for col, lab, colr in [("caida_auc_media_n", "Permutación (caída de AUC)", S[2]), ("SHAP_|phi|_medio_n", "SHAP |φ| medio", S[1]), ("LIME_|w|_medio_n", "LIME |w| medio", S[0])]:
        fg.add_bar(y=g.variable, x=g[col], name=lab, orientation="h", marker_color=colr,
                   hovertemplate="%{y}<br>" + lab + ": %{x:.2f} (relativo al máximo)<extra></extra>")
    base_layout(fg, height=520, barmode="group", bargap=0.25, bargroupgap=0.08); fg.update_xaxes(title="Importancia relativa (1 = máximo de cada método)")

    m = lime.merge(lvs, on=["fila", "tipo"])
    fl = go.Figure()
    for tipo, colr, name in [("FP", S[1], "Falsos positivos"), ("FN", S[0], "Falsos negativos")]:
        d = m[m.tipo == tipo]
        fl.add_trace(go.Scatter(x=d.R2_medio, y=d.jaccard_topk, mode="markers", name=name,
                                marker=dict(size=11, color=colr, line=dict(color="white", width=2)),
                                customdata=np.c_[d.fila, d.jaccard_topk_entre_semillas, d.spearman_abs],
                                hovertemplate="Impresión %{customdata[0]}<br>Fidelidad R² %{x:.3f}<br>Acuerdo con SHAP (Jaccard top-k) %{y:.2f}"
                                              "<br>Estabilidad entre semillas %{customdata[1]:.2f}<extra>" + name + "</extra>"))
    base_layout(fl, height=420); fl.update_xaxes(title="Fidelidad local de LIME (R²)", range=[0, 0.3]); fl.update_yaxes(title="Acuerdo con SHAP (Jaccard del top-k)", range=[0, 1.05])
    fl.add_vrect(x0=0.5, x1=0.3, fillcolor="white", line_width=0)

    rho = coh[["LIME_|w|_medio", "SHAP_|phi|_medio", "caida_auc_media"]].corr(method="spearman")
    return html.Div([
        section_head("Interpretabilidad", "LIME señala variables razonables, pero explica mal al propio modelo",
                     "Se explicaron 20 errores del MLP (10 falsos positivos y 10 falsos negativos) con LIME, y se contrastaron con SHAP y con "
                     "la importancia por permutación. Un buen explicador debe ser fiel (reproducir la predicción) y estable (no cambiar con la semilla)."),
        html.Div([
            kpi(fmt(lime.R2_medio.mean(), 2), "fidelidad media de LIME (R² local): baja", accent=True),
            kpi(fmt(lime.jaccard_topk_entre_semillas.mean(), 2), "estabilidad entre semillas (Jaccard del top-5)"),
            kpi(fmt(lvs.spearman_abs.mean(), 2), "acuerdo con SHAP (Spearman de los pesos)"),
            kpi(fmt(rho.iloc[0, 2], 2), "acuerdo del ranking global de LIME con la permutación"),
        ], className="kpi-row"),
        html.Div([card("¿Qué variables importan, según cada método?", graph(fg),
                       "LIME y SHAP coinciden entre sí (sitio, dominio, modelo de dispositivo), pero la permutación pone app_id primero: "
                       "los métodos locales y el global no cuentan la misma historia."),
                  card("Cada punto es una impresión explicada", graph(fl),
                       "Todas las explicaciones quedan con R² < 0,3. En los falsos negativos LIME y SHAP coinciden menos.")], className="grid-2"),
        card("Una explicación de LIME, caso por caso", figure_img("08_lime_casos.png", "Pesos de LIME para casos seleccionados (capítulo 8)")),
    ])


def page_conclusiones():
    items = [
        ("Lo que se encontró", [
            f"El MLP con hashing y 10 millones de filas es el mejor modelo (AUC {fmt(best.roc_auc_media)}), pero HistGradientBoosting queda a "
            f"{fmt(best.roc_auc_media - hgb10.roc_auc_media)} y entrena unas {best.train_s_media / hgb10.train_s_media:,.0f} veces más rápido.",
            "La codificación pesa más que el modelo: hashing supera a frecuencia por 0,07 de AUC en el mismo MLP.",
            "Con las mismas filas, scikit-learn y PySpark dan la misma calidad: la diferencia está en el costo, no en la predicción.",
            "En un solo equipo no hubo punto de cruce: PySpark se justifica cuando los datos no caben en una máquina o hay un clúster real.",
            "LIME es parcialmente estable y poco fiel; sus explicaciones deben contrastarse con SHAP y con la importancia global.",
        ]),
        ("Límites honestos", [
            "Una sola ventana de prueba (un día) y datos de 2014: la validez externa es limitada.",
            "C1 y C14–C21 están anonimizadas: las explicaciones sobre ellas no se traducen a lenguaje de negocio.",
            "Benchmark en una sola máquina (Colab Pro, 8 núcleos, 51 GB) con Spark en modo local.",
            "Los MLP de scikit-learn y Spark no son idénticos (activaciones, optimizador, entrada densa en Spark).",
        ]),
        ("Cómo se hizo", [
            "Partición temporal 21–28 / 29 / 30, ajustes solo con entrenamiento, target encoding out-of-fold.",
            "Cinco semillas con IC 95 %, bootstrap en prueba, DeLong y McNemar con corrección de Holm.",
            "Benchmark de punto de cruce, escalamiento fuerte y débil, ley de Amdahl y memoria pico incluida la JVM.",
            "Todo reproducible: entorno fijado, manifiesto SHA-256 de los datos y notebooks ejecutados publicados.",
        ]),
    ]
    return html.Div([
        section_head("Conclusiones", "Más datos y mejor codificación antes que un modelo más complejo",
                     "La pregunta práctica no fue «¿qué modelo es el mejor?», sino «¿cuánto cuesta cada milésima de AUC y cuánto se puede confiar en la explicación?»."),
        html.Div([html.Div([html.H3(t), html.Ul([html.Li(x) for x in xs])], className="viz-card") for t, xs in items], className="grid-3"),
        html.Div([
            html.H3("Material completo"),
            html.P("El análisis paso a paso, con todo el código y las salidas, está en el Jupyter Book. Este tablero solo lee los resultados ya calculados."),
            html.Div([dbc.Button("Jupyter Book del proyecto", href=BOOK, target="_blank", className="btn-primary-soft"),
                      dbc.Button("Repositorio en GitHub", href=REPO, target="_blank", className="btn-ghost")], className="hero-actions"),
        ], className="viz-card cta"),
    ])


PAGES = {"/": ("Contexto", page_contexto), "/datos": ("Datos", page_datos), "/modelos": ("Modelos", page_modelos),
         "/escalabilidad": ("Escalabilidad", page_escala), "/interpretabilidad": ("Interpretabilidad", page_interpretabilidad),
         "/conclusiones": ("Conclusiones", page_conclusiones)}

# ----------------------------------------------------------------------------- app
app = dash.Dash(__name__, title="Predicción de clics con MLP · Deep Learning", suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP, "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
                           {"name": "description", "content": "MLP para predecir clics en anuncios móviles (Avazu): scikit-learn vs PySpark e interpretabilidad con LIME."}])
server = app.server


@server.route("/figs/<path:name>")
def figs(name: str):
    if not name.endswith(".png") or not (FIGS / name).exists():
        abort(404)
    return send_from_directory(FIGS, name, max_age=86400)


@server.route("/healthz")
def healthz():
    return "ok"


app.layout = html.Div([
    dcc.Location(id="url"),
    html.Header(html.Div([
        html.A([html.Span(className="brand-dot"), "Clics · MLP"], href="/", className="brand"),
        html.Nav(id="nav", className="nav"),
        html.A("Jupyter Book ↗", href=BOOK, target="_blank", className="nav-book"),
    ], className="topbar-inner"), className="topbar"),
    html.Main(id="page", className="page"),
    html.Footer(html.Div([
        html.Span("Andrés Duarte Chaves · Proyecto Integrador de Deep Learning · Maestría en Ingeniería Biomédica, Universidad del Norte · 2026"),
        html.Span(["Datos: Avazu Click-Through Rate Prediction (Kaggle), uso según las reglas de la competencia. ",
                   html.A("Código", href=REPO, target="_blank")]),
    ], className="footer-inner"), className="footer"),
])


@app.callback(Output("page", "children"), Output("nav", "children"), Input("url", "pathname"))
def render(path):
    path = path if path in PAGES else "/"
    nav = [html.A(name, href=p, className="nav-link" + (" active" if p == path else "")) for p, (name, _) in PAGES.items()]
    return PAGES[path][1](), nav


@app.callback(Output("m-curve", "figure"), Input("m-metric", "value"), Input("m-fw", "value"), Input("m-enc", "value"))
def curve(metric, fws, encs):
    d = final[final.entorno.isin(fws or []) & final.encoding.isin(encs or [])]
    fig = go.Figure()
    for (fw, fam, enc), g in d.groupby(["entorno", "familia", "encoding"]):
        g = g.sort_values("n")
        fig.add_trace(go.Scatter(
            x=g.n, y=g[metric], mode="lines+markers", name=f"{fam} · {ENC_NAME[enc]} · {fw}",
            line=dict(color=MODEL_COLOR[fam], width=2, dash=ENC_DASH[enc]),
            marker=dict(size=9, symbol="circle" if fw == "scikit-learn" else "diamond", line=dict(color="white", width=1.5)),
            customdata=np.c_[g.modelo, g.n_semillas],
            hovertemplate=f"{fw} · %{{customdata[0]}} · {ENC_NAME[enc]}<br>%{{x:,}} filas<br>%{{y:.4f}}<br>%{{customdata[1]}} semilla(s)<extra></extra>"))
    base_layout(fig, height=440)
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, yanchor="top", font=dict(size=11)))
    fig.update_xaxes(title="Filas de entrenamiento (escala log)", type="log", **N_TICKS)
    fig.update_yaxes(title=dict(roc_auc_media="ROC AUC", log_loss_media="Log loss", ece_media="ECE", train_s_media="Segundos").get(metric),
                     type="log" if metric == "train_s_media" else "linear")
    return fig


if __name__ == "__main__":
    app.run(debug=False, port=8050)
