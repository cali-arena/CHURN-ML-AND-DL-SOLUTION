"""
Apresentação em formato tese: Predição de Churn em Clientes de Telecomunicações.
Execução a partir da raiz do projeto: streamlit run app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import base64
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from preprocess import (
    load_and_clean,
    load_and_clean_for_inference,
    split_stratified,
    TARGET_COL,
    load_preprocessor,
    transform_with_preprocessor,
)

# --- Page config (estilo acadêmico, fundo claro) ---
st.set_page_config(
    page_title="Predição de Churn – Priorização e Retenção",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS: clean, premium layout on white ---
st.markdown(
    """
    <style>
    /* Base and layout */
    .stApp, [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #1a1a1a; }
    .main .block-container, [data-testid="stMain"] .block-container { max-width: 880px; margin-left: auto; margin-right: auto; padding-top: 2rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem; }
    @media (min-width: 1440px) { .main .block-container, [data-testid="stMain"] .block-container { max-width: 920px; } }

    /* Remove gray blocks and highlights */
    ::selection { background: #e3f2fd; color: #0d47a1; }
    div[data-testid="stMarkdown"], .stMarkdown { background: transparent !important; }
    .stMarkdown > div { background: transparent !important; }
    [data-testid="stVerticalBlock"] > div { background: transparent !important; }
    h1, h2, h3, h4, p, li, span { -webkit-tap-highlight-color: transparent; }

    /* Typography */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div { color: #1a1a1a !important; font-size: 1rem; line-height: 1.65; }
    .stMarkdown p { margin-bottom: 0.85em; }
    .stMarkdown h1 { font-size: 1.75rem; font-weight: 600; color: #1a1a1a !important; letter-spacing: -0.02em; margin-top: 0; margin-bottom: 0.5em; line-height: 1.25; background: none !important; }
    .stMarkdown h2 { font-size: 1.35rem; font-weight: 600; color: #1a1a1a !important; margin-top: 1.75em; margin-bottom: 0.45em; line-height: 1.3; background: none !important; padding: 0 !important; border: none !important; }
    .stMarkdown h3 { font-size: 1.15rem; font-weight: 600; color: #1a1a1a !important; margin-top: 1.35em; margin-bottom: 0.4em; line-height: 1.35; background: none !important; }
    .stMarkdown h4 { font-size: 1.05rem; font-weight: 600; color: #1a1a1a !important; margin-top: 1.1em; margin-bottom: 0.35em; line-height: 1.4; background: none !important; }
    .stMarkdown ul, .stMarkdown ol { margin-bottom: 0.85em; padding-left: 1.4em; }
    .stMarkdown li { margin-bottom: 0.25em; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #1a1a1a !important; }

    /* Sidebar: menu de tese/relatório executivo */
    section[data-testid="stSidebar"] { background: #fafbfc; border-right: 1px solid #e1e4e8; }
    section[data-testid="stSidebar"] .stMarkdown { color: #24292e !important; }
    section[data-testid="stSidebar"] .stMarkdown a { 
        display: block; 
        padding: 0.5rem 0.75rem; 
        margin: 0.1rem 0; 
        color: #24292e !important; 
        text-decoration: none !important; 
        font-size: 0.875rem; 
        line-height: 1.5; 
        border-left: 3px solid transparent;
        border-radius: 0;
        transition: all 0.2s ease;
        font-weight: 400;
    }
    section[data-testid="stSidebar"] .stMarkdown a:hover, 
    section[data-testid="stSidebar"] .stMarkdown a:focus { 
        background: #f1f3f5; 
        color: #0366d6 !important; 
        border-left-color: #0366d6;
        padding-left: 0.9rem;
    }
    section[data-testid="stSidebar"] .stMarkdown a:visited { color: #24292e !important; }
    section[data-testid="stSidebar"] .stMarkdown p { 
        margin-bottom: 0.15rem; 
        line-height: 1.6;
    }
    section[data-testid="stSidebar"] h2 { 
        font-size: 0.75rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        color: #586069 !important; 
        margin-bottom: 1rem; 
        margin-top: 0.5rem;
        padding-bottom: 0.5rem; 
        border-bottom: 2px solid #e1e4e8; 
    }
    /* Remove números dos links (se houver) e melhora espaçamento */
    section[data-testid="stSidebar"] .stMarkdown ol,
    section[data-testid="stSidebar"] .stMarkdown ul {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    section[data-testid="stSidebar"] .stMarkdown li {
        margin: 0;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Dados (em cache) ---
@st.cache_data
def get_data(csv_path: Path):
    df, _ = load_and_clean(csv_path, save_customer_ids_path=None)
    df["ChurnLabel"] = df[TARGET_COL].map({0: "Sem churn", 1: "Churn"})
    return df


@st.cache_data
def get_test_scores(_csv_path: Path, _mp: Path, _pp: Path):
    """Load test set and model predictions (same split as evaluate.py)."""
    from tensorflow import keras

    df_clean, _ = load_and_clean(_csv_path, save_customer_ids_path=None)
    _, _, _df_test = split_stratified(
        df_clean,
        target_col=TARGET_COL,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    preprocessor = load_preprocessor(_pp)
    model = keras.models.load_model(_mp)
    X_test = transform_with_preprocessor(_df_test, preprocessor)
    probs = np.asarray(model.predict(X_test, verbose=0)).ravel()
    return _df_test, probs


csv_path = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not csv_path.exists():
    st.error(f"Dataset não encontrado: {csv_path}")
    st.stop()

# --- Sidebar: Menu de Navegação (estilo tese/relatório executivo) ---
st.sidebar.markdown("## Sumário")
st.sidebar.markdown(
    """
    [Visão Geral](#visao-geral)  
    [Modelo e Features](#modelo-features)  
    [Métricas e KPIs](#metricas-kpis)  
    [Análise de Churn](#analise-churn)  
    [Simulação de Risco](#simulacao-risco)  
    [Limitações e Próximos Passos](#limitacoes)  
    [Conclusão Executiva](#conclusao-executiva)
    """,
    unsafe_allow_html=False,
)

# =============================================================================
# Conteúdo principal (layout executivo)
# =============================================================================

# --- HERO: título, subtítulo, 3 KPIs em cards ---
_df = get_data(csv_path)
_churn_pct = 100 * _df[TARGET_COL].mean()
_receita_media = float(_df["MonthlyCharges"].mean())
_receita_em_risco = float(_df.loc[_df[TARGET_COL] == 1, "MonthlyCharges"].sum())

hero = st.container()
with hero:
    _logo_path = ROOT / "26813119-rede-icone-preto-e-branco-gratis-vetor.jpg"
    if _logo_path.exists():
        with open(_logo_path, "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            f'<div style="text-align: center; margin-bottom: 0.5rem;">'
            f'<img src="data:image/jpeg;base64,{_logo_b64}" style="max-width: 88px; height: auto; mix-blend-mode: multiply;" alt="Telco" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("# Predição de Churn em Clientes de Telecomunicações")
    st.markdown("**Priorize quem está em risco, simule campanhas e estime o impacto financeiro da retenção.**")
    st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("% Churn histórico", f"{_churn_pct:.1f}%")
    with k2:
        st.metric("Receita mensal média (US$)", f"{_receita_media:.2f}")
    with k3:
        st.metric("Receita em risco (US$/mês)", f"{_receita_em_risco:,.0f}")
    st.caption("Receita em risco = cobrança mensal dos que deram churn no histórico.")
    st.markdown("<div style='height: 2.5rem'></div>", unsafe_allow_html=True)

# --- Resultados em 30 segundos (síntese executiva) ---
_metrics_path = ROOT / "reports" / "metrics" / "test_metrics.json"
if _metrics_path.exists():
    with open(_metrics_path) as _f:
        _m = json.load(_f)
    _auc = _m.get("roc_auc", 0)
    _rec = _m.get("recall", 0)
    _prec = _m.get("precision", 0)
    _topn = _m.get("top_n_capture") or []
    _top10 = next((r for r in _topn if r.get("top_pct") == 10), None)
    st.subheader("Resultados em 30 segundos")
    st.markdown("- **Ordenação** (AUC {:.2f}), **captura** (Recall {:.2f}) e **eficiência** (Precision {:.2f}) do modelo.".format(_auc, _rec, _prec))
    if _top10:
        st.markdown("- Top **10%** mais arriscados cobrem **{:.1f}%** do churn. Use capacidade e priorização para definir quem contatar.".format(_top10["capture_pct"]))
    else:
        st.markdown("- Use a seção **Métricas e KPIs** para ver Top-N e impacto operacional.")
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    _c1, _c2, _c3, _c4 = st.columns(4)
    with _c1:
        st.markdown(f'<div style="border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.6rem; text-align: center;"><strong>AUC</strong><br><span style="font-size: 1.2rem;">{_auc:.2f}</span><br><small>Ordenação</small></div>', unsafe_allow_html=True)
    with _c2:
        st.markdown(f'<div style="border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.6rem; text-align: center;"><strong>Recall</strong><br><span style="font-size: 1.2rem;">{_rec:.2f}</span><br><small>Captura</small></div>', unsafe_allow_html=True)
    with _c3:
        st.markdown(f'<div style="border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.6rem; text-align: center;"><strong>Precision</strong><br><span style="font-size: 1.2rem;">{_prec:.2f}</span><br><small>Eficiência</small></div>', unsafe_allow_html=True)
    with _c4:
        if _top10:
            st.markdown(f'<div style="border: 1px solid #1976d2; border-radius: 6px; padding: 0.6rem; text-align: center; background: #e3f2fd;"><strong>Top 10%</strong><br><span style="font-size: 1.2rem;">{_top10["capture_pct"]:.1f}%</span> do churn<br><small>Impacto</small></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.6rem; text-align: center;"><strong>Top-N</strong><br><small>Seção Métricas</small></div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
else:
    st.subheader("Resultados em 30 segundos")
    st.caption("Execute `evaluate.py` para gerar métricas e ver a síntese (AUC, Recall, Precision, Top-N).")

st.divider()

# --- 1. Visão Geral ---
st.markdown('<div id="visao-geral"></div>', unsafe_allow_html=True)
sec1 = st.container()
with sec1:
    st.header("1. Visão Geral")
    st.markdown("*O que é:* contexto do problema de churn e da solução em risco. *Por que importa:* priorizar retenção antes da saída reduz perda. *Uso:* base para o gestor definir foco de campanha e capacidade.")
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    col_problema, col_solucao = st.columns(2)
    with col_problema:
        st.subheader("Problema")
        st.markdown("- Churn gera perda de receita, custo de aquisição e efeito em indicações.")
        st.markdown("- Reagir depois do cancelamento é tarde; identificar risco antes permite agir a tempo.")

    with col_solucao:
        st.subheader("Solução")
        st.markdown("- Modelo prevê risco por cliente; base ordenada para priorizar ofertas.")
        st.markdown("- Simulação de perfil e ofertas ajuda a desenhar campanhas e alocar esforço.")

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    st.subheader("Valor de negócio")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("**Reduzir perdas**  \nAtuar nos de maior risco antes da saída.")
    with v2:
        st.markdown("**Focar retenção**  \nRecursos onde mais impacta.")
    with v3:
        st.markdown("**Apoiar decisões**  \nGestor define quem contatar e com qual oferta.")

st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

# --- 2. Modelo e Features ---
st.markdown('<div id="modelo-features"></div>', unsafe_allow_html=True)
sec2 = st.container()
with sec2:
    st.header("2. Modelo e Features")
    st.markdown("*O que é:* rede neural (MLP) que combina compromisso, valor e engajamento. *Por que importa:* ordena clientes por chance de churn. *Uso:* o gestor define limite e ações; o modelo não decide sozinho.")
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Entradas**  \n- Compromisso: tenure, contrato  \n- Valor: cobrança mensal/total  \n- Engajamento: pagamento, serviços (internet, telefonia, streaming, suporte)")
    with m2:
        st.markdown("**Saída**  \n- Probabilidade de churn (0–1)  \n- Gestor escolhe corte e ofertas")

st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

# --- 3. Métricas e KPIs do Modelo ---
st.markdown('<div id="metricas-kpis"></div>', unsafe_allow_html=True)
sec3 = st.container()
with sec3:
    st.header("3. Métricas e KPIs do Modelo")
    st.markdown("*O que é:* métricas de ordenação e captura de churners no teste. *Por que importa:* medem quão bem o modelo prioriza risco. *Uso:* o gestor avalia qualidade e decide cortes.")
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    st.subheader("Como usar o sistema na prática")
    st.markdown(
        "1) Rodar o scoring mensal e obter o ranking por risco.  \n"
        "2) Definir a capacidade operacional (ex.: 300 contatos no mês).  \n"
        "3) Aplicar a campanha de retenção e medir o impacto, usando o simulador ou testes A/B."
    )
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

metrics_path = ROOT / "reports" / "metrics" / "test_metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    # --- Foco visual: AUC, Recall, Top-N | Secundária: Acurácia ---
    auc = metrics.get("roc_auc", 0)
    rec = metrics.get("recall", 0)
    prec = metrics.get("precision", 0)
    acc = metrics.get("accuracy")
    top_n_list = metrics.get("top_n_capture") or []
    top10 = next((r for r in top_n_list if r.get("top_pct") == 10), None)

    st.subheader("Métricas que apoiam decisão de retenção")
    st.markdown("**Foco:** AUC, Recall e Top-N capture. Cada uma com *O que mede* e *Por que importa para retenção*.")
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            '<div style="border: 1px solid #1976d2; border-radius: 8px; padding: 1rem; background: #e3f2fd;">'
            '<strong>AUC-ROC</strong><br><span style="font-size: 1.5rem;">{:.2f}</span><br><br>'
            '<strong>O que mede:</strong> Capacidade do modelo de ordenar clientes por risco (quanto maior, melhor a separação entre alto e baixo risco).<br><br>'
            '<strong>Por que importa para retenção:</strong> Garante que a lista priorizada coloque os mais arriscados no topo; sem boa ordenação, esforço de retenção é mal alocado.</div>'.format(auc),
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            '<div style="border: 1px solid #1976d2; border-radius: 8px; padding: 1rem; background: #e3f2fd;">'
            '<strong>Recall (Churn)</strong><br><span style="font-size: 1.5rem;">{:.2f}</span><br><br>'
            '<strong>O que mede:</strong> Dos clientes que de fato saíram, quantos % o modelo colocou entre os marcados em risco.<br><br>'
            '<strong>Por que importa para retenção:</strong> Traduz em “não perder quem ia sair”; recall alto significa que a retenção tem chance de alcançar a maioria dos churners.</div>'.format(rec),
            unsafe_allow_html=True,
        )
    with f3:
        capt_txt = f"{top10['capture_pct']:.1f}%" if top10 else "—"
        st.markdown(
            '<div style="border: 1px solid #1976d2; border-radius: 8px; padding: 1rem; background: #e3f2fd;">'
            '<strong>Top-N capture (ex. top 10%)</strong><br><span style="font-size: 1.5rem;">' + capt_txt + '</span> do churn<br><br>'
            '<strong>O que mede:</strong> Ao contatar só os X% mais arriscados, qual % dos churners reais você alcança.<br><br>'
            '<strong>Por que importa para retenção:</strong> Responde “se eu só puder ligar para 10% da base, quanto do churn cubro?”; direto na decisão de capacidade e meta.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    st.markdown("**Métricas secundárias**")
    if acc is not None:
        st.caption("**Acurácia** {:.1%} — % de acertos no corte usado (ficou/saiu). Útil para contexto; para retenção, AUC e Recall pesam mais.".format(acc))
    st.caption("**Precision** {:.2f} — Dos marcados em risco, quantos de fato saíram. Ajuda a calibrar custo de oferta em vão.".format(prec))
    st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)

    # --- ROC Curve ---
    st.subheader("Curva ROC")
    st.markdown("**O que este gráfico mostra:** se o modelo consegue colocar os mais arriscados no topo da lista (curva acima da diagonal = melhor que sorte).  \n**Como o gestor usa esta informação:** valida se a priorização faz sentido antes de definir quantos clientes contatar; curva que sobe cedo à esquerda indica que o ranking está útil.")
    st.markdown("<div style='height: 0.35rem'></div>", unsafe_allow_html=True)
    fpr = metrics.get("fpr")
    tpr = metrics.get("tpr")
    roc_auc = metrics.get("roc_auc", 0)
    if fpr is not None and tpr is not None:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Modelo (AUC = {roc_auc:.3f})", line=dict(color="#1976d2", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório", line=dict(color="#9e9e9e", width=1.5, dash="dash")))
        fig_roc.update_layout(
            title=dict(text="Curva ROC — Ordenação por risco", font=dict(size=16)),
            xaxis_title="Taxa de falsos positivos",
            yaxis_title="Taxa de verdadeiros positivos",
            xaxis=dict(range=[0, 1], showgrid=True, gridwidth=1, gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(range=[0, 1.02], showgrid=True, gridwidth=1, gridcolor="#f0f0f0", zeroline=False),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(t=48, b=44, l=52, r=24),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        roc_img = metrics.get("roc_plot_path")
        if roc_img and Path(roc_img).exists():
            st.image(roc_img, use_container_width=True)
        else:
            st.caption("Execute `evaluate.py` para gerar curva ROC.")
    st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)

    # --- Confusion Matrix ---
    st.subheader("Matriz de confusão")
    st.markdown("**O que este gráfico mostra:** quantos clientes o modelo acertou ou errou ao prever “fica” ou “sai” para um corte escolhido.  \n**Como o gestor usa esta informação:** vê o custo de errar (oferta em quem ficou x não contatar quem saiu) e decide se vale subir ou descer o corte conforme o orçamento de retenção.")
    st.markdown("<div style='height: 0.35rem'></div>", unsafe_allow_html=True)
    cm = metrics.get("confusion_matrix")
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        cm_arr = np.array(cm)
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_arr,
                x=["Previsto: Fica", "Previsto: Sai"],
                y=["Real: Fica", "Real: Sai"],
                colorscale=[[0, "#e8f5e9"], [0.5, "#81c784"], [1, "#2e7d32"]],
                text=cm_arr,
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=dict(text="Matriz de confusão (teste)", font=dict(size=16)),
            xaxis_title="Predição",
            yaxis_title="Real",
            margin=dict(t=48, b=44, l=52, r=24),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.caption("Execute `evaluate.py` para gerar matriz.")
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # --- Ranking Top-N: captura de churners (destaque) ---
    st.subheader("Ranking Top-N: captura de churners")
    st.markdown("**O que este gráfico mostra:** ao falar só com os X% mais arriscados, qual % de quem de fato saiu você alcança.  \n**Como o gestor usa esta informação:** bate capacidade (“contato nos top 20%”) com impacto (“cubro Y% do churn”) para dimensionar equipe e meta de retenção.")
    st.markdown("<div style='height: 0.35rem'></div>", unsafe_allow_html=True)
    top_n = metrics.get("top_n_capture") or []
    if top_n:
        top_df = pd.DataFrame(top_n)
        fig_top = go.Figure(go.Bar(x=top_df["top_pct"].astype(str) + "%", y=top_df["capture_pct"], text=top_df["capture_pct"].round(1), textposition="auto", marker_color="#1976d2", textfont=dict(size=11)))
        fig_top.update_layout(
            title=dict(text="% do churn total capturado ao contatar top X% por risco", font=dict(size=16)),
            xaxis_title="Top % de clientes contatados (por ordem de risco)",
            yaxis_title="% dos churners capturados",
            margin=dict(t=48, b=44, l=52, r=24),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#f0f0f0", range=[0, max(100, top_df["capture_pct"].max() * 1.05) if len(top_df) else 100]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.dataframe(top_df.rename(columns={"top_pct": "Top %", "capture_pct": "% churners capturados", "n_captured": "Churners capturados"}), use_container_width=True, hide_index=True)
    else:
        st.caption("Execute `evaluate.py` para gerar Top-N.")
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # --- Decile Lift / Gain (destaque) ---
    st.subheader("Decile Lift / Gain")
    st.markdown("**O que este gráfico mostra:** quanto do churn total você cobre ao contatar, em ordem de risco, só os primeiros X% da base.  \n**Como o gestor usa esta informação:** lê no eixo horizontal a % da base que pode contatar e no vertical o % do churn coberto; curva que sobe cedo confirma que priorizar por risco gera ganho rápido.")
    st.markdown("<div style='height: 0.35rem'></div>", unsafe_allow_html=True)
    decile = metrics.get("decile_gain") or []
    if decile:
        dg = pd.DataFrame(decile)
        fig_dg = go.Figure(go.Scatter(x=dg["pct_contacted"], y=dg["cumulative_pct_churn"], mode="lines+markers", name="Churn acumulado", line=dict(color="#1976d2", width=2.5), marker=dict(size=8, color="#1976d2")))
        fig_dg.update_layout(
            title=dict(text="Ganho acumulado: % do churn capturado ao contatar X% da base (por risco)", font=dict(size=16)),
            xaxis_title="% de clientes contatados (por ordem de risco)",
            yaxis_title="% do churn total capturado (acumulado)",
            xaxis=dict(dtick=10, showgrid=True, gridwidth=1, gridcolor="#f0f0f0"),
            yaxis=dict(range=[0, 105], showgrid=True, gridwidth=1, gridcolor="#f0f0f0"),
            margin=dict(t=48, b=44, l=52, r=24),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dg, use_container_width=True)
        st.dataframe(dg.rename(columns={"decile": "Decil", "pct_contacted": "% contatados", "churn_in_decile": "Churn no decil", "pct_churn_decile": "% churn (decil)", "cumulative_pct_churn": "% churn acumulado"}), use_container_width=True, hide_index=True)
        x_20 = 20
        x_30 = 30
        y_20 = next((r["cumulative_pct_churn"] for r in decile if r["pct_contacted"] == 20), None)
        y_30 = next((r["cumulative_pct_churn"] for r in decile if r["pct_contacted"] == 30), None)
        if y_20 is not None:
            st.info(f"**Exemplo de decisão:** com foco nos **{x_20}%** mais arriscados, você captura **{y_20:.1f}%** do churn total.")
        if y_30 is not None and y_30 != y_20:
            st.caption(f"Nos **{x_30}%** mais arriscados: **{y_30:.1f}%** do churn total.")
    else:
        st.caption("Execute `evaluate.py` para gerar Decile Lift/Gain.")

    # --- Priorização operacional (capacity-driven top-N) ---
    st.subheader("Priorização operacional")
    _model_path = ROOT / "models" / "best_model" / "churn_mlp.keras"
    _preprocessor_path = ROOT / "models" / "preprocessor" / "preprocessor.joblib"
    if not _model_path.exists() or not _preprocessor_path.exists():
        st.warning("Requer modelo treinado. Execute `train.py` e `evaluate.py`.")
    else:
        df_test, probs = get_test_scores(csv_path, _model_path, _preprocessor_path)
        n_test = len(df_test)
        default_n = min(100, max(1, n_test // 5))
        N = st.slider(
            "Capacidade mensal de contato (clientes)",
            min_value=1,
            max_value=n_test,
            value=default_n,
        )
        order = np.argsort(-probs)
        top_N_indices = order[:N]
        implied_threshold = float(probs[top_N_indices[-1]]) if N else 0.0
        y_test = np.asarray(df_test[TARGET_COL].values, dtype=np.float32)
        churners_captured = int(np.sum(y_test[top_N_indices]))
        charges_top = df_test["MonthlyCharges"].iloc[top_N_indices].values
        revenue_covered = float(np.sum(charges_top * y_test[top_N_indices]))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Threshold implícito", f"{implied_threshold:.3f}")
        with c2:
            st.metric("Churners capturados (esperado)", churners_captured)
        with c3:
            st.metric("Receita em risco coberta (US$/mês)", f"{revenue_covered:,.2f}")
        st.caption("Threshold definido pela capacidade, não pela probabilidade.")

        # --- Simulador de Retenção (calculadora de ROI) ---
        st.subheader("Simulador de ação de retenção")
        st.markdown("Calcule o **impacto** de uma campanha: insira **Campanha** (entradas) e veja **Impacto** (churn evitado, receita preservada, custo e benefício líquido).")
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        high_risk_pct = 20
        high_risk_pool_size = max(1, int(round(n_test * high_risk_pct / 100)))

        col_campanha, col_impacto = st.columns(2)
        with col_campanha:
            st.markdown("**Campanha — Entradas**")
            with st.container():
                pct_contacted = st.slider(
                    "% de clientes de alto risco contatados",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Dos clientes no top 20% por risco, que fração você contata.",
                )
                success_rate = st.slider(
                    "Taxa de sucesso da retenção (%)",
                    min_value=0,
                    max_value=100,
                    value=25,
                    help="Dos contatados que iam sair, quantos % permanecem após a ação.",
                )
                cost_per_contact = st.slider(
                    "Custo por contato (US$)",
                    min_value=0.0,
                    max_value=100.0,
                    value=15.0,
                    step=1.0,
                    format="%.1f",
                    help="Custo médio por tentativa.",
                )
                avg_discount = st.slider(
                    "Desconto médio oferecido (US$/mês)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    format="%.1f",
                    help="Desconto típico ao retido.",
                )

        n_contacted = max(0, int(round(pct_contacted / 100 * high_risk_pool_size)))
        if n_contacted > 0:
            contacted_idx = order[:n_contacted]
            churners_among_contacted = int(np.sum(y_test[contacted_idx]))
            churn_avoided = churners_among_contacted * (success_rate / 100)
            charges_contacted_churners = df_test["MonthlyCharges"].iloc[contacted_idx].values * y_test[contacted_idx]
            avg_charge_churners = float(np.sum(charges_contacted_churners)) / max(1, churners_among_contacted)
            revenue_per_retained_per_month = max(0.0, avg_charge_churners - avg_discount)
            months_horizon = 12
            revenue_preserved = churn_avoided * revenue_per_retained_per_month * months_horizon
            total_cost = n_contacted * cost_per_contact
            net_benefit = revenue_preserved - total_cost
        else:
            churn_avoided = revenue_preserved = total_cost = net_benefit = 0.0

        with col_impacto:
            st.markdown("**Impacto — Saídas**")
            st.markdown(
                '<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem; background: #fafafa;">'
                '<strong style="color: #616161;">Churn evitado</strong><br>'
                f'<span style="font-size: 1.4rem; font-weight: 600;">{churn_avoided:.1f} clientes</span><br>'
                '<small>Retidos pela ação.</small></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem; background: #fafafa;">'
                '<strong style="color: #616161;">Receita preservada (12 meses)</strong><br>'
                f'<span style="font-size: 1.4rem; font-weight: 600;">US$ {revenue_preserved:,.0f}</span><br>'
                '<small>Receita líquida dos retidos (após desconto).</small></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem; background: #fff3e0;">'
                '<strong style="color: #616161;">Custo da campanha</strong><br>'
                f'<span style="font-size: 1.4rem; font-weight: 600;">US$ {total_cost:,.0f}</span></div>',
                unsafe_allow_html=True,
            )
            net_color = "#2e7d32" if net_benefit >= 0 else "#c62828"
            st.markdown(
                f'<div style="border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem; background: {net_color}18; border-left: 4px solid {net_color};">'
                '<strong style="color: #424242;">Benefício líquido</strong><br>'
                f'<span style="font-size: 1.4rem; font-weight: 600; color: {net_color};">US$ {net_benefit:,.0f}</span><br>'
                '<small>Receita preservada − custo.</small></div>',
                unsafe_allow_html=True,
            )
        st.info(
            "**Premissas:** Alto risco = top 20% por score no conjunto de teste. "
            "Taxa de sucesso e custos são valores **hipotéticos**; o resultado é ilustrativo. "
            "Ajuste com dados reais de campanhas e custos da sua operação."
        )

    st.caption(
        "Resultados reais do conjunto de teste, carregados de `reports/metrics/test_metrics.json`. "
        "Para atualizar no futuro, rode `evaluate.py` localmente e suba o novo JSON para o Git."
    )
else:
    st.markdown("Métricas geradas por `evaluate.py` em `reports/metrics/test_metrics.json`. Execute na raiz do projeto.")

st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
st.markdown('<div id="analise-churn"></div>', unsafe_allow_html=True)
sec4 = st.container()
with sec4:
    st.header("4. Análise de Churn")
    st.markdown("*O que é:* distribuição de churn e relação com tenure/cobrança. *Por que importa:* identifica drivers de risco nos dados. *Uso:* gestor filtra por contrato e pagamento e analisa gráficos.")
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

df = get_data(csv_path)

a1, a2 = st.columns(2)
with a1:
    st.markdown("**Aumentam risco:**  \n- Tenure curto, contrato mês a mês  \n- Cobrança alta, pagamento manual  \n- Pouco uso de serviços")
with a2:
    st.markdown("**Reduzem risco:**  \n- Contrato anual/bienal  \n- Débito automático  \nUse os filtros para refinar.")

with st.expander("Filtros (tipo de contrato e forma de pagamento)", expanded=False):
    contract_options = sorted(df["Contract"].dropna().unique().tolist())
    payment_options = sorted(df["PaymentMethod"].dropna().unique().tolist())
    selected_contracts = st.multiselect("Tipo de contrato", options=contract_options, default=contract_options)
    selected_payments = st.multiselect("Forma de pagamento", options=payment_options, default=payment_options)
mask_c = df["Contract"].isin(selected_contracts) if selected_contracts else pd.Series(True, index=df.index)
mask_p = df["PaymentMethod"].isin(selected_payments) if selected_payments else pd.Series(True, index=df.index)
df_f = df.loc[mask_c & mask_p]
st.caption(f"Registros após filtros: **{len(df_f)}**.")

if len(df_f) == 0:
    st.warning("Nenhum registro com os filtros atuais. Ajuste tipo de contrato e/ou forma de pagamento.")
else:
    counts = df_f["ChurnLabel"].value_counts().sort_index()
    fig_churn = go.Figure(
        data=[go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), text=counts.values.tolist(), textposition="auto", marker_color=["#2e7d32", "#c62828"])],
        layout=go.Layout(xaxis_title="Churn", yaxis_title="Contagem", showlegend=False, margin=dict(t=40), hovermode="x"),
    )
    fig_churn.update_traces(hovertemplate="Churn: %{x}<br>Contagem: %{y}<extra></extra>")
    st.plotly_chart(fig_churn, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**O que este gráfico mostra:** quanto tempo os clientes ficam com a empresa, separando quem permaneceu de quem saiu.  \n**Como o gestor usa esta informação:** ver se quem sai é mais cliente novo ou antigo e onde focar retenção (onboarding, fidelidade).")
        fig_tenure = px.box(df_f, x="ChurnLabel", y="tenure", color="ChurnLabel", color_discrete_map={"Sem churn": "#2e7d32", "Churn": "#c62828"}, category_orders={"ChurnLabel": ["Sem churn", "Churn"]})
        fig_tenure.update_layout(showlegend=False, xaxis_title="Churn", yaxis_title="Tempo de relação (meses)", margin=dict(t=40))
        st.plotly_chart(fig_tenure, use_container_width=True)
    with c2:
        st.markdown("**O que este gráfico mostra:** valor da conta mensal dos clientes, comparando quem ficou com quem saiu.  \n**Como o gestor usa esta informação:** entender se o valor cobrado influencia a saída e ajustar ofertas ou preços nas ações de retenção.")
        fig_mc = px.box(df_f, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel", color_discrete_map={"Sem churn": "#2e7d32", "Churn": "#c62828"}, category_orders={"ChurnLabel": ["Sem churn", "Churn"]})
        fig_mc.update_layout(showlegend=False, xaxis_title="Churn", yaxis_title="Cobrança mensal (US$)", margin=dict(t=40))
        st.plotly_chart(fig_mc, use_container_width=True)

# --- Playbook de Retenção (Medidas Operacionais) — sempre visível na seção 4 ---
st.subheader("Playbook de Retenção (Medidas Operacionais)")
st.caption("Com base nos padrões que o modelo considera, a tabela abaixo apoia a decisão de retenção. Não implica causa única.")
_playbook = pd.DataFrame({
    "Segmento": ["Contrato", "Pagamento", "Cobrança", "Internet", "Relacionamento"],
    "Sinal de risco": [
        "Contrato mês a mês",
        "Pagamento manual (cheque, boleto etc.)",
        "Cobrança mensal alta",
        "Internet fibra",
        "Pouco tempo de relação com a empresa",
    ],
    "Ação sugerida": [
        "Ofertar contrato anual ou bienal; desconto por compromisso.",
        "Incentivar débito automático ou pagamento por cartão.",
        "Revisar plano e ofertas; considerar descontos para adequar ao valor percebido.",
        "Acompanhar reclamações e qualidade; ofertar suporte ou ajuste de plano.",
        "Priorizar onboarding e ofertas de fidelização nos primeiros meses.",
    ],
})
st.dataframe(_playbook, use_container_width=True, hide_index=True)

st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
st.markdown('<div id="simulacao-risco"></div>', unsafe_allow_html=True)
sec5 = st.container()
with sec5:
    st.header("5. Simulação de Risco")
    st.markdown("*O que é:* probabilidade de churn e nível de risco por perfil. *Por que importa:* testa cenários antes de contatar. *Uso:* gestor altera perfil e vê risco + ações sugeridas.")
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

model_path = ROOT / "models" / "best_model" / "churn_mlp.keras"
preprocessor_path = ROOT / "models" / "preprocessor" / "preprocessor.joblib"
if not model_path.exists() or not preprocessor_path.exists():
    st.warning("Modelo não encontrado. Execute `python src/train.py` na raiz do projeto.")
else:
    @st.cache_resource
    def load_model_and_preprocessor(_mp: Path, _pp: Path):
        from tensorflow import keras
        return keras.models.load_model(_mp), load_preprocessor(_pp)

    model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)

    st.caption("**Faixas:** Baixo &lt; 30%, Médio 30–60%, Alto &gt; 60%. Estimativas; resultado apoia decisão.")
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    def _opt(col): return sorted(df[col].dropna().unique().tolist())
    def _idx(col):
        opts = _opt(col)
        if not opts: return 0
        try: m = df[col].mode().iloc[0]
        except (IndexError, KeyError): return 0
        return opts.index(m) if m in opts else 0

    col_form, col_result = st.columns(2)
    with col_form:
        st.markdown("**Formulário — Perfil do cliente**")
        with st.container():
            tenure = st.slider("Tempo de relação (meses)", int(df["tenure"].min()), int(df["tenure"].max()), int(df["tenure"].median()))
            monthly_charges = st.slider("Cobrança mensal (US$)", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), float(df["MonthlyCharges"].median()), format="%.2f")
            total_charges = st.slider("Total cobrado até o momento (US$)", 0.0, float(df["TotalCharges"].max()), float(df["TotalCharges"].median()), format="%.2f")
            senior = st.selectbox("Idoso (65+)", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim", index=0)
            contract = st.selectbox("Tipo de contrato", _opt("Contract"), index=_idx("Contract"))
            payment = st.selectbox("Forma de pagamento", _opt("PaymentMethod"), index=_idx("PaymentMethod"))
            internet = st.selectbox("Internet", _opt("InternetService"), index=_idx("InternetService"))
            gender = st.selectbox("Gênero", _opt("gender"), index=_idx("gender"))
            partner = st.selectbox("Parceiro(a)", _opt("Partner"), index=_idx("Partner"))
            dependents = st.selectbox("Dependentes", _opt("Dependents"), index=_idx("Dependents"))
            phone = st.selectbox("Telefonia", _opt("PhoneService"), index=_idx("PhoneService"))
            multi = st.selectbox("Múltiplas linhas", _opt("MultipleLines"), index=_idx("MultipleLines"))
            onl_sec = st.selectbox("Segurança online", _opt("OnlineSecurity"), index=_idx("OnlineSecurity"))
            onl_bak = st.selectbox("Backup online", _opt("OnlineBackup"), index=_idx("OnlineBackup"))
            dev_prot = st.selectbox("Proteção de dispositivo", _opt("DeviceProtection"), index=_idx("DeviceProtection"))
            tech = st.selectbox("Suporte técnico", _opt("TechSupport"), index=_idx("TechSupport"))
            stream_tv = st.selectbox("Streaming TV", _opt("StreamingTV"), index=_idx("StreamingTV"))
            stream_mov = st.selectbox("Streaming filmes", _opt("StreamingMovies"), index=_idx("StreamingMovies"))
            paperless = st.selectbox("Fatura sem papel", _opt("PaperlessBilling"), index=_idx("PaperlessBilling"))

    profile = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges, "SeniorCitizen": senior,
        "gender": gender, "Partner": partner, "Dependents": dependents, "PhoneService": phone, "MultipleLines": multi,
        "InternetService": internet, "OnlineSecurity": onl_sec, "OnlineBackup": onl_bak, "DeviceProtection": dev_prot,
        "TechSupport": tech, "StreamingTV": stream_tv, "StreamingMovies": stream_mov, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
    }])
    X = transform_with_preprocessor(profile, preprocessor, target_col=None)
    proba = float(np.asarray(model.predict(X, verbose=0)).ravel()[0])
    risk_label = "Baixo" if proba < 0.3 else ("Médio" if proba <= 0.6 else "Alto")
    risk_drivers = []
    if contract == "Month-to-month":
        risk_drivers.append({"driver": "Contrato mês a mês", "why": "Quem está sem compromisso de prazo pode sair a qualquer momento, sem custo de saída.", "action": "Oferecer plano anual ou bienal com benefício (desconto, upgrade) em troca de fidelidade."})
    if tenure < 12:
        risk_drivers.append({"driver": f"Pouco tempo de relação ({tenure} meses)", "why": "Cliente ainda não criou hábito nem vínculo; troca de operadora pode parecer simples.", "action": "Reforçar valor do plano (uso, benefícios), contato proativo e oferta de suporte nos primeiros meses."})
    if "Electronic check" in str(payment) or "Mailed check" in str(payment):
        risk_drivers.append({"driver": "Pagamento manual (cheque)", "why": "Cada mês exige ação do cliente; atrasos e esquecimentos aumentam insatisfação e risco de saída.", "action": "Incentivar débito automático ou cartão, com pequeno benefício ou simplificação."})
    mc_med = float(df["MonthlyCharges"].median())
    if monthly_charges > mc_med * 1.2:
        risk_drivers.append({"driver": f"Cobrança mensal alta (US$ {monthly_charges:.1f})", "why": "Valor alto sem benefício claro pode gerar sensação de custo excessivo e vontade de cortar.", "action": "Mostrar o que está incluso no plano, ajustar oferta ao uso ou oferecer downgrade guiado em vez de cancelamento."})
    if "Fiber" in str(internet) or str(internet).lower() == "fiber optic":
        risk_drivers.append({"driver": "Internet fibra", "why": "Expectativa de qualidade e velocidade é alta; qualquer falha ou custo percebido pesa mais.", "action": "Garantir suporte ágil, revisar queixas e oferta de valor (velocidade, pacotes) antes que procure concorrente."})
    top3 = risk_drivers[:3]
    is_high_risk = proba >= 0.3

    with col_result:
        st.markdown("**Resultado — Risco e ações**")
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: #fafafa;">'
            '<strong style="color: #616161;">Probabilidade de churn</strong><br>'
            f'<span style="font-size: 1.75rem; font-weight: 600;">{proba:.1%}</span></div>',
            unsafe_allow_html=True,
        )
        risk_color = "#2e7d32" if risk_label == "Baixo" else ("#f9a825" if risk_label == "Médio" else "#c62828")
        st.markdown(
            f'<div style="border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: {risk_color}18; border-left: 4px solid {risk_color};">'
            '<strong style="color: #424242;">Nível de risco</strong><br>'
            f'<span style="font-size: 1.35rem; font-weight: 600; color: {risk_color};">{risk_label}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, proba))
        st.markdown("**Top 3 fatores + ação sugerida**")
        if not is_high_risk:
            st.caption("Risco baixo. Altere o perfil à esquerda para ver fatores e ações.")
        elif not top3:
            st.caption("Nenhum fator de alto impacto neste perfil.")
        else:
            for i, d in enumerate(top3, 1):
                st.markdown(
                    f'<div style="border: 1px solid #e8e8e8; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem; background: #fff;">'
                    f'<strong>{i}. {d["driver"]}</strong><br>'
                    f'<small style="color: #616161;">Por que: {d["why"]}</small><br>'
                    f'<small><strong>Ação:</strong> {d["action"]}</small></div>',
                    unsafe_allow_html=True,
                )

st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
st.markdown('<div id="limitacoes"></div>', unsafe_allow_html=True)
sec6 = st.container()
with sec6:
    st.header("6. Limitações e Próximos Passos")
    st.markdown("*O que é:* restrições do dado e do modelo. *Por que importa:* evita uso indevido. *Uso:* gestor considera limitações nas decisões.")
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("**Limitações**  \n- Sem reclamações, logs ou histórico de campanhas  \n- Correlações históricas; use escore + julgamento")
    with l2:
        st.markdown("**Próximos passos**  \n- Reclamações/NPS, logs, engajamento  \n- Retreinamento periódico (ex.: trimestral)")
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
st.markdown('<div id="conclusao-executiva"></div>', unsafe_allow_html=True)
sec7 = st.container()
with sec7:
    st.header("Conclusão Executiva")
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    st.markdown("**Entrega**  \n- Priorização por risco e foco em clientes em risco  \n- Estimativa de impacto (churn evitado, receita preservada, benefício líquido)  \n- Ações sugeridas por perfil (contrato, pagamento, cobrança, suporte)")
    st.markdown("**Não faz**  \n- Não substitui o gestor; quem contatar e com qual oferta é decisão humana  \n- Não garante causalidade; correlações históricas, não prova efeito sem testes  \n- Não usa dados em tempo real (reclamações, NPS, logs ficam de fora)")
    st.markdown("**Próximos passos**  \n- Dados: reclamações, NPS, uso de app/suporte, histórico de ofertas  \n- Testes A/B: ofertas em grupos controlados para calibrar simulador  \n- Retreinamento: cadência (ex.: trimestral) com dados recentes")
    st.markdown("**Este sistema apoia decisões humanas ao priorizar risco e estimar impacto financeiro; ele não substitui gestores.**")
