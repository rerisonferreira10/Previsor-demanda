# =============================================================================
# Previsor de Demanda Semanal
# Aplicativo de apoio ao planejamento da producao com metodos simples
# de previsao de demanda.
# Atividade pratica - Administracao da Producao
# Desenvolvido com apoio de IA generativa (vibe coding)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# CONFIGURACAO GERAL DA PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Previsor de Demanda Semanal",
    page_icon="📦",
    layout="wide",
)

# -----------------------------------------------------------------------------
# FUNCOES DE CALCULO - METODOS DE PREVISAO
# Cada funcao recebe a serie historica (lista/array de numeros) e devolve
# as previsoes para "n_semanas" futuras.
# -----------------------------------------------------------------------------

def metodo_ingenuo(serie, n_semanas):
    """A previsao repete sempre o ultimo valor observado."""
    ultimo = serie[-1]
    return [ultimo] * n_semanas


def metodo_media_movel(serie, n_semanas, janela=3):
    """Media das ultimas 'janela' semanas. Repete a mesma previsao
    para todas as semanas futuras (abordagem simples e didatica)."""
    janela = min(janela, len(serie))
    media = np.mean(serie[-janela:])
    return [media] * n_semanas


def metodo_suavizacao_exponencial(serie, n_semanas, alfa=0.3):
    """Suavizacao exponencial simples: F(t+1) = alfa*D(t) + (1-alfa)*F(t).
    A previsao inicial (F1) e o primeiro valor da serie."""
    previsao = serie[0]
    for valor_real in serie[1:]:
        previsao = alfa * valor_real + (1 - alfa) * previsao
    # A previsao final "convergida" e repetida para as semanas futuras
    return [previsao] * n_semanas


def metodo_regressao_linear(serie, n_semanas):
    """Regressao linear simples usando o indice da semana como variavel X."""
    X = np.arange(len(serie)).reshape(-1, 1)
    y = np.array(serie)
    modelo = LinearRegression()
    modelo.fit(X, y)
    X_futuro = np.arange(len(serie), len(serie) + n_semanas).reshape(-1, 1)
    previsoes = modelo.predict(X_futuro)
    # Demanda nao pode ser negativa
    previsoes = np.clip(previsoes, 0, None)
    return list(previsoes)


def calcular_erro_mae(serie):
    """Calcula o Erro Medio Absoluto (MAE) de cada metodo usando uma
    validacao simples 'um passo a frente' dentro do proprio historico."""
    erros = {"Ingenuo": [], "Media Movel": [], "Suavizacao Exponencial": [], "Regressao Linear": []}

    minimo_pontos = 4
    if len(serie) < minimo_pontos:
        return None

    for i in range(minimo_pontos - 1, len(serie) - 1):
        historico_parcial = serie[: i + 1]
        valor_real_seguinte = serie[i + 1]

        p_ing = metodo_ingenuo(historico_parcial, 1)[0]
        p_mm = metodo_media_movel(historico_parcial, 1)[0]
        p_se = metodo_suavizacao_exponencial(historico_parcial, 1)[0]
        p_rl = metodo_regressao_linear(historico_parcial, 1)[0]

        erros["Ingenuo"].append(abs(valor_real_seguinte - p_ing))
        erros["Media Movel"].append(abs(valor_real_seguinte - p_mm))
        erros["Suavizacao Exponencial"].append(abs(valor_real_seguinte - p_se))
        erros["Regressao Linear"].append(abs(valor_real_seguinte - p_rl))

    mae_final = {nome: float(np.mean(valores)) for nome, valores in erros.items() if valores}
    return mae_final


def classificar_tendencia(serie):
    """Classifica a tendencia geral da serie comparando a primeira e a
    segunda metade do historico."""
    metade = len(serie) // 2
    primeira_metade = np.mean(serie[:metade]) if metade > 0 else serie[0]
    segunda_metade = np.mean(serie[metade:])

    variacao_pct = (segunda_metade - primeira_metade) / primeira_metade * 100 if primeira_metade != 0 else 0

    desvio_padrao = np.std(serie)
    media_geral = np.mean(serie)
    coef_variacao = (desvio_padrao / media_geral * 100) if media_geral != 0 else 0

    if coef_variacao > 20:
        return "irregular", variacao_pct, coef_variacao
    elif variacao_pct > 8:
        return "crescente", variacao_pct, coef_variacao
    elif variacao_pct < -8:
        return "decrescente", variacao_pct, coef_variacao
    else:
        return "estavel", variacao_pct, coef_variacao


def gerar_recomendacao_gerencial(serie, previsoes_media, tendencia):
    """Gera o texto de interpretacao gerencial, conforme a secao 13
    do guia da atividade."""
    media_historica = np.mean(serie)
    media_prevista = np.mean(previsoes_media)

    mensagens = {
        "crescente": (
            "📈 **Demanda com tendencia de crescimento.** "
            "Recomenda-se verificar se a capacidade produtiva atual sera suficiente "
            "para atender as proximas semanas. Avalie antecipar compras de materia-prima "
            "e reforcar a equipe de producao caso o crescimento se confirme."
        ),
        "decrescente": (
            "📉 **Demanda com tendencia de queda.** "
            "Recomenda-se cautela na producao para evitar excesso de estoque e aumento "
            "dos custos de armazenagem. Considere reduzir lotes de producao e revisar "
            "pedidos de compra ja programados."
        ),
        "estavel": (
            "➖ **Demanda relativamente estavel.** "
            "A empresa pode usar a previsao como referencia para manter o planejamento "
            "de producao atual, sem grandes ajustes de capacidade."
        ),
        "irregular": (
            "⚠️ **Demanda com alta variacao (irregular).** "
            "Recomenda-se analisar fatores externos, sazonalidade, promocoes, eventos ou "
            "comportamento dos clientes antes de decidir. Previsoes para series irregulares "
            "tendem a ser menos confiaveis."
        ),
    }

    risco_falta = media_prevista > media_historica * 1.1
    risco_excesso = media_prevista < media_historica * 0.9

    texto = mensagens[tendencia] + "\n\n"

    if risco_falta:
        texto += "🔺 Ha indicio de **risco de falta de produto** caso a producao nao seja ajustada para cima.\n\n"
    if risco_excesso:
        texto += "🔻 Ha indicio de **risco de excesso de estoque** caso a producao nao seja reduzida.\n\n"
    if not risco_falta and not risco_excesso:
        texto += "✅ A previsao media nao indica forte risco de falta nem de excesso, mas continue monitorando.\n\n"

    texto += (
        "_Lembre-se: previsao de demanda nao e certeza. Use o resultado como apoio a decisao, "
        "combinando com o julgamento gerencial e o conhecimento do negocio._"
    )
    return texto


# -----------------------------------------------------------------------------
# INTERFACE - CABECALHO
# -----------------------------------------------------------------------------
st.title("📦 Previsor de Demanda Semanal")
st.caption(
    "Aplicativo de apoio ao planejamento da producao | "
    "Atividade pratica de Administracao da Producao"
)

with st.expander("ℹ️ Como usar este aplicativo", expanded=False):
    st.markdown(
        """
        1. Informe o **nome do produto** e as **demandas historicas semanais**.
        2. Escolha **quantas semanas futuras** deseja prever.
        3. Selecione um ou mais **metodos de previsao**.
        4. Veja a **tabela**, o **grafico** e a **recomendacao gerencial**.

        ⚠️ **Atencao:** previsao de demanda e uma estimativa baseada no passado,
        **nao e uma garantia do futuro**. Use os resultados com senso critico.
        """
    )

st.divider()

# -----------------------------------------------------------------------------
# SECAO 1 - ENTRADA DE DADOS
# -----------------------------------------------------------------------------
st.header("1. Entrada de dados")

col_a, col_b = st.columns([2, 1])

with col_a:
    nome_produto = st.text_input("Nome do produto", value="Produto A")

    st.markdown("**Demandas historicas semanais**")
    st.caption("Informe de 4 a 12 valores separados por virgula (ex.: 120, 125, 130, 128)")

    entrada_texto = st.text_area(
        "Demandas (separadas por virgula)",
        value="120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165",
        height=80,
        label_visibility="collapsed",
    )

with col_b:
    n_semanas_futuras = st.number_input(
        "Semanas futuras a prever", min_value=1, max_value=12, value=4, step=1
    )
    janela_media_movel = st.number_input(
        "Janela da media movel (semanas)", min_value=2, max_value=8, value=3, step=1
    )
    alfa_suavizacao = st.slider(
        "Alfa da suavizacao exponencial", min_value=0.05, max_value=0.9, value=0.3, step=0.05
    )

metodos_selecionados = st.multiselect(
    "Metodos de previsao a aplicar",
    ["Ingenuo", "Media Movel", "Suavizacao Exponencial", "Regressao Linear"],
    default=["Media Movel", "Suavizacao Exponencial", "Regressao Linear"],
)

# -----------------------------------------------------------------------------
# VALIDACAO DOS DADOS DE ENTRADA
# -----------------------------------------------------------------------------
serie = []
erro_entrada = None

texto_limpo = entrada_texto.strip()
if not texto_limpo:
    erro_entrada = "⚠️ Informe as demandas historicas para continuar."
else:
    partes = [p.strip() for p in texto_limpo.split(",") if p.strip() != ""]
    try:
        serie = [float(p.replace(",", ".")) if "." not in p and p.count(",") else float(p) for p in partes]
    except ValueError:
        erro_entrada = "⚠️ Use apenas numeros separados por virgula (ex.: 120, 125, 130)."

if not erro_entrada:
    if any(v < 0 for v in serie):
        erro_entrada = "⚠️ Foram encontrados valores negativos. A demanda deve ser zero ou positiva."
    elif len(serie) < 4:
        erro_entrada = "⚠️ Informe pelo menos 4 semanas de historico para obter uma previsao confiavel."
    elif not metodos_selecionados:
        erro_entrada = "⚠️ Selecione pelo menos um metodo de previsao."

if len(serie) >= 1 and len(serie) < 8 and not erro_entrada:
    st.warning(
        "ℹ️ Voce informou menos de 8 semanas de historico. "
        "O ideal e usar de 8 a 12 semanas para previsoes mais confiaveis."
    )

if erro_entrada:
    st.error(erro_entrada)
    st.stop()

st.success(f"✅ {len(serie)} semanas de historico carregadas para **{nome_produto}**.")

# -----------------------------------------------------------------------------
# SECAO 2 - SERIE HISTORICA
# -----------------------------------------------------------------------------
st.divider()
st.header("2. Serie historica")

semanas_hist = list(range(1, len(serie) + 1))
df_historico = pd.DataFrame({"Semana": semanas_hist, "Demanda": serie})
st.dataframe(df_historico, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SECAO 3 - CALCULO DAS PREVISOES
# -----------------------------------------------------------------------------
st.divider()
st.header("3. Previsao para as proximas semanas")

mapa_funcoes = {
    "Ingenuo": lambda s, n: metodo_ingenuo(s, n),
    "Media Movel": lambda s, n: metodo_media_movel(s, n, janela=int(janela_media_movel)),
    "Suavizacao Exponencial": lambda s, n: metodo_suavizacao_exponencial(s, n, alfa=alfa_suavizacao),
    "Regressao Linear": lambda s, n: metodo_regressao_linear(s, n),
}

resultados_previsao = {}
for metodo in metodos_selecionados:
    resultados_previsao[metodo] = mapa_funcoes[metodo](serie, int(n_semanas_futuras))

semanas_futuras = list(range(len(serie) + 1, len(serie) + 1 + int(n_semanas_futuras)))
df_previsao = pd.DataFrame({"Semana": semanas_futuras})
for metodo, valores in resultados_previsao.items():
    df_previsao[metodo] = [round(v, 1) for v in valores]

st.dataframe(df_previsao, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SECAO 4 - GRAFICO COMPARATIVO
# -----------------------------------------------------------------------------
st.divider()
st.header("4. Grafico: historico e previsao")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=semanas_hist, y=serie, mode="lines+markers", name="Historico",
    line=dict(color="#1f77b4", width=3),
))

cores_metodos = {
    "Ingenuo": "#888888",
    "Media Movel": "#ff7f0e",
    "Suavizacao Exponencial": "#2ca02c",
    "Regressao Linear": "#d62728",
}

for metodo, valores in resultados_previsao.items():
    fig.add_trace(go.Scatter(
        x=semanas_futuras, y=valores, mode="lines+markers", name=f"Previsao ({metodo})",
        line=dict(color=cores_metodos.get(metodo, "#9467bd"), width=2, dash="dash"),
    ))

fig.update_layout(
    xaxis_title="Semana",
    yaxis_title="Demanda",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SECAO 5 - COMPARACAO DE METODOS (ERRO)
# -----------------------------------------------------------------------------
st.divider()
st.header("5. Comparacao de metodos (erro medio absoluto)")

mae_resultados = calcular_erro_mae(serie)

if mae_resultados is None:
    st.info("ℹ️ Sao necessarias pelo menos 4 semanas de historico para calcular o erro dos metodos.")
else:
    df_mae = pd.DataFrame({
        "Metodo": list(mae_resultados.keys()),
        "Erro Medio Absoluto (MAE)": [round(v, 2) for v in mae_resultados.values()],
    }).sort_values("Erro Medio Absoluto (MAE)")

    st.dataframe(df_mae, use_container_width=True, hide_index=True)

    melhor_metodo = df_mae.iloc[0]["Metodo"]
    st.info(
        f"📌 Com base no historico, o metodo **{melhor_metodo}** apresentou o menor erro medio. "
        "Isso **nao garante** que sera o melhor metodo para as proximas semanas — use como "
        "referencia, nao como certeza."
    )

# -----------------------------------------------------------------------------
# SECAO 6 - RECOMENDACAO GERENCIAL
# -----------------------------------------------------------------------------
st.divider()
st.header("6. Recomendacao Gerencial")

tendencia, variacao_pct, coef_variacao = classificar_tendencia(serie)

col1, col2, col3 = st.columns(3)
col1.metric("Tendencia identificada", tendencia.capitalize())
col2.metric("Variacao 1ª x 2ª metade", f"{variacao_pct:.1f}%")
col3.metric("Coeficiente de variacao", f"{coef_variacao:.1f}%")

if resultados_previsao:
    primeiro_metodo = list(resultados_previsao.keys())[0]
    previsoes_referencia = resultados_previsao[primeiro_metodo]
    st.markdown(gerar_recomendacao_gerencial(serie, previsoes_referencia, tendencia))
else:
    st.warning("Selecione ao menos um metodo para gerar a recomendacao gerencial.")

st.divider()
st.caption(
    "⚠️ Este aplicativo tem finalidade didatica. As previsoes geradas sao estimativas "
    "baseadas em metodos estatisticos simples e nao substituem o julgamento gerencial."
)
