import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import os

# =============================================================================
# 1. CONFIGURACIÓN Y SEGURIDAD (LOGIN SIA)
# =============================================================================
st.set_page_config(page_title="SIA - Sistema de Inteligencia Analítica 2026", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None
if 'user_dep' not in st.session_state:
    st.session_state['user_dep'] = None

# =============================================================================
# 2. INGENIERÍA DE DATOS Y BALANCEO ESTADÍSTICO
# =============================================================================
@st.cache_data
def load_and_transform_data():
    # Metas PAIT 2026
    metas_data = """DEPENDENCIA,Meta_2026
ILM,38367
IRE AMAZONAS,925
IRE ÁNCASH,2000
IRE APURÍMAC,957
IRE AREQUIPA,4450
IRE AYACUCHO,1500
IRE CAJAMARCA,1600
IRE CALLAO,3272
IRE CUSCO,3446
IRE HUANCAVELICA,861
IRE HUÁNUCO,1150
IRE ICA,2570
IRE JUNÍN,3708
IRE LA LIBERTAD,5550
IRE LAMBAYEQUE,4100
IRE LIMA,2450
IRE LORETO,1500
IRE MADRE DE DIOS,620
IRE MOQUEGUA,980
IRE PASCO,750
IRE PIURA,2992
IRE PUNO,1534
IRE SAN MARTÍN,1476
IRE TACNA,1860
IRE TUMBES,1036
IRE UCAYALI,1600
DINI,1967"""
    df_metas = pd.read_csv(io.StringIO(metas_data))

    # Data Histórica 2025 (Cruda)
    data_2025 = """DEPENDENCIA,2025_01,2025_02,2025_03,2025_04,2025_05,2025_06,2025_07,2025_08,2025_09,2025_10,2025_11,2025_12
DINI,108,74,47,76,67,69,100,88,88,85,23,30
ILM,2908,2242,2781,2429,2473,2629,2765,2513,2908,2846,532,385
IRE AMAZONAS,9,19,72,29,34,34,48,51,67,50,13,0
IRE ÁNCASH,85,126,107,169,154,202,163,180,226,167,34,1
IRE APURÍMAC,65,55,52,58,78,65,38,57,75,60,2,5
IRE AREQUIPA,332,272,225,145,260,262,200,439,276,354,126,0
IRE AYACUCHO,71,110,72,135,120,72,116,180,185,157,9,37
IRE CAJAMARCA,83,81,68,153,110,139,117,114,211,191,7,5
IRE CALLAO,155,140,144,175,189,181,128,162,207,191,46,25
IRE CUSCO,300,139,157,208,251,198,290,215,263,230,36,0
IRE HUANCAVELICA,32,26,41,62,59,50,57,39,81,46,11,0
IRE HUÁNUCO,68,69,74,110,90,83,156,50,99,65,18,0
IRE ICA,233,165,111,127,152,175,147,147,244,241,47,0
IRE JUNÍN,231,129,186,187,213,208,298,185,340,328,103,0
IRE LA LIBERTAD,167,207,212,274,281,251,348,328,341,355,85,2
IRE LAMBAYEQUE,204,122,207,221,242,339,606,293,241,1107,124,164
IRE LIMA,259,195,225,239,208,197,135,192,260,196,28,6
IRE LORETO,56,83,67,81,36,42,64,45,88,94,16,5
IRE MADRE DE DIOS,21,32,92,40,52,71,54,26,46,96,54,0
IRE MOQUEGUA,71,41,72,81,62,65,70,59,90,74,16,0
IRE PASCO,43,24,29,31,68,53,56,40,65,49,8,0
IRE PIURA,145,226,146,195,185,208,201,212,197,224,41,0
IRE PUNO,84,11,137,16,159,246,170,65,55,147,29,3
IRE SAN MARTÍN,88,105,114,71,175,86,109,118,199,172,10,0
IRE TACNA,90,56,84,160,131,155,111,104,130,134,14,0
IRE TUMBES,42,25,16,51,68,60,71,52,87,78,13,0
IRE UCAYALI,59,77,76,136,126,153,166,72,172,77,25,0"""
    df_2025 = pd.read_csv(io.StringIO(data_2025))

    # Data Real 2026 (Ene - Abr)
    data_2026 = """DEPENDENCIA,2026_01,2026_02,2026_03,2026_04
DINI,77,43,71,47
ILM,2087,2233,2524,2595
IRE AMAZONAS,51,55,50,43
IRE ÁNCASH,120,218,185,245
IRE APURÍMAC,53,77,112,42
IRE AREQUIPA,234,276,246,287
IRE AYACUCHO,84,172,149,104
IRE CAJAMARCA,129,115,99,64
IRE CALLAO,167,145,194,160
IRE CUSCO,290,195,121,170
IRE HUANCAVELICA,41,26,61,44
IRE HUÁNUCO,53,138,135,59
IRE ICA,123,219,189,179
IRE JUNÍN,220,240,239,241
IRE LA LIBERTAD,117,169,485,321
IRE LAMBAYEQUE,232,295,224,291
IRE LIMA,99,271,238,214
IRE LORETO,47,71,91,84
IRE MADRE DE DIOS,35,80,99,28
IRE MOQUEGUA,16,88,62,58
IRE PASCO,75,62,100,49
IRE PIURA,125,218,234,184
IRE PUNO,60,82,93,142
IRE SAN MARTÍN,159,152,122,75
IRE TACNA,68,201,98,58
IRE TUMBES,25,55,78,143
IRE UCAYALI,64,148,89,83"""
    df_2026 = pd.read_csv(io.StringIO(data_2026))

    df = pd.merge(df_metas, df_2026, on='DEPENDENCIA')
    df = pd.merge(df, df_2025, on='DEPENDENCIA')

    # --- BALANCEO ESTADÍSTICO (CURACIÓN 2025) ---
    meses_sanos = [f'2025_{str(i).zfill(2)}' for i in range(5, 11)]
    df['Mediana_Operativa'] = df[meses_sanos].median(axis=1).astype(int)

    df['2025_11'] = df['Mediana_Operativa']
    df['2025_12'] = df['Mediana_Operativa']

    # --- CÁLCULOS YTD Y PESOS ESTACIONALES ---
    meses_todos_2025 = [f'2025_{str(i).zfill(2)}' for i in range(1, 13)]
    df['Total_2025_Curado'] = df[meses_todos_2025].sum(axis=1)

    # Calcular cuánto debieron hacer cada mes de Ene a Abr (Ideal Estacional)
    for i in range(1, 5):
        mes_str = str(i).zfill(2)
        df[f'Ideal_2026_{mes_str}'] = np.round(df['Meta_2026'] * (df[f'2025_{mes_str}'] / df['Total_2025_Curado']), 0)

    df['Peso_YTD_Estacional'] = df[['2025_01', '2025_02', '2025_03', '2025_04']].sum(axis=1) / df['Total_2025_Curado']

    df['Avance_YTD'] = df['2026_01'] + df['2026_02'] + df['2026_03'] + df['2026_04']
    df['Brecha_Restante'] = df['Meta_2026'] - df['Avance_YTD']
    df['Brecha_Restante'] = df['Brecha_Restante'].apply(lambda x: max(0, x))
    df['Peso_Nac'] = df['Meta_2026'] / df['Meta_2026'].sum()

    # --- FASE PRESCRIPTIVA (GOAL SEEKING CURADO) ---
    meses_restantes = [f'2025_{str(i).zfill(2)}' for i in range(5, 13)]
    df['Suma_Restante_2025'] = df[meses_restantes].sum(axis=1).replace(0, 1)

    for i, mes in enumerate(meses_restantes, start=5):
        mes_2026 = f'2026_{str(i).zfill(2)}'
        df[mes_2026] = np.floor((df[mes] / df['Suma_Restante_2025']) * df['Brecha_Restante']).astype(int)

    df['Suma_Prescriptiva'] = df[[f'2026_{str(i).zfill(2)}' for i in range(5, 13)]].sum(axis=1)
    df['Diferencia_Redondeo'] = df['Brecha_Restante'] - df['Suma_Prescriptiva']
    df['2026_10'] = df['2026_10'] + df['Diferencia_Redondeo']

    # --- KPIs DE RIESGO (SEMÁFORO YTD ESTACIONAL) ---
    df['Cuota_Ideal_Estacional'] = df['Meta_2026'] * df['Peso_YTD_Estacional']
    df['Cumplimiento_Pct'] = np.round((df['Avance_YTD'] / df['Cuota_Ideal_Estacional']) * 100, 1)

    condiciones = [df['Cumplimiento_Pct'] < 85.0, (df['Cumplimiento_Pct'] >= 85.0) & (df['Cumplimiento_Pct'] < 100.0), df['Cumplimiento_Pct'] >= 100.0]
    colores = ['#c0392b', '#f39c12', '#27ae60']
    estados = ['Alto Riesgo', 'Riesgo Medio', 'Óptimo']
    df['Color'] = np.select(condiciones, colores, default='#000000')
    df['Estado'] = np.select(condiciones, estados, default='Sin Clasificar')

    return df

df_master = load_and_transform_data()

# =============================================================================
# 3. PANTALLA DE ACCESO (LOGIN CON DISEÑO INSTITUCIONAL)
# =============================================================================
usuarios_validos = {dep.lower().replace(" ", ""): dep for dep in df_master['DEPENDENCIA']}

def login():
    st.markdown("""
    <style>
    .glass-card {
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-top: 50px;
    }
    .titulo-sia {
        text-align: center;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if os.path.exists("images.png"):
            st.image("images.png", use_container_width=True)

        st.markdown("<h2 class='titulo-sia'>SIA - Inteligencia Analítica 2026</h2>", unsafe_allow_html=True)

        with st.form("login_form"):
            user = st.text_input("Usuario Institucional")
            pwd = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Ingresar al SIA", use_container_width=True)

            if submit:
                if user == "admin" and pwd == "admin2026":
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = 'ADMIN'
                    st.rerun()
                elif user in usuarios_validos and pwd == "1234":
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = 'USER'
                    st.session_state['user_dep'] = usuarios_validos[user]
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas o no autorizadas.")
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state['logged_in']:
    login()
    st.stop()

# =============================================================================
# 4. APLICATIVO PRINCIPAL SIA
# =============================================================================
if st.session_state['user_role'] == 'USER':
    df = df_master[df_master['DEPENDENCIA'] == st.session_state['user_dep']]
else:
    df = df_master

st.sidebar.title(f"SIA 2026 | {st.session_state['user_role']}")
if st.session_state['user_role'] == 'ADMIN':
    menu = st.sidebar.radio("Navegación:", ["📊 Resumen Global SIA", "🚨 Monitor de Riesgos", "📈 Ruta Prescriptiva"])
    dependencia_sel = st.sidebar.selectbox("Filtro por Dependencia:", ["Nacional"] + df['DEPENDENCIA'].tolist())
else:
    menu = st.sidebar.radio("Navegación:", ["🚨 Mi Estatus YTD", "📈 Mi Ruta Prescriptiva"])
    dependencia_sel = st.session_state['user_dep']
    st.sidebar.info(f"Sesión activa: **{dependencia_sel}**")

if st.sidebar.button("Cerrar Sesión SIA"):
    st.session_state['logged_in'] = False
    st.rerun()

# --- VISTA 1: RESUMEN GLOBAL SIA ---
if menu == "📊 Resumen Global SIA":
    st.title("SIA - Sistema de Inteligencia Analítica 2026")
    meta_pait = 93221
    ejecutado = int(df['Avance_YTD'].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Meta Nacional PAIT", f"{meta_pait:,}")
    col2.metric("Avance YTD (Ene-Abr Real)", f"{ejecutado:,}")
    col3.metric("Brecha Operativa Restante", f"{meta_pait - ejecutado:,}")

    st.markdown("### Ranking Estacional de Cumplimiento")
    df_sorted = df.copy()
    color_order = {'#c0392b': 1, '#f39c12': 2, '#27ae60': 3}
    df_sorted['Order'] = df_sorted['Color'].map(color_order)
    df_sorted = df_sorted.sort_values(['Order', 'Cumplimiento_Pct'], ascending=[True, True])

    fig = go.Figure(go.Bar(
        x=df_sorted['Cumplimiento_Pct'], y=df_sorted['DEPENDENCIA'], orientation='h',
        marker_color=df_sorted['Color'], text=df_sorted['Cumplimiento_Pct'].astype(str) + '%', textposition='outside'
    ))
    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0, r=0, t=30, b=0))
    fig.add_vline(x=100, line_dash="solid", line_color="#27ae60", line_width=2)
    st.plotly_chart(fig, use_container_width=True)

# --- VISTA 2: MONITOR DE RIESGOS (PARETO VS INDIVIDUAL) ---
elif menu in ["🚨 Monitor de Riesgos", "🚨 Mi Estatus YTD"]:
    st.title("Monitor Analítico YTD Estacional")

    if dependencia_sel == "Nacional":
        # -------------------------------------------------------------
        # VISTA ADMINISTRADOR NACIONAL: SOLO GRÁFICO PARETO
        # -------------------------------------------------------------
        st.subheader("Dispersión Nacional (Efecto Pareto)")
        fig = go.Figure()
        for estado, color in zip(['Alto Riesgo', 'Riesgo Medio', 'Óptimo'], ['#c0392b', '#f39c12', '#27ae60']):
            df_temp = df[df['Estado'] == estado]
            fig.add_trace(go.Scatter(
                x=df_temp['Peso_Nac']*100, y=df_temp['Cumplimiento_Pct'], mode='markers+text',
                text=df_temp['DEPENDENCIA'].str.replace('IRE ', ''), textposition="top center",
                marker=dict(size=12, color=color, line=dict(width=1, color='white')), name=estado
            ))
        fig.add_hline(y=100, line_dash="dot", line_color="#27ae60", annotation_text="Meta Óptima (100%)")
        fig.update_layout(template="plotly_dark", height=600, xaxis_title="Peso Estructural Nacional (%)", yaxis_title="Cumplimiento YTD (%)")
        st.plotly_chart(fig, use_container_width=True)

    else:
        # -------------------------------------------------------------
        # VISTA INDIVIDUAL (IRE / DINI): TARJETAS + VELOCÍMETRO + BARRAS
        # -------------------------------------------------------------
        datos_dep = df[df['DEPENDENCIA'] == dependencia_sel].iloc[0]
        st.markdown(f"### Análisis de: {dependencia_sel}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h2 style='text-align: center; color: {datos_dep['Color']};'>{datos_dep['Estado']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center;'>{datos_dep['Cumplimiento_Pct']}% YTD</h1>", unsafe_allow_html=True)
        with col2:
            st.info(f"**Target Anual Asignado:** {int(datos_dep['Meta_2026']):,} órdenes.\n\n**Avance Real (Ene-Abr):** {int(datos_dep['Avance_YTD']):,}\n\n**Cuota Ideal Esperada:** {int(datos_dep['Cuota_Ideal_Estacional']):,}")

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Gráfico A: Velocímetro (Corregido: Título perfectamente alineado)
            max_gauge_val = max(150, datos_dep['Cumplimiento_Pct'] + 10)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = datos_dep['Cumplimiento_Pct'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, max_gauge_val], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': datos_dep['Color']},
                    'steps' : [
                        {'range': [0, 85], 'color': "rgba(192, 57, 43, 0.3)"},
                        {'range': [85, 100], 'color': "rgba(243, 156, 18, 0.3)"},
                        {'range': [100, max_gauge_val], 'color': "rgba(39, 174, 96, 0.3)"}],
                }
            ))

            # El título ahora está en el layout general para alinearse milimétricamente con el Gráfico B
            fig_gauge.update_layout(
                title='Velocímetro de Gestión YTD (%)',
                template="plotly_dark",
                height=350,
                margin=dict(t=80, b=20, l=20, r=20) # Mismos márgenes que el gráfico de barras
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            # Gráfico B: Descomposición del Cuatrimestre (Real vs Ideal)
            meses_q1 = ['Ene', 'Feb', 'Mar', 'Abr']
            reales = [datos_dep['2026_01'], datos_dep['2026_02'], datos_dep['2026_03'], datos_dep['2026_04']]
            ideales = [datos_dep['Ideal_2026_01'], datos_dep['Ideal_2026_02'], datos_dep['Ideal_2026_03'], datos_dep['Ideal_2026_04']]

            fig_bar = go.Figure(data=[
                go.Bar(name='Real 2026', x=meses_q1, y=reales, marker_color='#3498db'),
                go.Bar(name='Ideal Estacional', x=meses_q1, y=ideales, marker_color='rgba(255,255,255,0.2)')
            ])
            fig_bar.update_layout(template="plotly_dark", barmode='group', title='Descomposición: Real vs Ideal', height=350, margin=dict(t=80, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

# --- VISTA 3: RUTA PRESCRIPTIVA ---
elif menu in ["📈 Ruta Prescriptiva", "📈 Mi Ruta Prescriptiva"]:
    st.title("Modelado Prescriptivo: Ruta a Diciembre")

    if dependencia_sel == "Nacional":
        st.info("Seleccione una intendencia en el menú lateral.")
    else:
        d = df[df['DEPENDENCIA'] == dependencia_sel].iloc[0]
        meses_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

        val_real = [d['2026_01'], d['2026_02'], d['2026_03'], d['2026_04'], None, None, None, None, None, None, None, None]
        val_goal = [None, None, None, d['2026_04'], d['2026_05'], d['2026_06'], d['2026_07'], d['2026_08'], d['2026_09'], d['2026_10'], d['2026_11'], d['2026_12']]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=meses_labels, y=val_real, mode='lines+markers', name='Real (Descriptivo)', line=dict(color='#3498db', width=4), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=meses_labels, y=val_goal, mode='lines+markers', name='Goal-Seeking (Prescriptivo)', line=dict(color='#27ae60', width=4, dash='dash'), marker=dict(size=8)))
        fig.update_layout(template="plotly_dark", height=500, title=f"Estrategia de Cierre SIA: {dependencia_sel}", yaxis_title="Órdenes Cerradas")
        st.plotly_chart(fig, use_container_width=True)

        if d['Estado'] == 'Alto Riesgo':
            st.error(f"🚨 **Alerta Crítica:** Su cumplimiento estacional es {d['Cumplimiento_Pct']}%. SIA prescribe un incremento operativo inmediato. El modelo le exige {d['2026_10']} órdenes en Octubre para cerrar la brecha matemática.")
        elif d['Estado'] == 'Riesgo Medio':
            st.warning(f"⚠️ **Riesgo Moderado:** Con {d['Cumplimiento_Pct']}%, la meta está cerca. Optimice la producción siguiendo la curva punteada verde mes a mes.")
        else:
            st.success(f"✅ **Desempeño Óptimo:** Ritmo sobresaliente ({d['Cumplimiento_Pct']}%). Mantenga la distribución estacional para superar cómodamente la meta.")