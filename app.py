import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# CONFIGURACIÓN DE PÁGINA (Layout Wide)
st.set_page_config(page_title="E-Commerce Intelligence Suite", layout="wide")


@st.cache_data
def load_data():
    
    db_connection_str = 'postgresql://postgres:passw@localhost:5432/analistap'
    db_connection = create_engine(db_connection_str)
    
    # Query 1: Ventas Temporales (Para Forecasting)
    q_sales = """
    SELECT DATE(o.order_purchase_timestamp) as fecha, SUM(i.price) as ventas
    FROM olist_items i JOIN olist_orders o ON i.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY fecha ORDER BY fecha;
    """
    
    # Query 2: RFM (Para Segmentación de Clientes - ¡Nivel Avanzado!)
    # Analizamos: Recency (Días desde última compra), Frequency (Total compras), Monetary (Gasto total)
    q_rfm = """
    SELECT 
        o.customer_id,
        MAX(o.order_purchase_timestamp) as ultima_compra,
        COUNT(DISTINCT o.order_id) as frecuencia,
        SUM(i.price) as monto_total
    FROM olist_items i JOIN olist_orders o ON i.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.customer_id
    HAVING SUM(i.price) > 0
    LIMIT 5000; 
    """ 
    # Limitamos a 5000 para que tu PC no explote en vivo, en prod se quita el limit.

    df_sales = pd.read_sql(q_sales, db_connection)
    df_rfm = pd.read_sql(q_rfm, db_connection)
    
    return df_sales, df_rfm

# Carga de datos
try:
    with st.spinner('Conectando al Data Warehouse...'):
        df_sales, df_rfm = load_data()
        df_sales['fecha'] = pd.to_datetime(df_sales['fecha'])
except Exception as e:
    st.error(f"Error de conexión: {e}")
    st.stop()


st.title("E-Commerce Commerce Intelligence Suite")
st.markdown("Plataforma de análisis predictivo y segmentación de clientes.")

# KPIs TOP (Metrics Row)
col1, col2, col3, col4 = st.columns(4)
total_revenue = df_sales['ventas'].sum()
avg_ticket = df_rfm['monto_total'].mean()
last_day_sales = df_sales.iloc[-1]['ventas']

col1.metric("Ingresos Totales", f"${total_revenue:,.0f} BRL")
col2.metric("Ticket Promedio", f"${avg_ticket:.2f} BRL")
col3.metric("Ventas Último Día", f"${last_day_sales:.2f} BRL", delta="-12%")
col4.metric("Clientes Analizados", f"{len(df_rfm)}")

st.divider()

# PESTAÑAS PARA ORGANIZAR LA COMPLEJIDAD
tab1, tab2, tab3 = st.tabs([" Predicción de Ventas ", "👥 Segmentación de Clientes (Clustering)", "📊 Business Intelligence"])

with tab1:
    st.subheader("Modelo de Predicción de Demanda")
    
    # Preparación rápida de datos
    df_ts = df_sales.set_index('fecha').asfreq('D').fillna(0)
    df_ts['lag_1'] = df_ts['ventas'].shift(1)
    df_ts['lag_7'] = df_ts['ventas'].shift(7)
    df_ts['rolling_mean'] = df_ts['ventas'].rolling(7).mean()
    df_ts = df_ts.dropna()
    
    # Modelo
    train_size = int(len(df_ts) * 0.85)
    train, test = df_ts.iloc[:train_size], df_ts.iloc[train_size:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[['lag_1', 'lag_7', 'rolling_mean']], train['ventas'])
    preds = model.predict(test[['lag_1', 'lag_7', 'rolling_mean']])
    
    # Gráfico Interactivo con Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['ventas'], name='Histórico', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=test.index, y=test['ventas'], name='Realidad', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=preds, name='Predicción AI', line=dict(color='orange', dash='dash')))
    
    fig.update_layout(title="Forecast de Ventas (Random Forest)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("Análisis de Clusters de Clientes (K-Means)")
    st.info("Este módulo utiliza Aprendizaje No Supervisado para detectar patrones de comportamiento.")

    # Preprocesamiento para K-Means
    scaler = StandardScaler()
    # Usamos Frecuencia y Monto para agrupar
    X = df_rfm[['frecuencia', 'monto_total']]
    X_scaled = scaler.fit_transform(X)
    
    # Slider para elegir número de clusters
    k = st.slider("Número de Segmentos (Clusters)", 2, 6, 3)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_rfm['cluster'] = kmeans.fit_predict(X_scaled)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Gráfico de dispersión de Clusters
        fig_cluster = px.scatter(
            df_rfm, x='frecuencia', y='monto_total', color=df_rfm['cluster'].astype(str),
            title="Mapa de Segmentación de Clientes",
            labels={'cluster': 'Segmento', 'monto_total': 'Gasto Total', 'frecuencia': 'Compras'},
            template="plotly_dark", log_y=True # Log scale porque hay clientes que gastan mucho
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    with col_right:
        st.write("### Perfil de los Segmentos")
        # Explicación de los grupos
        profile = df_rfm.groupby('cluster')[['monto_total', 'frecuencia']].mean().sort_values('monto_total', ascending=False)
        st.dataframe(profile.style.highlight_max(axis=0))
        st.caption("Nota: El Cluster con mayor 'monto_total' son tus clientes VIP.")


with tab3:
    st.subheader("Exploración de Datos Crudos")
    st.dataframe(df_sales.sort_values('fecha', ascending=False).head(100), use_container_width=True)
    
    # Descargar reporte
    csv = df_sales.to_csv(index=False).encode('utf-8')
    st.download_button(" Descargar Reporte CSV", csv, "reporte_ventas.csv", "text/csv")