import pandas as pd
from sqlalchemy import create_engine


db_connection_str = 'postgresql://postgres:tupassword@localhost:5432/analistap'
db_connection = create_engine(db_connection_str)

try:
    # 2. La consulta SQL "Inteligente"
    # En lugar de traer todo, traemos solo lo necesario para el análisis
    query = """
    SELECT 
        o.order_purchase_timestamp as fecha,
        i.price as precio,
        p.product_category_name as categoria
    FROM olist_items i
    JOIN olist_orders o ON i.order_id = o.order_id
    JOIN olist_products p ON i.product_id = p.product_id
    WHERE o.order_status = 'delivered'
    """

    # 3. Cargar datos en Pandas
    print(" Consultando base de datos...")
    df = pd.read_sql(query, db_connection)
    
    print(f"Conexión exitosa! Se descargaron {len(df)} registros.")
    print(df.head())

    # 4. Pequeño análisis rápido para verificar
    df['fecha'] = pd.to_datetime(df['fecha'])
    print("\n--- Resumen de Ventas ---")
    print(df.groupby(df['fecha'].dt.to_period('M'))['precio'].sum().tail())

except Exception as e:
    print("❌ Error de conexión:", e)