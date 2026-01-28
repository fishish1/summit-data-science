from .analytics import MarketTrendsMixin, InventoryMixin, MLDataMixin
from .sql_fragments import SALES_EVENTS_SQL, ANALYTICS_SQL
from summit_housing.database import get_db
import pandas as pd

class MarketAnalytics(MarketTrendsMixin, InventoryMixin, MLDataMixin):
    """
    Unified entry point for housing market analytics.
    Combines SQL-heavy queries with python-side analysis.
    """
    
    def get_dataset_sample(self, limit: int = 1000) -> pd.DataFrame:
        sql = f"WITH metrics AS ({ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL)}) SELECT * FROM metrics LIMIT ?"
        with get_db() as db:
            return db.query(sql, (limit,))

    def get_seasonality_stats(self) -> pd.DataFrame:
        sql = """
        WITH metrics AS ({analytics_query})
        SELECT 
            strftime('%m', tx_date) as month_num,
            CASE strftime('%m', tx_date)
                WHEN '01' THEN 'Jan' WHEN '02' THEN 'Feb' WHEN '03' THEN 'Mar'
                WHEN '04' THEN 'Apr' WHEN '05' THEN 'May' WHEN '06' THEN 'Jun'
                WHEN '07' THEN 'Jul' WHEN '08' THEN 'Aug' WHEN '09' THEN 'Sep'
                WHEN '10' THEN 'Oct' WHEN '11' THEN 'Nov' WHEN '12' THEN 'Dec'
            END as month_name,
            city,
            COUNT(*) as sales_count,
            AVG(CASE WHEN sfla > 100 THEN estimated_price / sfla ELSE NULL END) as avg_ppsf,
            AVG(estimated_price) as avg_price
        FROM metrics
        WHERE tx_year > '2010'
        GROUP BY 1, 2, 3
        ORDER BY month_num
        """.format(analytics_query=ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL))
        
        with get_db() as db:
            return db.query(sql)

    def get_renovation_impact(self) -> pd.DataFrame:
        sql = """
        WITH metrics AS ({analytics_query})
        SELECT 
            tx_year,
            CASE WHEN is_renovated = 1 THEN 'Renovated' ELSE 'Original Condition' END as status,
            COUNT(*) as sales_count,
            AVG(estimated_price) as avg_price,
            AVG(CASE WHEN sfla > 100 THEN estimated_price / sfla ELSE NULL END) as avg_ppsf,
            AVG(growth_pct) as avg_appreciation
        FROM metrics
        WHERE tx_year > '2000'
        GROUP BY 1, 2
        ORDER BY tx_year, status
        """.format(analytics_query=ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL))
        
        with get_db() as db:
            return db.query(sql)

    # Note: Adding remaining complex methods to this class or additional mixins as needed
    def get_cumulative_supply_by_type(self) -> pd.DataFrame:
        sql = """
        WITH classified AS (
            SELECT 
                year_blt, units, sfla,
                CASE 
                    WHEN abst1 = '1212' OR abst2 = '1212' THEN 'Single Family'
                    WHEN abst1 IN ('1230', '1231') OR abst2 IN ('1230', '1231') THEN 'Condo'
                    WHEN abst1 = '1214' OR abst2 = '1214' THEN 'Townhouse'
                    ELSE 'Other'
                END as prop_type
            FROM raw_records
            WHERE year_blt IS NOT NULL AND year_blt > 1900 AND sfla > 0
        ),
        yearly AS (
            SELECT year_blt, prop_type, SUM(units) as new_units FROM classified GROUP BY 1, 2
        )
        SELECT 
            year_blt, prop_type, new_units,
            SUM(new_units) OVER (PARTITION BY prop_type ORDER BY year_blt ROWS UNBOUNDED PRECEDING) as cumulative_units
        FROM yearly
        ORDER BY year_blt, prop_type
        """
        with get_db() as db:
            df = db.query(sql)
        if df.empty: return df
        min_year, max_year = int(df['year_blt'].min()), int(df['year_blt'].max())
        all_years = range(min_year, max_year + 1)
        all_types = df['prop_type'].unique()
        full_idx = pd.MultiIndex.from_product([all_years, all_types], names=['year_blt', 'prop_type'])
        df_dense = df.set_index(['year_blt', 'prop_type']).reindex(full_idx)
        df_dense['cumulative_units'] = df_dense.groupby(level='prop_type')['cumulative_units'].ffill().fillna(0)
        df_dense['new_units'] = df_dense['new_units'].fillna(0)
        return df_dense.reset_index()

    def get_price_band_evolution(self) -> pd.DataFrame:
        sql = "SELECT strftime('%Y', tx_date) as tx_year, estimated_price FROM ({analytics_query}) WHERE tx_year > '1990'".format(analytics_query=ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL))
        with get_db() as db:
            df = db.query(sql)
        if df.empty: return pd.DataFrame(columns=['tx_year', 'price_band', 'sales_count'])
        yearly_medians = df.groupby('tx_year')['estimated_price'].median().reset_index(name='annual_median')
        df = df.merge(yearly_medians, on='tx_year')
        df['pct_of_median'] = (df['estimated_price'] / df['annual_median']) * 100
        def classify_band(pct):
            if pct < 50: return '1. Budget (<50% Med)'
            elif pct <= 100: return '2. Below Med (50-100%)'
            elif pct <= 150: return '3. Above Med (100-150%)'
            elif pct <= 200: return '4. Premium (150-200%)'
            else: return '5. Luxury (>200% Med)'
        df['price_band'] = df['pct_of_median'].apply(classify_band)
        return df.groupby(['tx_year', 'price_band']).size().reset_index(name='sales_count')
