from summit_housing.database import get_db
import pandas as pd
from .sql_fragments import SALES_EVENTS_SQL, ANALYTICS_SQL, MARKET_TRENDS_SQL

class MarketTrendsMixin:
    def get_sales_history(self) -> pd.DataFrame:
        sql = ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL)
        with get_db() as db:
            return db.query(sql)

    def get_market_trends(self, exclude_multiunit: bool = False) -> pd.DataFrame:
        filter_clause = "AND (units IS NULL OR units <= 1)" if exclude_multiunit else ""
        full_sql = MARKET_TRENDS_SQL.format(
            analytics_query=ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL),
            filter_clause=filter_clause
        )
        with get_db() as db:
            return db.query(full_sql)

    def get_top_flippers(self) -> pd.DataFrame:
        sql = """
        WITH metrics AS ({analytics_query})
        SELECT * FROM metrics 
        WHERE days_held IS NOT NULL 
          AND days_held < 365 * 2 
          AND growth_pct > 20     
        ORDER BY growth_pct DESC
        LIMIT 50
        """.format(analytics_query=ANALYTICS_SQL.format(sales_events_sql=SALES_EVENTS_SQL))
        
        with get_db() as db:
            return db.query(sql)

    def get_price_by_type(self) -> pd.DataFrame:
        sql = """
        WITH classified AS (
            SELECT 
                schno,
                CASE 
                    WHEN abst1 = '1212' OR abst2 = '1212' THEN 'Single Family'
                    WHEN abst1 IN ('1230', '1231') OR abst2 IN ('1230', '1231') THEN 'Condo'
                    WHEN abst1 = '1214' OR abst2 = '1214' THEN 'Townhouse'
                    WHEN abst1 IN ('100', '200') AND (abst2 IS NULL OR abst2 = '0') THEN 'Vacant Land'
                    ELSE 'Other'
                END as prop_type
            FROM raw_records
        ),
        sales AS (
            {sales_cte}
        )
        SELECT 
            strftime('%Y', s.tx_date) as tx_year,
            c.prop_type,
            AVG(s.estimated_price) as avg_price,
            COUNT(*) as sales_count
        FROM sales s
        JOIN classified c ON s.schno = c.schno
        WHERE tx_year >= '1980'
        GROUP BY 1, 2
        HAVING sales_count > 5
        ORDER BY tx_year, prop_type
        """.format(sales_cte=SALES_EVENTS_SQL)
        
        with get_db() as db:
            return db.query(sql)

class InventoryMixin:
    def get_cumulative_supply(self) -> pd.DataFrame:
        sql = """
        WITH yearly_builds AS (
            SELECT 
                year_blt,
                SUM(units) as units_built,
                SUM(sfla) as sqft_built
            FROM raw_records
            WHERE year_blt IS NOT NULL 
              AND year_blt > 1900
              AND sfla > 0
              AND (
                  (CAST(abst1 AS INTEGER) BETWEEN 1000 AND 1999) OR 
                  (abst1 IS NULL AND sfla > 0)
              )
            GROUP BY year_blt
        )
        SELECT 
            year_blt,
            units_built,
            SUM(units_built) OVER (ORDER BY year_blt ROWS UNBOUNDED PRECEDING) as cumulative_units,
            SUM(sqft_built) OVER (ORDER BY year_blt ROWS UNBOUNDED PRECEDING) as cumulative_sqft
        FROM yearly_builds
        ORDER BY year_blt
        """
        with get_db() as db:
            return db.query(sql)

    def get_owner_location_stats(self) -> pd.DataFrame:
        sql = """
        SELECT 
            CASE 
                WHEN city IN ('BRECKENRIDGE', 'FRISCO', 'DILLON', 'SILVERTHORNE', 'KEYSTONE', 'COPPER MOUNTAIN') THEN 'Local (In-County)'
                WHEN state = 'CO' THEN 'In-State (Non-Local)'
                ELSE 'Out-of-State'
            END as location_type,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM raw_records
        GROUP BY 1
        ORDER BY count DESC
        """
        with get_db() as db:
            return db.query(sql)

    def get_inventory_profile(self) -> pd.DataFrame:
        sql = """
        SELECT 
            CASE 
                WHEN city IN ('BRECKENRIDGE', 'FRISCO', 'DILLON', 'SILVERTHORNE', 'KEYSTONE', 'COPPER MOUNTAIN') THEN 'Local (In-County)'
                WHEN state = 'CO' THEN 'In-State (Non-Local)'
                ELSE 'Out-of-State'
            END as location_type,
            COUNT(*) as properties_owned,
            AVG(sfla) as avg_sqft,
            AVG(CASE 
                WHEN docfee1 < 10000 THEN docfee1 * 10000 
                ELSE docfee1 
            END) as avg_value,
            AVG(CASE 
                WHEN sfla > 100 THEN (CASE WHEN docfee1 < 10000 THEN docfee1 * 10000 ELSE docfee1 END) / sfla 
                ELSE NULL 
            END) as avg_ppsf
        FROM raw_records
        WHERE sfla > 0 AND docfee1 > 0
        GROUP BY 1
        ORDER BY avg_value DESC
        """
        with get_db() as db:
            return db.query(sql)

    def get_owner_purchase_trends(self) -> pd.DataFrame:
        sql = """
        SELECT 
            strftime('%Y', recdate1) as purchase_year,
            CASE 
                WHEN city IN ('BRECKENRIDGE', 'FRISCO', 'DILLON', 'SILVERTHORNE', 'KEYSTONE', 'COPPER MOUNTAIN') THEN 'Local (In-County)'
                WHEN state = 'CO' THEN 'In-State (Non-Local)'
                ELSE 'Out-of-State'
            END as location_type,
            COUNT(*) as buyer_count,
            AVG(CASE 
                WHEN docfee1 < 10000 THEN docfee1 * 10000 
                ELSE docfee1 
            END) as avg_purchase_price
        FROM raw_records
        WHERE purchase_year >= '1980' AND purchase_year IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1
        """
        with get_db() as db:
            return db.query(sql)

class MLDataMixin:
    def get_training_data(self) -> pd.DataFrame:
        sql = """
        WITH clean_sales AS (
            {sales_cte}
        )
        SELECT 
            s.schno,
            s.tx_date,
            s.estimated_price as price,
            r.beds,
            r.bath_tot as baths,
            r.sfla,
            r.year_blt,
            r.garage_size,
            r.lot_size as acres,
            r.grade,
            r.cond,
            r.scenic_view,
            CASE 
                WHEN r.town = 'B' THEN 'BRECKENRIDGE'
                WHEN r.town = 'F' THEN 'FRISCO'
                WHEN r.town = 'D' THEN 'DILLON'
                WHEN r.town = 'S' THEN 'SILVERTHORNE'
                WHEN r.town = 'K' THEN 'KEYSTONE'
                WHEN r.town = 'C' THEN 'COPPERMOUNTAIN'
                WHEN r.town = 'BR' THEN 'BLUERIVER'
                ELSE 'OTHER'
            END as city,
            CASE 
                WHEN r.abst1 = '1212' OR r.abst2 = '1212' THEN 'Single Family'
                WHEN r.abst1 IN ('1230', '1231') OR r.abst2 IN ('1230', '1231') THEN 'Condo'
                WHEN r.abst1 = '1214' OR r.abst2 = '1214' THEN 'Townhouse'
                ELSE 'Other'
            END as prop_type
        FROM clean_sales s
        JOIN raw_records r ON s.schno = r.schno
        WHERE s.estimated_price > 50000 
          AND r.sfla > 100 
          AND strftime('%Y', s.tx_date) > '1995' 
        ORDER BY s.tx_date ASC
        """.format(sales_cte=SALES_EVENTS_SQL)
        
        with get_db() as db:
            return db.query(sql)

    def get_comparable_sales(self, target_city: str, target_type: str, target_sfla: int) -> pd.DataFrame:
        type_filter = ""
        if target_type == "Single Family":
            type_filter = "AND (r.abst1 = '1212' OR r.abst2 = '1212')"
        elif target_type == "Condo":
            type_filter = "AND (r.abst1 IN ('1230', '1231') OR r.abst2 IN ('1230', '1231'))"
        elif target_type == "Townhouse":
            type_filter = "AND (r.abst1 = '1214' OR r.abst2 = '1214')"
        
        city_map = {
            'BRECKENRIDGE': 'B', 'FRISCO': 'F', 'DILLON': 'D', 
            'SILVERTHORNE': 'S', 'KEYSTONE': 'K', 'COPPERMOUNTAIN': 'C',
            'BLUERIVER': 'BR'
        }
        town_code = city_map.get(target_city)
        city_clause = f"AND r.town = '{town_code}'" if town_code else ""
        
        sql = """
        WITH recent_sales AS (
            {sales_cte}
        )
        SELECT 
            s.tx_date,
            s.estimated_price as price,
            r.sfla,
            r.beds,
            r.bath_tot as baths,
            r.year_blt,
            r.address,
            (s.estimated_price * 1.0 / r.sfla) as ppsf
        FROM recent_sales s
        JOIN raw_records r ON s.schno = r.schno
        WHERE s.estimated_price > 50000 
          AND s.tx_date >= date('now', '-18 months')
          AND r.sfla BETWEEN ? AND ? 
          {city_clause}
          {type_filter}
        ORDER BY s.tx_date DESC
        LIMIT 50
        """.format(
            sales_cte=SALES_EVENTS_SQL, 
            city_clause=city_clause,
            type_filter=type_filter
        )
        
        with get_db() as db:
            return db.query(sql, (int(target_sfla * 0.75), int(target_sfla * 1.25)))
