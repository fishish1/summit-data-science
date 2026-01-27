from summit_housing.database import get_db
import pandas as pd

# --- SQL Logic ---

# 1. Unpivot Sales Events CTE
# Combines the 4 slots into a normalized stream of sales.
SALES_EVENTS_SQL = """
WITH raw_sales AS (
    SELECT schno, recdate1 as tx_date, docfee1 as doc_fee, '1' as slot FROM raw_records
    UNION ALL
    SELECT schno, recdate2 as tx_date, docfee2 as doc_fee, '2' as slot FROM raw_records
    UNION ALL
    SELECT schno, recdate3 as tx_date, docfee3 as doc_fee, '3' as slot FROM raw_records
    UNION ALL
    SELECT schno, recdate4 as tx_date, docfee4 as doc_fee, '4' as slot FROM raw_records
),
clean_sales AS (
    SELECT 
        schno,
        tx_date,
        doc_fee,
        -- Calculate Estimated Price: 
        -- If doc_fee is small (< 10,000), assume it's a fee (0.01% of price).
        -- If doc_fee is large (>= 10,000), assume it's already the raw price.
        CASE 
            WHEN doc_fee < 10000 THEN CAST(doc_fee * 10000 AS INTEGER)
            ELSE CAST(doc_fee AS INTEGER)
        END as estimated_price,
        slot
    FROM raw_sales
    WHERE tx_date IS NOT NULL 
      AND doc_fee > 0 -- Only market sales
)
SELECT * FROM clean_sales
"""

# 2. Market Analytics Query
# Uses Window Functions to calculate trends
ANALYTICS_SQL = f"""
WITH sales AS (
    {SALES_EVENTS_SQL}
),
sales_with_metrics AS (
    SELECT 
        s.schno,
        s.tx_date,
        s.estimated_price,
        
        -- Window Function 1: Price Appreciation
        -- Calculate the price of the PREVIOUS sale for this property
        LAG(s.estimated_price) OVER (PARTITION BY s.schno ORDER BY s.tx_date) as prev_price,
        
        -- Window Function 2: Holding Period
        -- Calculate days since previous sale
        julianday(s.tx_date) - julianday(LAG(s.tx_date) OVER (PARTITION BY s.schno ORDER BY s.tx_date)) as days_held,
        
        -- Window Function 3: Sales Velocity
        -- How many times has this property sold?
        COUNT(*) OVER (PARTITION BY s.schno) as times_sold,
        
        -- Map Town Code to Real Name
        CASE 
            WHEN r.town = 'B' THEN 'BRECKENRIDGE'
            WHEN r.town = 'F' THEN 'FRISCO'
            WHEN r.town = 'D' THEN 'DILLON'
            WHEN r.town = 'S' THEN 'SILVERTHORNE'
            WHEN r.town = 'K' THEN 'KEYSTONE'
            WHEN r.town = 'C' THEN 'COPPER/COUNTY'
            WHEN r.town = 'M' THEN 'MONTEZUMA'
            WHEN r.town = 'R' THEN 'RURAL'
            WHEN r.town = 'BR' THEN 'BLUE RIVER'
            WHEN r.town = 'H' THEN 'HEENEY'
            ELSE 'OTHER/COUNTY'
        END as city,
        r.sfla,
        -- Renovation Logic: If Effective Year > Actual Year
        CASE WHEN r.adj_year_blt > r.year_blt THEN 1 ELSE 0 END as is_renovated,
        r.units,
        r.address,
        -- Quality Features (The Kitchen Sink)
        r.grade,
        r.cond,
        r.scenic_view
    FROM sales s
    JOIN raw_records r ON s.schno = r.schno
)
SELECT 
    *,
    -- Calculate Growth %
    ROUND((estimated_price - prev_price) * 100.0 / prev_price, 2) as growth_pct,
    
    -- Extract Year for aggregation
    strftime('%Y', tx_date) as tx_year
FROM sales_with_metrics
ORDER BY tx_date DESC
"""

# 3. Aggregated Trends (Moving Averages)
MARKET_TRENDS_SQL = """
WITH metrics AS (
    {analytics_query}
),
yearly_stats AS (
    SELECT 
        tx_year,
        city,
        AVG(estimated_price) as avg_price,
        -- Calculate PPSF (Filter out bad data)
        AVG(CASE WHEN sfla > 100 THEN estimated_price / sfla ELSE NULL END) as avg_ppsf,
        COUNT(*) as sales_count
    FROM metrics
    WHERE tx_year IS NOT NULL AND tx_year > '1990'
      {filter_clause}
    GROUP BY tx_year, city
)
SELECT 
    tx_year,
    city,
    avg_price,
    avg_ppsf,
    sales_count,
    
    -- Window Function 4: Moving Average (Price)
    AVG(avg_price) OVER (
        PARTITION BY city 
        ORDER BY tx_year 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as avg_price_3yr_ma,
    
    -- Window Function 5: Moving Average (PPSF)
    AVG(avg_ppsf) OVER (
        PARTITION BY city 
        ORDER BY tx_year 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as avg_ppsf_3yr_ma
FROM yearly_stats
ORDER BY city, tx_year
"""

class MarketAnalytics:
    def get_sales_history(self) -> pd.DataFrame:
        """Returns the raw unpivoted sales history with metrics."""
        with get_db() as db:
            # We format the string to include the base CTE
            return db.query(ANALYTICS_SQL)

    def get_market_trends(self, exclude_multiunit: bool = False) -> pd.DataFrame:
        """Returns aggregated market trends with moving averages."""
        with get_db() as db:
            # Inject the base query into the trend query
            # We need to construct it carefully to avoid f-string mess
            # Simplest way: Define the full CTEs in SQL or compose strings
            
            filter_clause = "AND (units IS NULL OR units <= 1)" if exclude_multiunit else ""
            
            # Composing logic:
            full_sql = MARKET_TRENDS_SQL.format(
                analytics_query=ANALYTICS_SQL,
                filter_clause=filter_clause
            )
            return db.query(full_sql)

    def get_top_flippers(self) -> pd.DataFrame:
        """Returns properties with short holding periods and high appreciation."""
        sql = """
        WITH metrics AS ({analytics_query})
        SELECT * FROM metrics 
        WHERE days_held IS NOT NULL 
          AND days_held < 365 * 2 -- Sold within 2 years
          AND growth_pct > 20     -- > 20% profit
        ORDER BY growth_pct DESC
        LIMIT 50
        """.format(analytics_query=ANALYTICS_SQL)
        
        with get_db() as db:
            return db.query(sql)

    def get_dataset_sample(self, limit: int = 1000) -> pd.DataFrame:
        """Returns a sample of the raw records joined with sales metrics."""
        sql = f"""
        WITH metrics AS ({ANALYTICS_SQL})
        SELECT * FROM metrics
        LIMIT ?
        """
        with get_db() as db:
            return db.query(sql, (limit,))

    def get_cumulative_supply(self) -> pd.DataFrame:
        """
        Returns the cumulative number of units built over time.
        Demonstrates Window Function: SUM() OVER (ORDER BY year ROWS UNBOUNDED PRECEDING)
        """
        sql = """
        WITH yearly_builds AS (
            SELECT 
                year_blt,
                SUM(units) as units_built,
                SUM(sfla) as sqft_built
            FROM raw_records
            WHERE year_blt IS NOT NULL 
              AND year_blt > 1900
              AND sfla > 0 -- Must have living area
              -- Trick: Limit to Residential Abstract Codes (1000-1999)
              -- Common: 1212 (SFR), 1230 (Condo), 1215 (Duplex)
              AND (
                  (CAST(abst1 AS INTEGER) BETWEEN 1000 AND 1999) OR 
                  (abst1 IS NULL AND sfla > 0) -- Include if uncoded but has living area
              )
            GROUP BY year_blt
        )
        SELECT 
            year_blt,
            units_built,
            -- Window Function: Running Total
            SUM(units_built) OVER (ORDER BY year_blt ROWS UNBOUNDED PRECEDING) as cumulative_units,
            SUM(sqft_built) OVER (ORDER BY year_blt ROWS UNBOUNDED PRECEDING) as cumulative_sqft
        FROM yearly_builds
        ORDER BY year_blt
        """
        with get_db() as db:
            return db.query(sql)

    def get_owner_location_stats(self) -> pd.DataFrame:
        """
        Analyzes percentage of In-County vs Out-of-State owners.
        Demonstrates CASE Logic.
        """
        sql = """
        SELECT 
            CASE 
                -- If Owner City maps to one of our towns, they are local
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

    def get_cumulative_supply_by_type(self) -> pd.DataFrame:
        """
        Returns cumulative units built over time, segmented by Property Type.
        Useful for Stacked Area Charts.
        Includes Python-side densification to ensure straight lines (no dips) for years with 0 construction.
        """
        sql = """
        WITH classified AS (
            SELECT 
                year_blt,
                units,
                sfla,
                CASE 
                    WHEN abst1 = '1212' OR abst2 = '1212' THEN 'Single Family'
                    WHEN abst1 IN ('1230', '1231') OR abst2 IN ('1230', '1231') THEN 'Condo'
                    WHEN abst1 = '1214' OR abst2 = '1214' THEN 'Townhouse'
                    ELSE 'Other'
                END as prop_type
            FROM raw_records
            WHERE year_blt IS NOT NULL 
              AND year_blt > 1900
              AND sfla > 0
        ),
        yearly AS (
            SELECT 
                year_blt,
                prop_type,
                SUM(units) as new_units
            FROM classified
            GROUP BY 1, 2
        )
        SELECT 
            year_blt,
            prop_type,
            new_units,
            SUM(new_units) OVER (PARTITION BY prop_type ORDER BY year_blt ROWS UNBOUNDED PRECEDING) as cumulative_units
        FROM yearly
        ORDER BY year_blt, prop_type
        """
        with get_db() as db:
            df = db.query(sql)
            
        if df.empty:
            return df
            
        # Densify Data: Ensure every (Year, Type) exists
        min_year = int(df['year_blt'].min())
        max_year = int(df['year_blt'].max())
        all_years = range(min_year, max_year + 1)
        all_types = df['prop_type'].unique()
        
        # Create MultiIndex of all possible combinations
        full_idx = pd.MultiIndex.from_product([all_years, all_types], names=['year_blt', 'prop_type'])
        
        # Reindex
        df_dense = df.set_index(['year_blt', 'prop_type']).reindex(full_idx)
        
        # Forward Fill cumulative units (carry over previous total) and fill NaN with 0
        df_dense['cumulative_units'] = df_dense.groupby(level='prop_type')['cumulative_units'].ffill().fillna(0)
        df_dense['new_units'] = df_dense['new_units'].fillna(0)
        
        return df_dense.reset_index()

    def get_price_band_evolution(self) -> pd.DataFrame:
        """
        Analyzes market composition by price tiers relative to the annual MEDIAN.
        Buckets: <50%, 50-100%, 100-150%, 150-200%, >200% of that year's median price.
        Moved logic to Pandas to support Median calculation (unavailable in standard SQLite).
        """
        sql = """
        SELECT 
            strftime('%Y', tx_date) as tx_year,
            estimated_price
        FROM ({analytics_query})
        WHERE tx_year > '1990'
        """.format(analytics_query=ANALYTICS_SQL)
        
        with get_db() as db:
            df = db.query(sql)
            
        if df.empty:
            return pd.DataFrame(columns=['tx_year', 'price_band', 'sales_count'])
            
        # Calculate Annual Median
        yearly_medians = df.groupby('tx_year')['estimated_price'].median().reset_index()
        yearly_medians.rename(columns={'estimated_price': 'annual_median'}, inplace=True)
        
        # Merge
        df = df.merge(yearly_medians, on='tx_year')
        
        # Calculate % of Median
        df['pct_of_median'] = (df['estimated_price'] / df['annual_median']) * 100
        
        # Bucket
        def classify_band(pct):
            if pct < 50: return '1. Budget (<50% Med)'
            elif pct <= 100: return '2. Below Med (50-100%)'
            elif pct <= 150: return '3. Above Med (100-150%)'
            elif pct <= 200: return '4. Premium (150-200%)'
            else: return '5. Luxury (>200% Med)'
            
        df['price_band'] = df['pct_of_median'].apply(classify_band)
        
        # Aggregate
        stats = df.groupby(['tx_year', 'price_band']).size().reset_index(name='sales_count')
        
        return stats

    def get_training_data(self) -> pd.DataFrame:
        """
        Fetches clean sales data for Machine Learning.
        Returns: [tx_date, price, beds, baths, sfla, year_blt, garage_size, city, prop_type]
        """
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
            -- Quality Features
            r.grade,
            r.cond,
            r.scenic_view,
            r.town as city_code, -- Use code if needed, but case below handles names
            -- Use CASE for Clean City
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
            -- Prop Type
            CASE 
                WHEN r.abst1 = '1212' OR r.abst2 = '1212' THEN 'Single Family'
                WHEN r.abst1 IN ('1230', '1231') OR r.abst2 IN ('1230', '1231') THEN 'Condo'
                WHEN r.abst1 = '1214' OR r.abst2 = '1214' THEN 'Townhouse'
                ELSE 'Other'
            END as prop_type
        FROM clean_sales s
        JOIN raw_records r ON s.schno = r.schno
        WHERE s.estimated_price > 50000 -- Filter minimal sales
          AND r.sfla > 100 -- Filter bad data
          AND strftime('%Y', s.tx_date) > '1995' -- Modern Era
        ORDER BY s.tx_date ASC
        """.format(sales_cte=SALES_EVENTS_SQL)
        
        with get_db() as db:
            return db.query(sql)

    def get_inventory_profile(self) -> pd.DataFrame:
        """
        Analyzes the profile of current owners (Inventory Analysis).
        Compares Locals vs Out-of-State on Home Size and Value.
        """
        sql = """
        SELECT 
            CASE 
                WHEN city IN ('BRECKENRIDGE', 'FRISCO', 'DILLON', 'SILVERTHORNE', 'KEYSTONE', 'COPPER MOUNTAIN') THEN 'Local (In-County)'
                WHEN state = 'CO' THEN 'In-State (Non-Local)'
                ELSE 'Out-of-State'
            END as location_type,
            COUNT(*) as properties_owned,
            AVG(sfla) as avg_sqft,
            -- Smart Price Logic for Current Value (based on last sale docfee1)
            AVG(CASE 
                WHEN docfee1 < 10000 THEN docfee1 * 10000 
                ELSE docfee1 
            END) as avg_value,
            AVG(CASE 
                WHEN sfla > 100 THEN (CASE WHEN docfee1 < 10000 THEN docfee1 * 10000 ELSE docfee1 END) / sfla 
                ELSE NULL 
            END) as avg_ppsf
        FROM raw_records
        WHERE sfla > 0 AND docfee1 > 0 -- Ensure valid size and market sale
        GROUP BY 1
        ORDER BY avg_value DESC
        """
        with get_db() as db:
            return db.query(sql)


    def get_price_by_type(self) -> pd.DataFrame:
        """
        Returns average price trends segmented by Property Type.
        Uses raw Abstract Codes to classify (1212=SFR, 1230=Condo).
        """
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
        WHERE tx_year > '2000'
        GROUP BY 1, 2
        HAVING sales_count > 5 -- Filter noise
        ORDER BY tx_year, prop_type
        """.format(sales_cte=SALES_EVENTS_SQL)
        
        with get_db() as db:
            return db.query(sql)

    def get_owner_purchase_trends(self) -> pd.DataFrame:
        """
        Analyzes distinct trends of *current* owners based on when they bought (recdate1).
        Note: We cannot analyze past owners of sold properties, only current holders.
        """
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
        WHERE purchase_year > '1990' AND purchase_year IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1
        """
        with get_db() as db:
            return db.query(sql)

    def get_seasonality_stats(self) -> pd.DataFrame:
        """
        Aggregates sales metrics by Month and City to reveal seasonal patterns.
        """
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
        WHERE tx_year > '2010' -- Focus on recent history for relevance
        GROUP BY 1, 2, 3
        ORDER BY month_num
        """.format(analytics_query=ANALYTICS_SQL)
        
        with get_db() as db:
            return db.query(sql)
    def get_renovation_impact(self) -> pd.DataFrame:
        """
        Aggregates metrics to compare Renovated vs Original Condition properties.
        """
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
        """.format(analytics_query=ANALYTICS_SQL)
        
        with get_db() as db:
            return db.query(sql)

    def get_comparable_sales(self, target_city: str, target_type: str, target_sfla: int) -> pd.DataFrame:
        """
        Finds comparable sales from the last 18 months.
        Criteria: Same City, Same Type, SqFt +/- 25%.
        """
        type_filter = ""
        if target_type == "Single Family":
            type_filter = "AND (r.abst1 = '1212' OR r.abst2 = '1212')"
        elif target_type == "Condo":
            type_filter = "AND (r.abst1 IN ('1230', '1231') OR r.abst2 IN ('1230', '1231'))"
        elif target_type == "Townhouse":
            type_filter = "AND (r.abst1 = '1214' OR r.abst2 = '1214')"
        
        # City Filter
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
        
        min_sqft = int(target_sfla * 0.75)
        max_sqft = int(target_sfla * 1.25)
        
        with get_db() as db:
            return db.query(sql, (min_sqft, max_sqft))
