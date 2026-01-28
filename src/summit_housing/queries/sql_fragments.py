# --- SQL Logic Fragments ---

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
    {{sales_events_sql}}
),
sales_with_metrics AS (
    SELECT 
        s.schno,
        s.tx_date,
        s.estimated_price,
        
        -- Window Function 1: Price Appreciation
        LAG(s.estimated_price) OVER (PARTITION BY s.schno ORDER BY s.tx_date) as prev_price,
        
        -- Window Function 2: Holding Period
        julianday(s.tx_date) - julianday(LAG(s.tx_date) OVER (PARTITION BY s.schno ORDER BY s.tx_date)) as days_held,
        
        -- Window Function 3: Sales Velocity
        COUNT(*) OVER (PARTITION BY s.schno) as times_sold,
        
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
        CASE WHEN r.adj_year_blt > r.year_blt THEN 1 ELSE 0 END as is_renovated,
        r.units,
        r.address,
        r.grade,
        r.cond,
        r.scenic_view
    FROM sales s
    JOIN raw_records r ON s.schno = r.schno
)
SELECT 
    *,
    ROUND((estimated_price - prev_price) * 100.0 / prev_price, 2) as growth_pct,
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
        AVG(CASE WHEN sfla > 100 THEN estimated_price / sfla ELSE NULL END) as avg_ppsf,
        COUNT(*) as sales_count
    FROM metrics
    WHERE tx_year IS NOT NULL AND tx_year >= '1980'
      {filter_clause}
    GROUP BY tx_year, city
)
SELECT 
    tx_year,
    city,
    avg_price,
    avg_ppsf,
    sales_count,
    
    AVG(avg_price) OVER (
        PARTITION BY city 
        ORDER BY tx_year 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as avg_price_3yr_ma,
    
    AVG(avg_ppsf) OVER (
        PARTITION BY city 
        ORDER BY tx_year 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as avg_ppsf_3yr_ma
FROM yearly_stats
ORDER BY city, tx_year
"""
