
-- Retail Business Performance & Profitability Analysis SQL Queries

-- 1. Create Profit Column
ALTER TABLE retail_data
ADD Profit DECIMAL(10,2);

UPDATE retail_data
SET Profit = Sales - Cost;

-- 2. Profit Margin by Category
SELECT 
    Category,
    SUM(Sales) AS Total_Sales,
    SUM(Profit) AS Total_Profit,
    ROUND((SUM(Profit)/SUM(Sales))*100,2) AS Profit_Margin_Percent
FROM retail_data
GROUP BY Category
ORDER BY Profit_Margin_Percent ASC;

-- 3. Profit Margin by Sub-Category
SELECT 
    Sub_Category,
    SUM(Sales) AS Total_Sales,
    SUM(Profit) AS Total_Profit,
    ROUND((SUM(Profit)/SUM(Sales))*100,2) AS Profit_Margin_Percent
FROM retail_data
GROUP BY Sub_Category
ORDER BY Profit_Margin_Percent ASC;

-- 4. Seasonal Monthly Sales Trend
SELECT 
    MONTH(Order_Date) AS Month,
    Category,
    SUM(Sales) AS Monthly_Sales
FROM retail_data
GROUP BY MONTH(Order_Date), Category
ORDER BY Month;

-- 5. Inventory Performance Analysis
SELECT 
    Category,
    AVG(Inventory_Days) AS Avg_Inventory_Days,
    SUM(Profit) AS Total_Profit
FROM retail_data
GROUP BY Category
ORDER BY Avg_Inventory_Days DESC;
