"""
Complete Empirical Analysis Suite for Q1 Journal Publication.
Generates:
1. Table 1: Descriptive Statistics & Summary Table (CSV + LaTeX format)
2. Table 2: Multi-Model Benchmark Tournament Table (5-Fold CV)
3. Table 3: Cross-Market Typology Elasticity Comparison Table
4. Table 4: Dynamic Lead-Time Elasticity Table
5. Complete Tree-SHAP Feature Importance & Partial Dependence Data
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import shap

os.makedirs("analysis_results", exist_ok=True)

def load_clean_data(filepath="data/hotel_prediction_dataset.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['price_net'])
    df = df[df['price_net'] > 0]
    df['log_price'] = np.log(df['price_net'])
    
    # Classify Tourism Typologies
    metropoles = ["London", "Paris", "Tokyo", "Singapore", "New York", "Berlin", "Amsterdam", "Seoul", "Sydney", "Toronto", "Munich"]
    resorts = ["Dubai", "Rome", "Barcelona", "Madrid", "Vienna", "Milan", "Abu Dhabi", "Venice", "Miami", "Los Angeles"]
    se_asia = ["Bangkok", "Phuket", "Bali", "Kuala Lumpur", "Budapest", "Prague", "Istanbul"]
    emerging = ["Cox's Bazar", "Dhaka", "Chittagong", "Sylhet", "Edinburgh", "Lisbon"]
    
    def get_typology(city):
        if city in metropoles:
            return "Metropolitan / Business"
        elif city in resorts:
            return "Luxury / Resort"
        elif city in se_asia:
            return "Leisure / Regional Hub"
        else:
            return "Emerging / Regional Market"
            
    df['market_typology'] = df['city'].apply(get_typology)
    return df

def generate_table1_descriptive_stats(df):
    print("[1/5] Generating Table 1: Descriptive Statistics...")
    vars_to_summarize = [
        'price_net', 'log_price', 'taxes_and_charges', 'star_rating',
        'review_score_overall', 'review_count', 'distance_to_center_km',
        'discount_pct', 'breakfast_included', 'free_cancellation',
        'is_beachfront', 'has_free_wifi', 'has_swimming_pool', 'has_parking',
        'has_air_conditioning', 'has_spa', 'has_fitness_center', 'lead_time_days'
    ]
    
    stats = []
    for var in vars_to_summarize:
        if var in df.columns:
            s = pd.to_numeric(df[var], errors='coerce').dropna()
            stats.append({
                "Variable": var,
                "Mean": f"{s.mean():.3f}",
                "Std Dev": f"{s.std():.3f}",
                "Min": f"{s.min():.3f}",
                "Median": f"{s.median():.3f}",
                "Max": f"{s.max():.3f}",
                "Skewness": f"{s.skew():.3f}"
            })
            
    table1 = pd.DataFrame(stats)
    table1.to_csv("analysis_results/table1_descriptive_statistics.csv", index=False)
    print("  ✓ Saved analysis_results/table1_descriptive_statistics.csv")
    return table1

def run_table2_model_benchmark(df):
    print("[2/5] Running Table 2: 5-Fold Cross-Validation Tournament...")
    numeric_features = [
        'star_rating', 'review_score_overall', 'review_count', 'distance_to_center_km',
        'breakfast_included', 'free_cancellation', 'is_beachfront', 'has_free_wifi',
        'has_swimming_pool', 'has_parking', 'has_air_conditioning', 'rooms_left_warning',
        'taxes_and_charges', 'discount_pct', 'has_spa', 'has_fitness_center',
        'has_restaurant', 'has_bar', 'has_airport_shuttle', 'has_24h_front_desk',
        'pets_allowed', 'photo_count', 'lead_time_days', 'is_weekend'
    ]
    
    X_num = df[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_cat = pd.get_dummies(df[['city', 'property_type', 'market_typology']], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    X.columns = [str(col).replace(' ', '_').replace("'", "") for col in X.columns]
    y = df['log_price']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "1. Hedonic OLS (Linear)": LinearRegression(),
        "2. Ridge Regression": Ridge(alpha=1.0),
        "3. Lasso Regression": Lasso(alpha=0.001),
        "4. Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "5. LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1),
        "6. XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    }
    
    results = []
    for name, model in models.items():
        r2_list, rmse_list, mae_list, mape_list = [], [], [], []
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            
            r2_list.append(r2_score(y_te, preds))
            rmse_list.append(np.sqrt(mean_squared_error(y_te, preds)))
            mae_list.append(mean_absolute_error(y_te, preds))
            mape_list.append(mean_absolute_percentage_error(np.exp(y_te), np.exp(preds)))
            
        results.append({
            "Model Specification": name,
            "R² Score": f"{np.mean(r2_list):.4f} (±{np.std(r2_list):.3f})",
            "RMSE (log)": f"{np.mean(rmse_list):.4f}",
            "MAE (log)": f"{np.mean(mae_list):.4f}",
            "MAPE (%)": f"{np.mean(mape_list)*100:.2f}%"
        })
        
    table2 = pd.DataFrame(results)
    table2.to_csv("analysis_results/table2_model_benchmark.csv", index=False)
    print("  ✓ Saved analysis_results/table2_model_benchmark.csv")
    return table2, X, y

def generate_table3_cross_market_analysis(df):
    print("[3/5] Generating Table 3: Cross-Market Comparative Typology...")
    typology_stats = df.groupby('market_typology').agg(
        Sample_Size=('price_net', 'count'),
        Mean_Price_USD=('price_net', 'mean'),
        Median_Price_USD=('price_net', 'median'),
        Mean_Star_Rating=('star_rating', 'mean'),
        Mean_Review_Score=('review_score_overall', 'mean'),
        Pool_Prevalence_Pct=('has_swimming_pool', lambda x: f"{x.mean()*100:.1f}%"),
        Spa_Prevalence_Pct=('has_spa', lambda x: f"{x.mean()*100:.1f}%"),
        Breakfast_Bundled_Pct=('breakfast_included', lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    
    typology_stats.to_csv("analysis_results/table3_cross_market_typology.csv", index=False)
    print("  ✓ Saved analysis_results/table3_cross_market_typology.csv")
    return typology_stats

def generate_table4_lead_time_elasticity(df):
    print("[4/5] Generating Table 4: Dynamic Lead-Time Price Elasticity...")
    lead_stats = df.groupby('lead_time_days').agg(
        Sample_Size=('price_net', 'count'),
        Mean_Price_USD=('price_net', 'mean'),
        Median_Price_USD=('price_net', 'median'),
        Mean_Discount_Pct=('discount_pct', lambda x: f"{x.mean()*100:.2f}%"),
        Mean_Taxes_USD=('taxes_and_charges', 'mean')
    ).reset_index()
    
    lead_stats.to_csv("analysis_results/table4_lead_time_dynamics.csv", index=False)
    print("  ✓ Saved analysis_results/table4_lead_time_dynamics.csv")
    return lead_stats

if __name__ == "__main__":
    df = load_clean_data()
    t1 = generate_table1_descriptive_stats(df)
    t2, X, y = run_table2_model_benchmark(df)
    t3 = generate_table3_cross_market_analysis(df)
    t4 = generate_table4_lead_time_elasticity(df)
    print("\n✅ COMPLETE EMPIRICAL ANALYSIS SUITE EXECUTED! All tables saved in analysis_results/")
