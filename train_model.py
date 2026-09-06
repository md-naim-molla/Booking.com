"""
Benchmark Machine Learning Pipeline & SHAP Explainability for Hotel Price Prediction.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import shap

def load_and_preprocess(filepath="data/hotel_prediction_dataset.csv"):
    df = pd.read_csv(filepath)
    print(f"[+] Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")

    # Target variable: log(price_net)
    df = df.dropna(subset=['price_net'])
    df = df[df['price_net'] > 0]
    
    y = np.log(df['price_net'])
    
    # Feature Selection for ML modeling
    numeric_features = [
        'star_rating', 'review_score_overall', 'review_count', 'distance_to_center_km',
        'breakfast_included', 'free_cancellation', 'is_beachfront', 'has_free_wifi',
        'has_swimming_pool', 'has_parking', 'has_air_conditioning', 'rooms_left_warning',
        'taxes_and_charges', 'discount_pct', 'has_spa', 'has_fitness_center',
        'has_restaurant', 'has_bar', 'has_airport_shuttle', 'has_24h_front_desk',
        'pets_allowed', 'photo_count'
    ]
    
    # Ensure numeric columns exist and impute medians
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Categorical features: One-Hot Encode City & Property Type
    cat_cols = ['city', 'property_type']
    for c in cat_cols:
        if c not in df.columns:
            df[c] = 'Unknown'
        df[c] = df[c].fillna('Unknown')
        
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([df[numeric_features], X_cat], axis=1)
    
    # Clean column names for LightGBM/XGBoost compatibility
    X.columns = [str(col).replace(' ', '_').replace("'", "") for col in X.columns]
    
    return X, y, df

def evaluate_models(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "1. Hedonic OLS (Linear)": LinearRegression(),
        "2. Ridge Regression": Ridge(alpha=1.0),
        "3. Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "4. LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "5. XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    }
    
    results = []
    
    print("\n" + "=" * 75)
    print("📊 5-FOLD CROSS-VALIDATION BENCHMARK TOURNAMENT")
    print("=" * 75)
    
    for name, model in models.items():
        r2_scores, rmse_scores, mae_scores, mape_scores = [], [], [], []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            r2_scores.append(r2_score(y_test, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
            mae_scores.append(mean_absolute_error(y_test, preds))
            mape_scores.append(mean_absolute_percentage_error(np.exp(y_test), np.exp(preds)))
            
        results.append({
            "Model": name,
            "R² Score": f"{np.mean(r2_scores):.4f} (±{np.std(r2_scores):.3f})",
            "RMSE (log)": f"{np.mean(rmse_scores):.4f}",
            "MAE (log)": f"{np.mean(mae_scores):.4f}",
            "MAPE (%)": f"{np.mean(mape_scores)*100:.2f}%"
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("=" * 75)
    
    return res_df

def compute_shap_importance(X, y):
    print("\n[+] Computing Tree-SHAP Explainability (XGBoost)...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate Mean Absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    
    importance_df.to_csv("data/shap_feature_importance.csv", index=False)
    
    print("\n🏆 TOP 15 PREDICTIVE FEATURES BY SHAP IMPORTANCE (Marginal Impact on Log Price):")
    print("-" * 60)
    for i, row in importance_df.head(15).reset_index().iterrows():
        print(f"{i+1:2d}. {row['Feature']:<30} | SHAP: {row['Mean_Absolute_SHAP']:.4f}")
    print("-" * 60)
    print("  ✓ Saved SHAP rankings to data/shap_feature_importance.csv")

if __name__ == "__main__":
    X, y, df = load_and_preprocess()
    results_table = evaluate_models(X, y)
    compute_shap_importance(X, y)
