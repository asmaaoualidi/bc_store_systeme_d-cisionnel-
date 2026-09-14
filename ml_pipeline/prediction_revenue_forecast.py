"""
PIPELINE MODULE 3 - Prévision du revenu (Revenue Forecast)
================================================================
Corrections appliquées AVANT tout calcul (décidées et validées avec l'audit) :
  1. Bug ETL : fact_payment.status_key = 2 correspond au code "PENDING"
     du domaine SUBSCRIPTION (dim_status), pas au domaine PAYMENT.
     51 paiements concernés, tous avec paid_date_key rempli (preuve qu'ils
     ont réellement été encaissés) -> recodés en status_key = 5 (SUCCEEDED).
  2. dim_plan.billing_cycle mélange deux langues selon la source
     (MongoDB: "monthly"/"annual", PostgreSQL: "Mensuel"/"Annuel")
     -> harmonisé en "Mensuel"/"Annuel".
  3. Le dernier mois du dataset (contenant DATASET_TODAY) est TOUJOURS
     partiel -> exclu de toute agrégation et de toute validation temporelle.

Indicateurs calculés par mois :
  - revenue_total              : somme des paiements SUCCEEDED (par created_at)
  - active_companies           : nb d'entreprises ayant >=1 abonnement
                                  couvrant ce mois (même définition que Module 1)
  - average_revenue_per_company (ARPU) = revenue_total / active_companies

Modèles comparés (par indicateur, sélection automatique du meilleur par MAE) :
  naive / moving_avg / linear_reg / growth_rate / log_linear

IMPORTANT : ce script ne "triche" pas pour améliorer les résultats.
Si un indicateur reste mal prédit (MAE relative élevée), c'est reporté
tel quel avec un niveau de fiabilité explicite.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# 0. Connexion MySQL
# ---------------------------------------------------------------------------
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME_CLEAN = "dwh_clean"

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_CLEAN}"
)
conn = engine

TRAIN_TEST_SPLIT_N_TEST = 3   # nb de mois complets gardés pour le test
FORECAST_HORIZON = 3

# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------
pay = pd.read_sql("SELECT * FROM fact_payment", conn)
# NOTE : fact_subscription contient AUSSI company_key -> on ne sélectionne
# pas cette colonne ici pour éviter un conflit de nom au merge avec fact_payment.
sub = pd.read_sql("SELECT subscription_key, company_key, plan_key, module_key, "
                   "start_date_key, end_date_key FROM fact_subscription", conn)
plan = pd.read_sql("SELECT plan_key, billing_cycle FROM dim_plan", conn)
status = pd.read_sql("SELECT * FROM dim_status", conn)
dd = pd.read_sql("SELECT date_key, full_date FROM dim_date", conn)

dd["full_date"] = pd.to_datetime(dd["full_date"])
date_map = dict(zip(dd["date_key"], dd["full_date"]))
pay["created_at"] = pd.to_datetime(pay["created_at"])
sub["start_date"] = sub["start_date_key"].map(date_map)
sub["end_date"] = sub["end_date_key"].map(date_map)

# ---------------------------------------------------------------------------
# 2. Correction 1 : bug ETL status_key=2 -> 5 (SUCCEEDED)
# ---------------------------------------------------------------------------
pay["status_key"] = pay["status_key"].replace({2: 5})
status_pay_map = status[status["status_domain"] == "payment"].set_index("status_key")["status_code"]
pay["status_code"] = pay["status_key"].map(status_pay_map)

# ---------------------------------------------------------------------------
# 3. Correction 2 : harmonisation billing_cycle
# ---------------------------------------------------------------------------
billing_cycle_map = {"monthly": "Mensuel", "Mensuel": "Mensuel", "annual": "Annuel", "Annuel": "Annuel"}
plan["billing_cycle_normalized"] = plan["billing_cycle"].map(billing_cycle_map)

# ---------------------------------------------------------------------------
# 4. Date de référence du dataset + exclusion du mois partiel
# ---------------------------------------------------------------------------
DATASET_TODAY = max(sub["start_date"].max(), pay["created_at"].max())
current_month_start = pd.Timestamp(year=DATASET_TODAY.year, month=DATASET_TODAY.month, day=1)

months_all = pd.date_range("2025-06-01", current_month_start, freq="MS")
months_complete = months_all[months_all < current_month_start]  # on exclut le mois en cours (partiel)

print(f"DATASET_TODAY = {DATASET_TODAY.date()}  ->  mois exclu (partiel) : {current_month_start.date()}")
print(f"{len(months_complete)} mois complets utilisables pour l'analyse et la validation.")

# ---------------------------------------------------------------------------
# 5. Agrégation mensuelle : revenue_total, active_companies, ARPU
# ---------------------------------------------------------------------------
succ = pay[pay["status_code"] == "SUCCEEDED"].merge(sub, on="subscription_key").merge(plan, on="plan_key")
succ["month"] = succ["created_at"].dt.to_period("M").dt.to_timestamp()

revenue_by_month = succ[succ["month"].isin(months_complete)].groupby("month")["amount"].sum()

active_companies_by_month = {}
for m in months_complete:
    active = sub[(sub["start_date"] <= m) & (sub["end_date"] >= m)]
    active_companies_by_month[m] = active["company_key"].nunique()

monthly = pd.DataFrame({
    "revenue_total": revenue_by_month.reindex(months_complete, fill_value=0.0),
    "active_companies": pd.Series(active_companies_by_month),
})
monthly["average_revenue_per_company"] = (
    monthly["revenue_total"] / monthly["active_companies"].replace(0, np.nan)
).fillna(0)
monthly["revenue_growth_pct"] = monthly["revenue_total"].pct_change() * 100

print("\n=== Série mensuelle finale (mois complets uniquement) ===")
print(monthly.round(1).to_string())

# ---------------------------------------------------------------------------
# 6. Fonctions de modélisation (5 candidats)
# ---------------------------------------------------------------------------
def evaluate_models(y, n_test):
    """Walk-forward validation (rolling-origin) : à chaque étape, on entraîne
    sur tout l'historique disponible jusqu'à t, on prédit UNIQUEMENT le mois
    t+1, puis on avance d'un mois et on recommence. C'est plus robuste qu'un
    simple split train/test fixe, surtout avec peu de points (retenu suite
    à la comparaison avec l'analyse walk-forward proposée en discussion)."""
    n = len(y)
    min_train = n - n_test  # même point de départ que l'ancien holdout, mais on avance mois par mois

    errors = {name: [] for name in ["naive", "moving_avg", "linear_reg", "growth_rate", "log_linear"]}

    for cutoff in range(min_train, n):
        y_train = y[:cutoff]
        y_true_next = y[cutoff]  # valeur réelle du mois suivant à prédire
        t_train = np.arange(cutoff).reshape(-1, 1)

        preds = {}
        preds["naive"] = y_train[-1]
        preds["moving_avg"] = y_train[-3:].mean()

        lr = LinearRegression().fit(t_train, y_train)
        preds["linear_reg"] = max(lr.predict([[cutoff]])[0], 0)

        growth_rates = pd.Series(y_train).pct_change().dropna()
        growth_rates = growth_rates.replace([np.inf, -np.inf], np.nan).dropna()
        avg_growth = np.clip(growth_rates.median() if len(growth_rates) else 0, -0.5, 2.0)
        preds["growth_rate"] = y_train[-1] * (1 + avg_growth)

        if (y_train > 0).all():
            lr_log = LinearRegression().fit(t_train, np.log(y_train))
            preds["log_linear"] = np.exp(lr_log.predict([[cutoff]])[0])
        else:
            preds["log_linear"] = None

        for name, pred in preds.items():
            if pred is None:
                continue
            errors[name].append(abs(pred - y_true_next))

    y_test_mean = y[min_train:].mean()
    results = []
    for name, errs in errors.items():
        if not errs:
            continue
        mae = np.mean(errs)
        mae_rel = mae / y_test_mean * 100 if y_test_mean > 0 else np.nan
        results.append((name, mae, mae_rel))

    # avg_growth final (sur toute la série) pour le forecast futur du meilleur modèle si retenu
    growth_rates_full = pd.Series(y).pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    avg_growth_full = np.clip(growth_rates_full.median() if len(growth_rates_full) else 0, -0.5, 2.0)

    return sorted(results, key=lambda x: x[1]), avg_growth_full

def forecast_best_model(y, best_name, avg_growth, horizon):
    """Ré-entraîne le meilleur modèle sur TOUTE la série et prévoit `horizon` mois."""
    t = np.arange(len(y)).reshape(-1, 1)
    if best_name == "naive":
        return np.full(horizon, y[-1])
    elif best_name == "moving_avg":
        return np.full(horizon, y[-3:].mean())
    elif best_name == "linear_reg":
        lr = LinearRegression().fit(t, y)
        future_t = np.array([[len(y) + i] for i in range(horizon)])
        return np.clip(lr.predict(future_t), 0, None)
    elif best_name == "growth_rate":
        path = [y[-1]]
        for _ in range(horizon):
            path.append(path[-1] * (1 + avg_growth))
        return np.array(path[1:])
    elif best_name == "log_linear":
        lr_log = LinearRegression().fit(t, np.log(y))
        future_t = np.array([[len(y) + i] for i in range(horizon)])
        return np.exp(lr_log.predict(future_t))
    else:
        raise ValueError(best_name)

def fiabilite_label(mae_rel):
    if mae_rel <= 30:
        return "High"
    elif mae_rel <= 60:
        return "Medium"
    else:
        return "Low"

# ---------------------------------------------------------------------------
# 7. Application sur revenue_total ET ARPU (les deux indicateurs)
# ---------------------------------------------------------------------------
report_rows = []
for target_col, target_label in [
    ("revenue_total", "Revenue total (DH)"),
    ("average_revenue_per_company", "ARPU - Revenu moyen par entreprise (DH)"),
]:
    y = monthly[target_col].values.astype(float)
    n_test = TRAIN_TEST_SPLIT_N_TEST

    results, avg_growth = evaluate_models(y, n_test)
    print(f"\n=== Comparaison des modèles : {target_label} ===")
    for name, mae, mae_rel in results:
        print(f"  {name:12s}  MAE={mae:>12,.0f}  MAE_relatif={mae_rel:>5.1f}%")

    best_name, best_mae, best_mae_rel = results[0]
    forecast_path = forecast_best_model(y, best_name, avg_growth, FORECAST_HORIZON)

    print(f"  -> Meilleur modèle retenu : {best_name} "
          f"(MAE_relatif={best_mae_rel:.1f}%, fiabilité={fiabilite_label(best_mae_rel)})")
    print(f"  -> Prévision {FORECAST_HORIZON} mois : {np.round(forecast_path, 1)}")

    report_rows.append({
        "indicateur": target_label,
        "valeur_actuelle": round(y[-1], 1),
        "prevision_3m": round(forecast_path[-1], 1),
        "meilleur_modele": best_name,
        "mae": round(best_mae, 1),
        "mae_relatif_pct": round(best_mae_rel, 1),
        "fiabilite": fiabilite_label(best_mae_rel),
    })

report = pd.DataFrame(report_rows)
report["snapshot_date"] = current_month_start.strftime("%Y-%m-%d")
report["forecast_date"] = (current_month_start + pd.DateOffset(months=FORECAST_HORIZON)).strftime("%Y-%m-01")

print("\n=== RÉSUMÉ FINAL - Module 3 ===")
print(report.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Export : MySQL (pour Power BI) + CSV (backup local + historique brut)
# ---------------------------------------------------------------------------
report.to_sql("prediction_revenue_forecast", engine, if_exists="replace", index=False)
report.to_csv("./prediction_revenue_forecast.csv", index=False)
monthly.reset_index().rename(columns={"index": "month"}).to_csv("./revenue_history_monthly.csv", index=False)

print("\nTable 'prediction_revenue_forecast' écrite dans MySQL (schema dwh_clean).")
print("Historique mensuel exporté dans revenue_history_monthly.csv (pour graphique Power BI).")