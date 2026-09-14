"""
PIPELINE FINAL - Module 2 : Popularité et prévision des modules ERP
=======================================================================
Démarche validée :
  1. Harmonisation module_key -> module_group (fusion des modules identiques
     entre MongoDB et PostgreSQL, "Facturation & Comptabilité" gardé isolé
     car c'est un bundle PostgreSQL sans équivalent 1:1 côté MongoDB)
  2. Agrégation mensuelle : active_subscriptions, new_subscriptions
  3. Target = part relative (%), pas le nombre brut, pour neutraliser
     l'effet de croissance globale de la base clients
  4. Comparaison de 3 modèles par module (naive / moyenne mobile /
     régression linéaire temporelle), le meilleur (MAE la plus basse
     sur les 4 derniers mois de test) est retenu automatiquement
  5. Prévision à horizon 3 mois (pas plus, vu les 16 points disponibles)
  6. Contraintes : prévisions bornées [0,100], puis normalisées pour
     que la somme des parts prévues = 100%
  7. Indicateur de fiabilité (MAE relative) : 🟢 <=30% / 🟠 <=60% / 🔴 >60%

Sortie : table `prediction_module_popularity` (MySQL, schema dwh_clean) + CSV.
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

TRAIN_N = 12
FORECAST_HORIZON = 3

# ---------------------------------------------------------------------------
# 1. Harmonisation module_key -> module_group (à valider/adapter selon
#    les module_key réels de TA base, vérifiés via dim_module)
# ---------------------------------------------------------------------------
module_group_map = {
    1: "Parc automobile", 2: "Immobilisation", 3: "Ressources Humaines",
    4: "Facturation", 5: "Comptabilité", 6: "Gestion de Stock", 7: "CRM",
    8: "Achats", 9: "Trésorerie", 10: "Reporting",
    11: "Facturation & Comptabilité", 12: "CRM", 13: "Ressources Humaines",
    14: "Gestion de Stock",
}

# ---------------------------------------------------------------------------
# 2. Chargement des données
# ---------------------------------------------------------------------------
sub = pd.read_sql("SELECT * FROM fact_subscription", conn)
dd = pd.read_sql("SELECT date_key, full_date FROM dim_date", conn)

dd["full_date"] = pd.to_datetime(dd["full_date"])
date_map = dict(zip(dd["date_key"], dd["full_date"]))
sub["start_date"] = sub["start_date_key"].map(date_map)
sub["end_date"] = sub["end_date_key"].map(date_map)
sub["module_group"] = sub["module_key"].map(module_group_map)

missing_map = sub[sub["module_group"].isna()]["module_key"].unique()
if len(missing_map):
    raise ValueError(f"module_key non mappés dans module_group_map : {missing_map}. "
                      f"Vérifie dim_module et complète le dictionnaire ci-dessus.")

DATASET_TODAY = pd.Timestamp(max(sub["start_date"].max(), pd.Timestamp.now()))
DATASET_TODAY = sub["start_date"].max()  # dernier événement réel observé

# ---------------------------------------------------------------------------
# 3. Construction de la série mensuelle (active + nouvelles souscriptions)
# ---------------------------------------------------------------------------
months = pd.date_range("2025-06-01", DATASET_TODAY, freq="MS")  # on saute mai (0 partout)
groups = sorted(sub["module_group"].unique())

rows = []
for m in months:
    active = sub[(sub["start_date"] <= m) & (sub["end_date"] >= m)]
    active_counts = active.groupby("module_group").size()
    window_start = m - pd.Timedelta(days=30)
    new_subs = sub[(sub["start_date"] > window_start) & (sub["start_date"] <= m)]
    new_counts = new_subs.groupby("module_group").size()
    for g in groups:
        rows.append([m, g, active_counts.get(g, 0), new_counts.get(g, 0)])

ts = pd.DataFrame(rows, columns=["month", "module_group", "active_subscriptions", "new_subscriptions_30d"])

pivot_active = ts.pivot(index="month", columns="module_group", values="active_subscriptions")
total_active = pivot_active.sum(axis=1)
share = pivot_active.div(total_active, axis=0) * 100
share = share.reset_index()
share["t"] = range(len(share))
modules = [c for c in share.columns if c not in ["month", "t"]]

print(f"{len(months)} mois exploitables x {len(modules)} module_group.")

# ---------------------------------------------------------------------------
# 4. Comparaison des modèles + prévision à horizon 3 mois, par module
# ---------------------------------------------------------------------------
results = []
for mod in modules:
    y = share[mod].values
    t = share["t"].values.reshape(-1, 1)

    y_train, y_test = y[:TRAIN_N], y[TRAIN_N:]
    t_train, t_test = t[:TRAIN_N], t[TRAIN_N:]

    naive_pred = np.full(len(y_test), y_train[-1])
    mae_naive = mean_absolute_error(y_test, naive_pred)

    ma_pred = np.full(len(y_test), y_train[-3:].mean())
    mae_ma = mean_absolute_error(y_test, ma_pred)

    lr = LinearRegression().fit(t_train, y_train)
    lr_pred = np.clip(lr.predict(t_test), 0, 100)
    mae_lr = mean_absolute_error(y_test, lr_pred)

    candidates = {"naive": mae_naive, "moving_avg": mae_ma, "linear_reg": mae_lr}
    best_model = min(candidates, key=candidates.get)
    best_mae = candidates[best_model]
    mae_relatif = best_mae / y_test.mean() * 100 if y_test.mean() > 0 else np.nan
    fiabilite = "High" if mae_relatif <= 30 else ("Medium" if mae_relatif <= 60 else "Low")

    if best_model == "linear_reg":
        lr_final = LinearRegression().fit(t, y)
        future_t = np.array([[len(share) + i] for i in range(FORECAST_HORIZON)])
        forecast_path = np.clip(lr_final.predict(future_t), 0, 100)
    elif best_model == "moving_avg":
        forecast_path = np.full(FORECAST_HORIZON, y[-3:].mean())
    else:
        forecast_path = np.full(FORECAST_HORIZON, y[-1])

    current = y[-1]
    forecast_3m = forecast_path[-1]
    slope_est = (forecast_3m - current) / FORECAST_HORIZON
    trend = "Hausse" if slope_est > 0.3 else ("Baisse" if slope_est < -0.3 else "Stable")

    results.append({
        "module_group": mod,
        "part_actuelle_pct": round(current, 1),
        "part_prevue_3m_pct": round(forecast_3m, 1),
        "tendance": trend,
        "meilleur_modele": best_model,
        "mae": round(best_mae, 2),
        "mae_relatif_pct": round(mae_relatif, 0),
        "fiabilite": fiabilite,
    })

res = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 5. Normalisation finale (la somme des parts prévues doit faire ~100%)
# ---------------------------------------------------------------------------
res["part_prevue_3m_norm_pct"] = (res["part_prevue_3m_pct"] / res["part_prevue_3m_pct"].sum() * 100).round(1)
res = res.sort_values("part_prevue_3m_norm_pct", ascending=False)
res["forecast_date"] = (DATASET_TODAY + pd.DateOffset(months=FORECAST_HORIZON)).strftime("%Y-%m-01")
res["snapshot_date"] = DATASET_TODAY.strftime("%Y-%m-%d")

print("\n=== Résultat final ===")
print(res.to_string(index=False))
print(f"\nSomme part_actuelle : {res['part_actuelle_pct'].sum():.1f}%")
print(f"Somme part_prevue (normalisée) : {res['part_prevue_3m_norm_pct'].sum():.1f}%")

# ---------------------------------------------------------------------------
# 6. Export : MySQL (pour Power BI) + CSV (backup local)
# ---------------------------------------------------------------------------
cols_out = ["module_group", "snapshot_date", "forecast_date", "part_actuelle_pct",
            "part_prevue_3m_norm_pct", "tendance", "meilleur_modele", "mae",
            "mae_relatif_pct", "fiabilite"]
output = res[cols_out]

output.to_sql("prediction_module_popularity", engine, if_exists="replace", index=False)
output.to_csv("./prediction_module_popularity.csv", index=False)
ts.to_csv("./module_popularity_history.csv", index=False)  # historique mensuel brut, utile pour le graph Power BI

print("\nTable 'prediction_module_popularity' écrite dans MySQL (schema dwh_clean).")
print("Historique mensuel exporté dans module_popularity_history.csv (pour le graphique Power BI).")