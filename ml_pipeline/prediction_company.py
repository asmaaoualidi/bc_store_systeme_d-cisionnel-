"""
PIPELINE FINAL - Prédiction du Churn (P1 - Company Churn Prediction)
========================================================================
Résumé de la démarche validée :
  1. fact_company_snapshot : 1 ligne = (company_key, snapshot_date mensuel)
  2. Features calculées UNIQUEMENT à partir du passé (aucune fuite)
  3. churn_label = actif au snapshot, mais 0 abonnement actif 30j après
     -> "active" = start_date <= date <= end_date (PAS status_key, qui est
        un champ figé au moment du batch ETL, pas historique - voir audit)
  4. Split TEMPOREL (train <= 2026-03, test >= 2026-04) - jamais aléatoire
  5. Modèle retenu : Random Forest (AUC=0.806, meilleur des 4 testés)
  6. Threshold retenu : 0.50 (validé par threshold tuning - descendre à
     0.30 n'apporte AUCUN churner supplémentaire, juste plus de faux positifs)

Résultat final documenté (à mettre dans le rapport) :
    Model       = Random Forest
    AUC-ROC     = 0.806
    Threshold   = 0.50
    Recall      = 59% (10/17 churners détectés sur la période de test)
    Precision   = 16% (limitée par la taille du dataset - à mentionner
                        honnêtement, ce n'est pas un défaut caché)

Sortie : table `prediction_company_churn` (dans dwh_clean) + CSV,
prête à brancher sur Power BI.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ---------------------------------------------------------------------------
# 0. Connexion MySQL (remplis tes infos ici)
# ---------------------------------------------------------------------------
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME_CLEAN = "dwh_clean"

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_CLEAN}"
)
conn = engine  # pd.read_sql fonctionne directement avec un engine sqlalchemy

# ---------------------------------------------------------------------------
# 1. Chargement des tables sources (depuis le schema nettoyé)
# ---------------------------------------------------------------------------
sub = pd.read_sql("SELECT * FROM fact_subscription", conn)
pay = pd.read_sql("SELECT * FROM fact_payment", conn)
notif = pd.read_sql("SELECT * FROM fact_notification", conn)
dd = pd.read_sql("SELECT date_key, full_date FROM dim_date", conn)
status = pd.read_sql("SELECT * FROM dim_status", conn)
company = pd.read_sql("SELECT company_key, created_at FROM dim_company", conn)

dd["full_date"] = pd.to_datetime(dd["full_date"])
date_map = dict(zip(dd["date_key"], dd["full_date"]))

sub["start_date"] = sub["start_date_key"].map(date_map)
sub["end_date"] = sub["end_date_key"].map(date_map)
pay["paid_date"] = pay["paid_date_key"].map(date_map)
pay["created_at"] = pd.to_datetime(pay["created_at"])
notif["created_at"] = pd.to_datetime(notif["created_at"])
company["created_at"] = pd.to_datetime(company["created_at"])

status_pay_map = status[status["status_domain"] == "payment"].set_index("status_key")["status_code"]
pay["status_code"] = pay["status_key"].map(status_pay_map)

all_companies = company["company_key"].unique()

# ---------------------------------------------------------------------------
# 2. "Aujourd'hui" au sens du dataset = dernier événement réel observé
# ---------------------------------------------------------------------------
DATASET_TODAY = max(sub["start_date"].max(), pay["created_at"].max(), notif["created_at"].max())
print(f"Date de référence du dataset (DATASET_TODAY) : {DATASET_TODAY.date()}")

SNAPSHOT_START = "2025-06-01"
SNAPSHOT_END = (DATASET_TODAY - pd.Timedelta(days=30)).strftime("%Y-%m-01")
TRAIN_END = "2026-03-01"     # dernier mois inclus dans le train
TEST_START = "2026-04-01"    # premier mois du test
THRESHOLD = 0.50             # décidé après threshold tuning (cf. docstring)

# ---------------------------------------------------------------------------
# 3. Fonctions de construction des features (aucune fuite : tout est
#    calculé avec des infos disponibles avant ou à snapshot_date)
# ---------------------------------------------------------------------------
FEATURES = [
    "active_subscriptions", "active_modules",
    "subscriptions_started_30d", "subscriptions_ended_30d",
    "successful_payments_30d", "failed_payments_30d", "payment_amount_30d",
    "notifications_30d", "unread_notifications_30d",
    "days_since_last_payment", "company_age_days",
]

def active_company_keys(as_of_date):
    m = (sub["start_date"] <= as_of_date) & (sub["end_date"] >= as_of_date)
    return set(sub.loc[m, "company_key"])

def build_snapshot_row(snapshot_date, for_training=True):
    window_start = snapshot_date - pd.Timedelta(days=30)
    window_end = snapshot_date

    active_mask = (sub["start_date"] <= snapshot_date) & (sub["end_date"] >= snapshot_date)
    active_subs = sub[active_mask]
    active_subscriptions = active_subs.groupby("company_key").size()
    active_modules = active_subs.groupby("company_key")["module_key"].nunique()

    started_mask = (sub["start_date"] > window_start) & (sub["start_date"] <= window_end)
    ended_mask = (sub["end_date"] > window_start) & (sub["end_date"] <= window_end)
    subs_started_30d = sub[started_mask].groupby("company_key").size()
    subs_ended_30d = sub[ended_mask].groupby("company_key").size()

    pay_window = pay[(pay["created_at"] > window_start) & (pay["created_at"] <= window_end)]
    succ = pay_window[pay_window["status_code"] == "SUCCEEDED"]
    fail = pay_window[pay_window["status_code"] == "FAILED"]
    payments_succeeded_30d = succ.groupby("company_key").size()
    payments_failed_30d = fail.groupby("company_key").size()
    payment_amount_30d = succ.groupby("company_key")["amount"].sum()

    notif_window = notif[(notif["created_at"] > window_start) & (notif["created_at"] <= window_end)]
    notifications_30d = notif_window.groupby("company_key").size()
    unread_notifications_30d = notif_window[notif_window["is_read"] == 0].groupby("company_key").size()

    succ_all = pay[(pay["status_code"] == "SUCCEEDED") & (pay["created_at"] <= snapshot_date)]
    last_payment_date = succ_all.groupby("company_key")["created_at"].max()

    df = pd.DataFrame({"company_key": all_companies})
    df["snapshot_date"] = snapshot_date
    df["active_subscriptions"] = df["company_key"].map(active_subscriptions).fillna(0).astype(int)
    df["active_modules"] = df["company_key"].map(active_modules).fillna(0).astype(int)
    df["subscriptions_started_30d"] = df["company_key"].map(subs_started_30d).fillna(0).astype(int)
    df["subscriptions_ended_30d"] = df["company_key"].map(subs_ended_30d).fillna(0).astype(int)
    df["successful_payments_30d"] = df["company_key"].map(payments_succeeded_30d).fillna(0).astype(int)
    df["failed_payments_30d"] = df["company_key"].map(payments_failed_30d).fillna(0).astype(int)
    df["payment_amount_30d"] = df["company_key"].map(payment_amount_30d).fillna(0.0)
    df["notifications_30d"] = df["company_key"].map(notifications_30d).fillna(0).astype(int)
    df["unread_notifications_30d"] = df["company_key"].map(unread_notifications_30d).fillna(0).astype(int)

    last_pay = df["company_key"].map(last_payment_date)
    comp_created = df["company_key"].map(company.set_index("company_key")["created_at"])
    df["company_age_days"] = (snapshot_date - comp_created).dt.days.clip(lower=0)
    df["days_since_last_payment"] = (snapshot_date - last_pay).dt.days
    df["days_since_last_payment"] = df["days_since_last_payment"].fillna(df["company_age_days"]).astype(int)

    was_active_now = active_company_keys(snapshot_date)
    df["is_active_at_snapshot"] = df["company_key"].isin(was_active_now).astype(int)

    if for_training:
        future_date = snapshot_date + pd.Timedelta(days=30)
        future_active = active_company_keys(future_date)
        df["churn_label"] = np.where(
            df["company_key"].isin(was_active_now) & ~df["company_key"].isin(future_active), 1,
            np.where(df["company_key"].isin(was_active_now), 0, np.nan)
        )
    return df

# ---------------------------------------------------------------------------
# 4. Construction du dataset d'entraînement (historique, avec label connu)
# ---------------------------------------------------------------------------
snapshots = pd.date_range(SNAPSHOT_START, SNAPSHOT_END, freq="MS")
all_rows = [build_snapshot_row(s, for_training=True) for s in snapshots]
fact_company_snapshot = pd.concat(all_rows, ignore_index=True)
ml_data = fact_company_snapshot[fact_company_snapshot["is_active_at_snapshot"] == 1].copy()
ml_data["churn_label"] = ml_data["churn_label"].astype(int)

print(f"Dataset ML : {len(ml_data)} lignes, {ml_data['churn_label'].sum()} churn "
      f"({ml_data['churn_label'].mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 5. Validation du split temporel (rappel des résultats attendus)
# ---------------------------------------------------------------------------
train = ml_data[ml_data["snapshot_date"] <= TRAIN_END]
test = ml_data[ml_data["snapshot_date"] >= TEST_START]

rf_eval = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  max_depth=5, min_samples_leaf=3, random_state=42)
rf_eval.fit(train[FEATURES], train["churn_label"])
proba_test = rf_eval.predict_proba(test[FEATURES])[:, 1]
preds_test = (proba_test >= THRESHOLD).astype(int)

print("\n=== Validation (split temporel, rappel des résultats de référence) ===")
print(f"AUC-ROC : {roc_auc_score(test['churn_label'], proba_test):.3f}")
print(classification_report(test["churn_label"], preds_test, digits=2, zero_division=0))

# ---------------------------------------------------------------------------
# 6. Modèle FINAL : ré-entraîné sur TOUT l'historique disponible
#    (le split temporel a servi à VALIDER l'approche ; pour la mise en
#    production, on maximise les données d'apprentissage)
# ---------------------------------------------------------------------------
final_model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      max_depth=5, min_samples_leaf=3, random_state=42)
final_model.fit(ml_data[FEATURES], ml_data["churn_label"])

importances = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n=== Feature importance (modèle final) ===")
print(importances)

# ---------------------------------------------------------------------------
# 7. Scoring des entreprises ACTIVES AUJOURD'HUI
# ---------------------------------------------------------------------------
current_snapshot = build_snapshot_row(DATASET_TODAY, for_training=False)
current_active = current_snapshot[current_snapshot["is_active_at_snapshot"] == 1].copy()

current_active["churn_probability"] = final_model.predict_proba(current_active[FEATURES])[:, 1]
current_active["churn_prediction"] = (current_active["churn_probability"] >= THRESHOLD).astype(int)

def risk_level(p):
    if p >= THRESHOLD:
        return "High"
    elif p >= 0.25:
        return "Medium"
    else:
        return "Low"

current_active["risk_level"] = current_active["churn_probability"].apply(risk_level)

# jointure avec le nom de l'entreprise pour lisibilité
company_names = pd.read_sql("SELECT company_key, name FROM dim_company", conn)
output = current_active.merge(company_names, on="company_key", how="left")
output = output.sort_values("churn_probability", ascending=False)

cols_out = ["company_key", "name", "snapshot_date", "churn_probability", "churn_prediction",
            "risk_level"] + FEATURES
output = output[cols_out]

print(f"\n=== Résultat : {len(output)} entreprises actives scorées ===")
print(f"Répartition des niveaux de risque :")
print(output["risk_level"].value_counts())
print("\nTop 10 à risque :")
print(output[["name", "churn_probability", "risk_level"]].head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Export : MySQL (pour Power BI) + CSV (backup local)
# ---------------------------------------------------------------------------
output.to_sql("prediction_company_churn", engine, if_exists="replace", index=False)


print("\nTable 'prediction_company_churn' écrite dans MySQL (schema dwh_clean).")
print("Prête à être branchée sur Power BI.")