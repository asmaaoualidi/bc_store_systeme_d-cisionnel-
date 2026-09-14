# Data Warehouse & Prédiction — Plateforme ERP SaaS

Projet de fin d'année (PFA) réalisé dans le cadre d'un stage chez **BC Skills Group** (Safi, Maroc), consistant à concevoir un Data Warehouse et développer des modules d'analyse prédictive pour une plateforme ERP en mode SaaS.

## 🎯 Objectifs du projet

- Consolider les données issues de deux systèmes sources hétérogènes (**MongoDB** + **PostgreSQL**) dans un Data Warehouse MySQL structuré en schéma en étoile
- Auditer et corriger la qualité des données avant toute modélisation
- Développer trois modules de Machine Learning :
  - **Prédiction du churn** (Random Forest, AUC-ROC = 0.806)
  - **Prévision de la popularité des modules** (comparaison Naive / Moving Average / Linear Regression)
  - **Analyse et décomposition du chiffre d'affaires** (ARPU vs croissance du portefeuille)
- Restituer les résultats via un tableau de bord décisionnel **Power BI**

## 📂 Structure du dépôt

```

│   
├── ml-pipelines/           # Pipelines Python par module prédictif
│   ├── churn/
│   ├── module-popularity/
│   └── revenue/
├── data/            # Exports SQL prêts à importer 
├── dashboard/              # Tableau de bord Power BI (.pbix) 
├── docs/            
└── diagrams/                # Diagrammes UML/MCD (cas d'utilisation, activité, MCD) cree par plantuml

```

## 🏗️ Architecture du Data Warehouse

Schéma en étoile avec 5 dimensions (`dim_company`, `dim_module`, `dim_plan`, `dim_status`, `dim_date`) et 3 tables de faits (`fact_subscription`, `fact_payment`, `fact_notification`). Voir [`diagrams/mcd.png`](diagrams/mcd.png) pour le détail.

## 🔍 Points méthodologiques clés

- **Prévention de la fuite de données** : détection et correction d'une fuite de données dans le modèle de churn (variable `nb_active_subscriptions` corrélée à 100% avec le label)
- **Validation temporelle** : split train/test chronologique (jamais aléatoire) pour tous les modules prédictifs, avec walk-forward validation pour le module revenue
- **Transparence sur les limites** : le module de prévision du chiffre d'affaires n'a pas atteint le seuil de fiabilité retenu (30%) — décision assumée de ne pas présenter de prévision chiffrée dans le dashboard plutôt que d'afficher un résultat peu fiable

## ⚙️ Stack technique

- **Base de données** : MySQL (Data Warehouse), MongoDB & PostgreSQL (sources)
- **Traitement des données & ML** : Python (pandas, scikit-learn, XGBoost, SQLAlchemy)
- **Restitution** : Power BI (modèle de données, mesures DAX, dashboard)

## 🚀 Utilisation

Chaque script dans `ml-pipelines/` est autonome : renseigner les identifiants de connexion MySQL en haut du fichier (`DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_NAME_CLEAN`), puis exécuter :

```bash
pip install pandas scikit-learn xgboost sqlalchemy mysql-connector-python
python ml-pipelines/churn/final_churn_pipeline.py
```

Les scripts SQL dans `data`  à exécuter directement dans MySQL Workbench sur le schéma cible.

## 📊 Résultats principaux

| Module | Modèle retenu | Métrique clé |
|---     |---            |---           |
| Churn  | Random Forest | AUC-ROC = 0.806, Recall = 59% |
| Popularité modules | Naive / Moving Avg / Linear Reg (par module) | 10/11 modules avec fiabilité élevée |
| Revenue | — (aucun modèle retenu) | Meilleure MAE relative = 48.4% (> seuil 30%) |

## 📄 Documentation

Le détail complet de la méthodologie, des résultats et des choix de modélisation est disponible dans [`docs/`](docs/), notamment le rapport de stage complet.

---

*Projet réalisé par Asmaa Oualidi — EMSI, Ingénierie Informatique et Réseaux.*
