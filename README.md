# mlflow

### Ce projet contient des démos de la formation MLflow proposée par l'équipe Hi! Paris Engineering.

#
### **Pour la configuration de l'environnement, installer les requirements:**
#
```bash 
pip install -r requirements.txt 
```



## Dans la suite de ce readme, nous traitons les démos de la formation un par un :
1) Demo MLflow Tracking

    Dans ce module, nous explorons le suivi des expériences d'apprentissage automatique avec MLflow. Nous commençons 
    par les bases, examinons l'ensemble de données immobilier que nous utiliserons, puis suivons un processus d'entraînement
    de modèle simple. En tant que scientifique des données dans une entreprise immobilière, notre objectif est de prédire 
    les prix des maisons en utilisant des caractéristiques telles que la taille du terrain, la surface habitable et le type 
    de construction. Nous utilisons le codage one-hot pour traiter les variables catégorielles.

    Une fois que nous avons les données, le prétraitement et le choix du modèle sont essentiels. Pour cette démonstration,
    nous appliquons l'encodage des colonnes catégorielles et utilisons un modèle de régression linéaire simple. Nous 
    évaluons le modèle en calculant l'erreur quadratique moyenne et expérimentons en supprimant différentes fonctionnalités 
    pour améliorer les performances. Nous utilisons MLflow Tracking pour organiser les expériences, les exécutions, 
    les paramètres et les métriques liés à la formation du modèle.
```bash 
python .\src\MT_experiment.py 
```
l'interface .

Pour   lancer l'interface  utilisateur de MLflow , nous devons exécuter 
mlflow ui dans le répertoire où se trouve le répertoire mlruns.

```bash 
mlflow ui
``` 


Pour attacher des artefacts à des expériences dans MLflow

```bash 
python .\src\Artifacts_experiment.py
```
Actualisez la page MlFlow pour voir les résultats.

### Pour le travail collaboratif (à refaire)

```bash 
python .\src\CS_experiment.py
```

```bash 
mlflow server --host 0.0.0.0  --backend-store-uri sqlite:///data/mlflow/mlruns.db --default-artifact-root file:///data/mlflow/artifacts
```

2) Demo MLflow Models

    Dans cette démonstration, nous créons une classe personnalisée pour un modèle MLflow qui intègre l'étape de codage 
    one-hot dans la méthode de prédiction. Nous enregistrons le modèle et examinons comment cela se reflète dans 
    l'interface utilisateur de MLflow. La classe personnalisée hérite de PythonModel et implémente les méthodes
    load_context et predict. Nous utilisons également une méthode auxiliaire pour effectuer le codage one-hot. 
    En fin de compte, le modèle est enregistré avec le flavor pyfunc dans MLflow.


```bash 
mlflow server --host 0.0.0.0
```


```bash 
python .\src\MM_experiment.py
```
3) Batch and Real-time Prediction 

```bash 
python .\src\MM_make_test_data.py
```


```bash 
$Env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

```bash 
mlflow models predict -m runs:/key/custom_model -i test.csv --no-conda -t csv
```
 




