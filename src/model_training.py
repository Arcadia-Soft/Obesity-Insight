from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, make_scorer, confusion_matrix
from random import randint
import pandas as pd
import numpy as np

def split_data(df, random_state, test_size = 0.2):
    """
    Suddivisione del dataset in set di training e test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame da suddividere.
    random_state : int
        Seed per la generazione casuale.
    test_size : float, default=0.2
        Percentuale di dati di test (0.0-1.0)

    Returns
    -------
    x_train : pd.DataFrame
        Dataset di training.
    x_test : pd.DataFrame
        Dataset di test.
    y_train : pd.DataFrame
        Target di training.
    y_test : pd.DataFrame
        Target di test.
        
    Raises
    ------
    ValueError
        Se il valore di test_size non è compreso tra 0.0 e 1.0.
    """
    if test_size < 0 or test_size > 1:
        raise ValueError('test_size deve essere un valore compreso tra 0.0 e 1.0')
    x = df.drop('Obesity', axis=1)
    y = df['Obesity']
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def get_models_names():
    """
    Restituisce una lista dei nomi dei modelli di machine learning disponibili.
    
    Returns
    -------
    list[str]
        Lista dei nomi dei modelli supportati.
    """
    return ['decision_tree', 'kNN', 'logistic_regression', 'random_forest']

def get_param_grids():
    """
    Restituisce un dizionario contenente i grid di parametri per vari modelli di machine learning.

    Il dizionario include i grid di parametri per i seguenti modelli:
    - Decision Tree
    - k-Nearest Neighbors (kNN)
    - Logistic Regression
    - Random Forest

    Ogni grid di parametri è un dizionario in cui le chiavi sono i parametri e i valori sono le possibili scelte per ogni parametro.

    Returns
    -------
    dict[str, dict[str, list[Any]]]
        Un dizionario avente come chiavi i nomi dei modelli e come valori i grid di parametri per ogni modello.
    """
    return {
        'decision_tree': {
            'criterion': ['log_loss', 'entropy'],
            'max_depth': [5, 10, 15],
            'random_state': [42]
        },
        'kNN': {
            'weights': ['uniform', 'distance'],
            'n_neighbors': [5, 7, 10]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'newton-cg', 'sag'],
            'random_state': [42]
        },
        'random_forest': {
            'n_estimators': [25, 50, 100],
            'criterion': ['log_loss', 'entropy'],
            'max_depth': [5, 10, 15],
            'random_state': [42]
        }
    }

def init_models():
    """
    Restituisce un dizionario contenente i modelli iniziallizati di machine learning disponibili.

    Returns
    -------
    dict[str, sklearn.base.BaseEstimator]
        Un dizionario avente come chiavi i nomi dei modelli e come valori i modelli iniziallizati.        
    """
    return {
        'decision_tree': DecisionTreeClassifier(),
        'kNN': KNeighborsClassifier(),
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier()
    }

def train_model(model, param_grid, x_train, y_train, cv_external_folds=5, cv_internal_folds=5, show_info=False, random_state=42):
    """
    Addestramento di `model`, andando a cercare i parametri ottimali per esso, tra quelli specificati in param_grid.<br>
    Utilizza una cross-validation esterna per la ricerca modelli ottimali che riescono a generalizzare senza andare in overfitting,<br>
    mentre una cross-validation interna per la ricerca di parametri ottimali che riescono a generalizzare senza andare in overfitting.<br>
    
    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Modello da addestrare   
    param_grid : dict
        Dizionario contenente i parametri da testare
    x_train : pd.DataFrame
        Features di input per il training
    y_train : pd.DataFrame
        Feature Target per il training
    cv_external_fold : int, default=5
        Numero di fold per la cross-validation esterna
    cv_internal_fold : int, default=5
        Numero di fold per la cross-validation nella ricerca dei parametri
    show_info bool, default=False
        Se True, mostra informazioni dettagliate durante il training
    random_state : int, default=42
        Seed per la separazione dei dati di training in fold, per la cross-validation esterna
    
    Returns
    -------
    result : dict[str, Any]
        Dizionario contenente il risultato del training, contenente i seguenti campi:
        - model: il modello addestrato con i parametri ottimali
        - params: i parametri ottimali trovati
        - validation_score: il punteggio F1-score, mediato sul numero totale di instanze vere per ogni classe,<br>
        cioè usando `f1_score(y_true, y_pred, average='weighted')`

    Raises
    ------
    ValueError
        Il modello non riesce a generalizzare, cioè va in overfitting sui validation set della cross-validation esterna
    """
    overfitting = True
    best_result = None
    
    scorer = make_scorer(f1_score, average='weighted')
    kf = KFold(n_splits=cv_internal_folds)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring=scorer, n_jobs=-1)
    
    external_folds = StratifiedShuffleSplit(n_splits=cv_external_folds, test_size=0.2, random_state=random_state)
    
    for i, (train_index, valid_index) in enumerate(external_folds.split(x_train, y_train)):
        if show_info:
            print(f'[CV EXTERNAL] starting fold {i + 1}/{cv_external_folds}')

        x_train_fold, x_valid_fold = x_train.iloc[train_index], x_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        grid_search.fit(x_train_fold, y_train_fold)
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        if show_info:
            print(f'[CV INTERNAL] done - best score: {best_score}, best params: {best_params}')
        
        best_estimator = grid_search.best_estimator_
        valid_score = scorer(best_estimator, x_valid_fold, y_valid_fold)
        
        if show_info:
            print(f'[CV EXTERNAL] fold {i + 1}/{cv_external_folds} done - score: {valid_score}')
        
        if valid_score < best_score:
            if best_result is None or valid_score > best_result['validation_score']:
                overfitting = False
                best_result = {
                    'model': best_estimator,
                    'params': best_params,
                    'validation_score': valid_score
                }

    if overfitting:
        print('Il modello non è in grado di generalizzare.')
        raise ValueError('Il modello non è in grado di generalizzare.')
    
    if show_info:
        print(f'[FINAL] best score: {best_result["validation_score"]}, best params: {best_result["params"]}\n')
    
    best_result['random_state'] = random_state
    
    return best_result

def iterate_training(model_name, model, param_grid, x_train, y_train, cv_external_folds=5, cv_internal_folds=5, show_info=False, max_iterations=100):
    """
    Esegue il training del modello `model` con i parametri ottimali, cercando i parametri ottimali per esso, tra quelli specificati in param_grid.<br>
    Utilizza una cross-validation esterna per la ricerca modelli ottimali che riescono a generalizzare senza andare in overfitting,<br>
    mentre una cross-validation interna per la ricerca di parametri ottimali che riescono a generalizzare senza andare in overfitting.<br>
    
    Parameters
    ----------
    model_name : str
        Nome del modello
    model : sklearn.base.BaseEstimator
        Modello da addestrare
    param_grid : dict
        Dizionario contenente i parametri da testare
    x_train : pd.DataFrame
        Features di input per il training
    y_train : pd.DataFrame
        Feature Target per il training
    cv_external_fold : int, default=5
        Numero di fold per la cross-validation esterna
    cv_internal_fold : int, default=5
        Numero di fold per la cross-validation nella ricerca dei parametri
    show_info bool, default=False
        Se True, mostra informazioni dettagliate durante il training
    max_iterations : int, default=100
        Numero massimo di iterazioni per trovare i parametri ottimali
    
    Returns
    -------
    result : dict[str, Any]
        Dizionario contenente il risultato del training, contenente i seguenti campi:
        - model: il modello addestrato con i parametri ottimali
        - params: i parametri ottimali trovati
        - validation_score: il punteggio F1-score, mediato sul numero totale di instanze vere per ogni classe,<br>
        cioè usando `f1_score(y_true, y_pred, average='weighted')`
    
    Raises
    ------
    ValueError
        Il modello non riesce a generalizzare sui i dati di training, cioè va in overfitting
    """
    for i in range(max_iterations):
        try:
            seed = randint(0, 1000)
            if show_info:
                print(f'Iteration {i + 1}/{max_iterations} - {model_name} with random state: {seed}')
            return train_model(model, param_grid, x_train, y_train, cv_external_folds, cv_internal_folds, show_info, random_state=seed)
        except ValueError:
            if i == max_iterations - 1:
                raise ValueError('Il modello non è in grado di generalizzare.')
            continue

def train_models(x_train, y_train, cv_external_folds=5, cv_internal_folds=5, show_info=False, max_iterations=100):
    """
    Addestramento di tutti i modelli supporati, tramite la funzione `train_model(...)`

    Parameters
    ----------
    x_train : pd.DataFrame
        Features di input per il training
    y_train : pd.DataFrame
        Feature Target per il training
    param_grid : dict
        Dizionario contenente i parametri da testare
    cv_external_fold : int, default=5
        Numero di fold per la cross-validation esterna
    cv_parameter_fold : int, default=5
        Numero di fold per la cross-validation nella ricerca dei parametri
    show_info bool, default=False
        In base al valore, se True mostra informazioni durante il training
    max_iterations : int, default=1000
        Numero massimo di iterazioni per il training di un modello

    Returns
    -------
    results : dict
        Dizionario contenente i risultati di training dei modelli supportati, ogni valore associata ad un modello è un dizionario<br>
        contenente i seguenti campi:
        - model: il modello addestrato con i parametri ottimali
        - params: i parametri ottimali trovati
        - validation_score: il punteggio F1-score, mediato sul numero totale di instanze vere per ogni classe,<br>
        cioè usando `f1_score(y_true, y_pred, average='weighted')`
    overfitting_models : list
        Lista dei modelli che non sono in grado di generalizzare sui dati di training
    """
    models = init_models()
    param_grids = get_param_grids()
    results = {}
    
    overfitting_models = []
    
    for model_name, model in models.items():
        try:
            results[model_name] = iterate_training(model_name, model, param_grids[model_name], x_train, y_train, cv_external_folds, cv_internal_folds, show_info)
        except ValueError:
            overfitting_models.append(model_name)

    return results, overfitting_models
    
def evaluate_model(model, x_test, y_test):
    """
    Valutazione del modello sui dati di test
    
    Parameters
    ----------
    model: estimator object
        Modello di machine learning
    x_test : pd.DataFrame
        Features di input per la valutazione.
    y_test : pd.DataFrame
        Feature Target per la valutazione.
        
    Returns
    -------
    evaluation : dict[str, Any]
        Dizionario contenente le metriche di valutazione, con i seguenti campi:
        - report: report di classificazione del modello.
        - accuracy: accuratezza bilanciata, per evitare l'effetto della disuguaglianza di esempi per ogni classe.
        - f1: F1-score, mediato sul numero totale di instanze vere per ogni classe,<br>
        cioè usando `f1_score(y_true, y_pred, average='weighted')`
        - confusion_matrix: matrice di confusione delle predizioni.
    """
    y_pred = model.predict(x_test)
    return {
        'report': classification_report(y_test, y_pred),
        'accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
        
def evaluate_models(models, x_test, y_test):
    """
    Valutazione di tutti i modelli supportati, tramite la funzione `evaluate_model(...)`

    Parameters
    ----------
    models : list
        Dizionario contenente i modelli da valutare, con chiave il nome del modello e valore il modello addestrato
    x_test : pd.DataFrame
        Features di input per la valutazione.
    y_test : pd.DataFrame
        Feature Target per la valutazione.

    Returns
    -------
    results : dict[str, dict[str, Any]]
        Dizionario contenente i risultati di valutazione per ogni modello, per ogni modello si avrà un dizionario con i seguenti campi:
        - report: report di classificazione del modello.
        - accuracy: accuratezza bilanciata, per evitare l'effetto della disuguaglianza di esempi per ogni classe.
        - f1: F1-score, mediato sul numero totale di instanze vere per ogni classe,<br>
        cioè usando `f1_score(y_true, y_pred, average='weighted')`
        - confusion_matrix: matrice di confusione delle predizioni.
    """
    return {
        model_name: evaluate_model(model, x_test, y_test)
        for model_name, model in models.items()
    }

if __name__ == '__main__':
    # Esempio di utilizzo
    #   dataset usato: https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction/data processato
    
    # Caricamento dei dati
    print('Caricamento dei dati...\n')
    df = pd.read_csv('data/processed_data.csv')
    
    # Split dei dati in training e test
    print('Split dei dati in training e test...\n')
    x_train, x_test, y_train, y_test = split_data(df, 42)
    
    # Addestramento dei modelli
    print('Addestramento dei modelli...\n')
    results_train, overfitting_models = train_models(x_train, y_train, cv_external_folds=5, cv_internal_folds=5, show_info=True)
    models = {
        model_name: model['model']
        for model_name, model in results_train.items() if model_name not in overfitting_models
    }
    
    # Valutazione dei modelli
    print('Valutazione dei modelli...\n')
    results = evaluate_models(models, x_test, y_test)
    
    # Stampa dei risultati
    print('Risultati:')
    for model_name, evaluation in results.items():
        print(f'{model_name}:')
        
        if model_name in overfitting_models:
            print('Il modello non è in grado di generalizzare sui dati di training\n')
        else:
            print(f' - Parametri ottimali:\n    {results_train[model_name]["params"]}\n')
            print(f' - Validation score (F1):\n    {results_train[model_name]["validation_score"]}\n')
            
            if evaluation['f1'] > results_train[model_name]['validation_score']:
                print(f' - Il modello è in overfitting sui dati di test, con f1: {evaluation["f1"]}\n')
            else:
                print(' - Il modello è in grado di generalizzare sui dati di test\n')
                for metric_name, value in evaluation.items():
                    print(f' - {metric_name}: \n{value}')
        print()