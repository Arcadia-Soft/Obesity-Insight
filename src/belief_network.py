from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, K2Score, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import copy
import pickle

class BeliefNetwork:
    """
    Classe per la creazione e l'apprendimento di una rete bayesiana.
    
    Attributes
    ----------
    data : pd.DataFrame
        Dataset di input.
    _creation_mode
        Modalità di creazione della struttura della rete bayesiana, tra:
        - 'expert': la struttura della rete è definita manualmente
        - 'k2': la struttura della rete è apprenduta tramite Hill Climbing Search con metrica di scoring K2
        - 'bic': la struttura della rete è apprenduta tramite Hill Climbing Search con metrica di scoring Bayesian Information Criterion
    """
    def __init__(self, data):
        self.data = data
        self._creation_mode = None
        self._edges = None
        self._model = None
        self._inference = None

    def learn_structure(self, scoring_method='k2'):
        """
        Apprendimento della struttura della rete bayesiana, tramite Hill Climbing Search.
        
        Parameters
        ----------
        scoring_method : str, default='k2'
            Il metodo di scoring da utilizzare per l'apprendimento della struttura, tra:
            - 'k2': metrica di scoring K2
            - 'bic': metrica di scoring Bayesian Information Criterion
            
        Raises
        ------
        ValueError
            Se viene fornito un metodo di scoring non presente tra quelli supportati.
        """
        estimator = HillClimbSearch(self.data)
        
        scoring_methods = {
            'k2': K2Score(self.data),
            'bic': BicScore(self.data)
        }
        
        if scoring_method not in scoring_methods:
            raise ValueError("Scoring method deve essere 'k2' o 'bic'")

        structure = estimator.estimate(scoring_method=scoring_methods[scoring_method], max_indegree=7, max_iter=int(1e3), show_progress=True)
        
        self._edges = structure.edges()
        self._creation_mode = scoring_method
        self._model = None
        self._inference = None
        
    def set_edges_model(self, edges):
        """
        Imposta manualmente le relazioni tra le variabili.
        
        Parameters
        ----------
        edges : ist[tuple[str, str]]
            Lista di tuple (nodo_padre, nodo_figlio)
        """
        self._creation_mode = 'expert'
        self._edges = edges
        self._model = None
        self._inference = None
        
    def get_edges(self):
        """
        Restituisce le relazioni tra le variabili.
        
        Returns
        -------
        Optional[list[tuple[str, str]]]
            Lista degli archi o None se la struttura non è definita
        """
        return copy.deepcopy(self._edges) if self._edges is not None else None

    def learn_parameters(self):
        """
        Apprende i parametri della rete usando Maximum Likelihood.
        
        Raises
        ------
        ValueError
            Se la struttura della rete non è stata definita
        """
        if self._edges is None:
            raise ValueError("Definire prima la struttura della rete")
            
        self._model = BayesianNetwork(self._edges)
        self._model.fit(
            self.data,
            estimator=MaximumLikelihoodEstimator
        )
        self._inference = VariableElimination(self._model)

    def predict(self, variables, evidence):
        """
        Predizione dei valori delle variabili date le prove.
        
        Parameters
        ----------
        variables : list
            Lista delle variabili da predire.
        evidence : dict
            Dizionario delle prove.

        Returns
        -------
        dict[str, int]
            Dizionario con le variabili come chiavi e i valori predetti come valori, per la combinazione di valori più probabile.
        
        Raises
        ------
        ValueError
            Se il modello non è stato addestrato
        """
        if self._inference is None:
            raise ValueError("Addestrare prima il modello")
            
        inference = self.infer(variables, evidence)
        max_prob_idx = inference.values.argmax()
        combination = np.unravel_index(max_prob_idx, inference.cardinality)
        return dict(zip(variables, combination))

    def infer(self, variables, evidence):
        """
        Calcola la distribuzione di probabilità condizionata delle variabili date le prove.
        
        Parameters
        ----------
        variables : list
            Lista delle variabili da predire.
        evidence : dict
            Dizionario delle prove.
        
        Returns
        -------
        pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor
            La distribuzione di probabilità condizionata delle variabili date le prove.
        
        Raises
        ------
        ValueError
            Se il modello non è stato addestrato
        """
        if self._inference is None:
            raise ValueError("Addestrare prima il modello")
            
        return self._inference.query(variables=variables, evidence=evidence)

    def get_creation_mode(self):
        """
        Restituisce il metodo di creazione della rete bayesiana.
        
        Returns
        -------
        str
            Il metodo di creazione della rete bayesiana.
        """
        return self._creation_mode

    def evaluate_bn(self, df_test):
        """
        Valutazione della accuratezza della rete bayesiana.
        
        Parameters
        ----------
        df_test : pandas.DataFrame
            DataFrame contenente i dati di test.

        Returns
        -------
        float
            Accuratezza della rete bayesiana.
        Raises
        ------
        ValueError
            Se il modello non è stato addestrato
        """
        if self._inference is None:
            raise ValueError("Addestrare prima il modello")

        target_variable = 'Obesity'
        y_true = df_test[target_variable].values
        predictions = []

        for _, row in df_test.iterrows():
            evidence = row.drop(target_variable).to_dict()
            pred = self.predict([target_variable], evidence)
            predictions.append(pred[target_variable])

        return balanced_accuracy_score(y_true, predictions)
        
    def save_to_file(self, path_file):
        """
        Salvataggio della rete bayesiana su file.
        
        Parameters
        ----------
        path_file : str
            Percorso dove salvare la rete bayesiana.
        """
        with open(path_file, 'wb') as out:
            pickle.dump(self, out)
            
    def load_from_file(self, path_file):
        """
        Caricamento della rete bayesiana da file.
        
        Parameters
        ----------
        path_file : str
            Percorso da cui caricare la rete bayesiana.
            
        Raises
        ------
        ValueError
            Se il file non contiene un oggetto BeliefNetwork valido
        FileNotFoundError
            Se il file non esiste
        """
        try:
            with open(path_file, 'rb') as f:
                loaded = pickle.load(f)
                if not isinstance(loaded, BeliefNetwork):
                    raise ValueError(
                        f"Il file {path_file} non contiene un oggetto BeliefNetwork"
                    )
                self.__dict__.update(loaded.__dict__)
        except FileNotFoundError:
            raise FileNotFoundError(f"File non trovato: {path_file}")
    
    def __copy__(self):
        new_instance = BeliefNetwork(self.data)
        new_instance._edges = self._edges
        new_instance._model = self._model
        new_instance._inference = self._inference
        new_instance._creation_mode = self._creation_mode
        return new_instance

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from data_processing import discretize_features
    import pandas as pd
    
    # Caricamento dei dati
    print("Caricamento dei dati...\n")
    df = pd.read_csv('data/processed_data.csv')
    
    # Discretizzazione delle features
    discretize_info = {
        'Age': {
            'bins': [14, 20, 30, 40, 50, 61],
            'labels': ['[14-20]', '(20-30]', '(30-40]', '(40-50]', '(50-61]']
        },
        'Weight': {
            'bins': [39, 65, 91, 117, 143, 173],
            'labels': ['[39-65]', '(65-91]', '(91-117]', '(117-143]', '(143-173]']
        },
        'Height': {
            'bins': [1.45, 1.54, 1.63, 1.72, 1.81, 1.90, 1.99],
            'labels': ['[1.45-1.54]', '(1.54-1.63]', '(1.63-1.72]', '(1.72-1.81]', '(1.81-1.90]', '(1.90-1.99]']
        },
        'BMI': {
            'bins': [0, 18.5, 24.9, 29.9, 34.9, 39.9, 60],
            'labels': ['<=18.5', '18.5-24-9', '25-29.9', '30-34.9', '35-39.9', '>=40']
        }
    }
    
    df, _ = discretize_features(df, discretize_info)
    
    # Separazione dei dati in training e test
    print("Separazione dei dati in training e test...\n")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Creazione della rete bayesiana
    print("Creazione della rete bayesiana...\n")
    BN = BeliefNetwork(df_train)
    
    # Apprendimento della struttura della rete bayesiana
    print("Apprendimento della struttura della rete bayesiana...\n")
    BN.learn_structure(scoring_method='k2')
    
    # Apprendimento dei parametri della rete bayesiana
    print("Apprendimento dei parametri della rete bayesiana...\n")
    BN.learn_parameters()
    
    # Valutazione della rete bayesiana
    print("Valutazione della rete bayesiana...\n")
    accuracy = BN.evaluate_bn(df_test)
    print(f"Accuratezza della rete bayesiana: {accuracy}\n")
    
    # Inferenza sulla rete bayesiana
    print("Inferenza sulla rete bayesiana...\n")
    evidence={'FAF':3, 'FCVC':3, 'Weight':1}
    inference = BN.infer(variables=['Obesity'], evidence=evidence)
    print(inference)
    
    # Salvataggio della rete bayesiana su file
    # print("Salvataggio della rete bayesiana su file...\n")
    # BN.save_to_file('BN.pkl')
    