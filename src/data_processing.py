import pandas as pd
import numpy as np

def load_and_clean_data(path):
    """
    Carica e pulisce il dataset.
    
    Parameters
    ----------
    path : str
        Percorso del file CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame senza righe duplicate e valori nulli.
    
    Raises
    ------
    FileNotFoundError
        Se il file non esiste
    """
    df = pd.read_csv(path)
    
    # Rimozione duplicati
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        print("Duplicati rimossi")
    
    # Rimozione delle righe con valori nulli
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)
        print("Righe con valori nulli rimosse")
    
    return df

def preprocess_data(df):
    """
    Preprocessa il dataset, arrotondando i valori numerici e mappando le variabili categoriche.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame da preprocessare.
    
    Returns
    -------
    df_processed : pd.DataFrame
        DataFrame preprocessato.
    """
    df_processed = df.copy()
    
    # Arrotondamento dei valori delle colonne
    cols_numeric = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    df_processed[cols_numeric] = np.ceil(df[cols_numeric]).astype(pd.Int64Dtype())
    
    # Rimozione feature non rilevanti
    irrelevant_features = ['Gender', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    df_processed.drop(columns=irrelevant_features, inplace=True)

    # Encoding feature categoriche
    label_encoders = get_label_encoders()
    categorical_columns = ['family_history', 'CAEC', 'Obesity', 'FAVC']
    
    for col in categorical_columns:
        df_processed[col] = label_encoders[col].transform(df_processed[col])

    # Feature engineering
    df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)
    df_processed['Genetic_and_Behavioral_Risk'] = (df_processed['family_history'] + df_processed['FAVC']) / 2
    
    return df_processed

def get_label_encoders():
    """
    Crea gli encoder per le variabili categoriche.
    
    Returns
    -------
    dict[str, data_processing.Encoder]
        Dizionario degli encoder per ogni feature categorica
    """
    label_encoders = {}
    yes_no_response = ['no', 'yes']
    responses = {
        'family_history': yes_no_response,
        'FAVC': yes_no_response,
        'FCVC': ['Never', 'Sometimes', 'Always'],
        'NCP': ['One ', 'Two', 'Three', 'More than three'],
        'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
        'CH2O': ['less than a liter', 'between 1 and 2 L', 'more than 2 L'],
        'FAF': ['I do not have', '1 or 2 days', '2 or 4 days', '4 or 5 days'],
        'TUE': ['between 0 and 2 hours', 'between 3 and 5 hours', 'more than 5 hours'],
        'Obesity': [ 'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    }

    label_encoders = {}
    for col in get_categorical_columns():
        start_value = 1 if col in ['FCVC', 'NCP', 'CH2O'] else 0
        encoder = Encoder(start_value)
        encoder.fit(responses[col])
        label_encoders[col] = encoder
    
    return label_encoders

def get_target_column():
    """
    Restituisce la feature target del dataset
    
    Returns
    -------
    str
        Nome della feature target
    """
    return 'Obesity'

def get_categorical_columns():
    """
    Restituisce le feature categoriche.
    
    Returns
    -------
    list[str]
        Lista delle colonne categoriche
    """
    return ['family_history', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'Obesity']

def get_numerical_columns():
    """
    Restituisce le feature numeriche.
    
    Returns
    -------
    list[str]
        Lista delle colonne numeriche
    """
    return ['Age', 'Height', 'Weight', 'BMI', 'Genetic_and_Behavioral_Risk']

def discretize_features(df, discretize_info):
    """
    Discretizza le feature numeriche specificate.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame da discretizzare.
    discretize_info : dict
        Configurazione per la discretizzazione, un dizionario dove le chiavi sono i nomi delle colonne e i valori sono dizionari con 'bins'<br>
        e 'labels' per la discretizzazione. Esempio: {'nome_colonna': {'bins': [0, 1, 2], 'labels': ['basso', 'alto']}}

    Returns
    -------
    df : pandas.DataFrame
        Il DataFrame con le feature discretizzate.
    disc_label_encoders : dict
        Dizionario degli encoder delle etichette per le feature discretizzate.
    """
    df_copy = df.copy()
    disc_label_encoders = {}
    
    for column, info in discretize_info.items():
        df_copy, encoder = discretize_feature(
            df_copy, column, info['bins'], info['labels']
        )
        disc_label_encoders[column] = encoder
    
    return df_copy, disc_label_encoders

def discretize_feature(df, column, bins, labels):
    """
    Discretizza una singola feature

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenente la feature da discretizzare.
    column : str
        Nome della feature da discretizzare.
    bins : list
        Limiti degli intervalli per la discretizzazione.
    labels : list[str]
        Etichette per gli intervalli.

    Returns
    -------
    df : pandas.DataFrame
        Il DataFrame con la feature discretizzata.
    encoder : Encoder
        L'encoder della feature discretizzata.
    """
    df_copy = df.copy()
    df_copy[column] = pd.cut(df_copy[column], bins=bins, labels=labels, include_lowest=True)
    
    encoder = Encoder()
    df_copy[column] = encoder.fit_transform(df_copy[column], labels)
    
    return df_copy, encoder

class Encoder:
    """
    Encoder per la conversione di valori categorici in numerici.
    
    Parameters
    ----------
    _start_value : int, optional
        Valore di partenza per il mapping.
    
    Attributes
    ----------
    mapping : dict
        Dizionario che mappa i valori originali a valori interi.
    """
    def __init__(self, start_value=0):
        self.mapping = {}
        self._start_value = start_value

    def fit(self, array):
        """
        Crea il mapping dei valori categorici a valori interi.
        
        Parameters
        ----------
        array : list, np.ndarray
            Lista o array di valori su cui effettuare il mapping.
        """
        unique_values = pd.Series(array).unique()
        self.mapping = {value: idx + self._start_value for idx, value in enumerate(unique_values)}

    def transform(self, array):
        """
        Applica la trasformazione ai valori.
        
        Parameters
        ----------
        array : list, np.ndarray
            Valori da trasformare.
        
        Returns
        -------
        np.ndarray
            Valori trasformati.
            
        Raises
        ------
        ValueError
            Se il mapping non è stato creato (i.e., `fit` non è stato chiamato).
        """
        if not self.mapping:
            raise ValueError("Eseguire prima il metodo 'fit'")
            
        return np.array([self.mapping[value] for value in array])
    
    def get_categorical_values(self):
        """
        Restituisce i valori unici utilizzati per il mapping.

        Returns
        -------
        list
            Lista di valori unici.
        """
        return list(self.mapping.keys())
    
    def fit_transform(self, array, values=None):
        """
        Esegue fit e transform in un'unica chiamata.
        
        Parameters
        ----------
        array : list, np.ndarray
            Valori da trasformare.
        values :  list, optional
            Valori predefiniti per il mapping
        
        Returns
        -------
        np.ndarray
            Valori trasformati.
        """
        self.fit(values if values is not None else array)
        return self.transform(array)

    def inverse_transform(self, values):
        """
        Converte i valori numerici nelle categorie originali.
        
        Parameters
        ----------
        values : list, int, np.ndarray
            Valore/i da convertire.
        
        Returns
        -------
        np.ndarray, type()
            Categoria/e convertita/e.
        
        Raises
        ------
        KeyError
            Se uno valore non è presente nel mapping.
        """
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        
        if isinstance(values, (int, np.integer)):
            if values not in reverse_mapping:
                raise KeyError(f"Valore {values} non trovato nel mapping")
            return reverse_mapping[values]
        
        try:
            return np.array([reverse_mapping[value] for value in values])
        except KeyError as e:
            raise KeyError(f"Valore {e.args[0]} non trovato nel mapping")

    def __str__(self):
        return '\n'.join(f"{key} -> {value}" for key, value in self.mapping.items())
    
    def toFormattedString(self, spaces=1):
        """
        Rappresentazione testuale formattata del mapping.

        Parameters
        ----------
        spaces : int, default 1
            Numero di tabulazioni.

        Returns
        -------
        str
            Mapping formattato con tabulazioni
        """
        space = '\x20' * spaces
        return ''.join(f'{space}- {key} -> {value}\n' for key, value in self.mapping.items())

if __name__ == "__main__":
    # Esempio di utilizzo
    #   dataset usato: https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction/data
    
    # Caricamento
    df = pd.read_csv('data/raw_data.csv')
    
    # Preprocessing
    df_processed = preprocess_data(df)
    
    # Salvataggio del dataset preprocessato
    # df_processed.to_csv('data/processed_data.csv', index=False)
    
    # Visualizzazione delle feature categoriche
    categorical_columns = get_categorical_columns()
    print(f"Feature categoriche: {categorical_columns}\n")
    
    # Visualizzazione delle feature numeriche
    numerical_columns = get_numerical_columns()
    print(f"Feature numeriche: {numerical_columns}\n")
    
    # Visualizzazione della feature target
    target_columns = get_target_column()
    print(f"Feature target: {target_columns}\n")
    
    # Ottenimento degli encoder per le feature categoriche
    encoders = get_label_encoders()
    
    # Visualizzazione mapping per le feature categoriche
    for column in encoders:
        print(f"Mapping per {column}:")
        print(encoders[column].toFormattedString(spaces=4))
    
    # Stampa del dataset preprocessato
    print("Dataset preprocessato:")
    print(df_processed.head(), end='\n\n')
    
    # Discretizzazione di feature numeriche
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
    
    df, disc_label_encoders = discretize_features(df_processed, discretize_info)
    
    # Visualizzazione del dataset discretizzato
    print("Dataset discretizzato:")
    print(df.head(), end='\n\n')
    
    # Visualizzazione degli encoder per le feature discretizzate
    for column in disc_label_encoders:
        print(f"Encoder per {column}:")
        print(disc_label_encoders[column].toFormattedString(spaces=4))
    
    # Salvataggio del dataset discretizzato
    # df.to_csv('data/discretized_processed_data.csv', index=False)    