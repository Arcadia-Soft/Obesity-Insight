# Obesity Insight

## Introduzione
Questo repository raccoglie il progetto realizzato per il corso di ICON 2024/2025 presso l'Università degli Studi di Bari Aldo Moro.

L'obiettivo principale è dimostrare le conoscenze acquisite durante il corso attraverso un'analisi dettagliata e applicazioni pratiche degli argomenti trattati su un dataset.

Obesity Insight è un progetto di machine learning che costruisce un modello predittivo per classificare i livelli di obesità negli individui provenienti da Colombia, Perù e Messico. Il modello utilizza dati raccolti attraverso questionari sulle abitudini alimentari e le condizioni fisiche.

Report del progetto: 
- Report [Markdown](reports/report.md)
- Report [PDF](reports/report.pdf)

Notebook del progetto: [Colab](https://colab.research.google.com/drive/1LSFmYoN1W2xFRYsA_VJK_Vsr5P-E_j11?usp=sharing)

## Obiettivi del Progetto
- Analizzare il dataset per identificare le caratteristiche chiave per la previsione del livello di obesità
- Addestrare e valutare vari modelli di Machine Learning per la classificazione del livello di obesità
- Selezionare il modello ottimale basato sulle prestazioni e capacità di generalizzazione
- Costruire una Rete di Credenze per modellare le relazioni tra caratteristiche e livelli di obesità

## Metodologia
1. **Raccolta e Preprocessamento dei Dati**
   - Raccolta del dataset
   - Pulizia dei dati
   - Gestione dei valori mancanti
   - Codifica delle variabili categoriche
   - Creazione di feature derivate per migliorare le prestazioni del modello

2. **Selezione e Addestramento del Modello**
   - Selezione di molteplici modelli di Machine Learning
   - Processo di addestramento
   - Ottimizzazione degli iperparametri usando la cross-validation

3. **Valutazione e Confronto dei Modelli**
   - Valutazione delle prestazioni usando metriche appropriate
   - Confronto e selezione del modello

4. **Costruzione Belief Network**
   - Modellazione delle relazioni tra le feature e livelli di obesità in maniera manuale e automatica tramite l'algoritmo di Hill Climbing
   - Apprendimento automatico delle distribuzioni di probabilità per ogni feature

5. **Raccomandazione e Conclusioni**
   - Modellazione delle relazioni tra le feature in vincoli che portino ad avere il livello di obesità `Normal_Weight`
   - Risoluzione del CSP creato con i vincoli definiti
   - Calcolo della probabilità di essere `Normal_Weight` con le raccomandazioni ottenute, tramite l'utilizzo di una Belief Network creata ed addestrata precedentemente
## Struttura del Progetto
```
obesity_insight/
│
├── models/
│   └── run_doc/
│
├── data/
│   ├── processed_data.csv
│   └── raw_data.csv
│
├── notebooks/
│   └── Obesity_Insight.ipynb
│
├── reports/
│   ├── img/
│   ├── models/
│   ├── documentazione.pdf
│   └── report.md
│
├── result_system/
│
├── src/
│   ├── belief_network.py
│   ├── data_processing.py
│   ├── logger.py
│   ├── model_training.py
│   ├── recommendation.py
│   └── utils.py
│
├── main.py
├── run_doc.py
├── LICENSE
└── README.md
```

- **data/**: Contiene i dati grezzi e il dataset processato.
- **models/**: Contiene i modelli addestrati e salvati.
- **notebooks/**: Contiene il notebook principale.
- **src/**: Contiene i moduli Python per il preprocessing, addestramento dei modelli, creazione della Belief Network e sistema di raccomandazione.
- **reports/**: Contiene il report e la documentazione del progetto.
- **result_system/**: Contiene i risultati ottenuti dall'esecuzione di `main.py`.

## Installazione
1. Clona il repository
```bash
git clone https://github.com/yourusername/obesity-insight.git
```

2. Sposta nella directory del progetto
```bash
cd obesity-insight
```

3. Crea un ambiente virtuale
```bash
python -m venv venv
```

4. Attiva l'ambiente virtuale
```bash
source venv/bin/activate
```

5. Installa le dipendenze richieste
```bash
pip install -r requirements.txt
```

## Utilizzo
Per ottenere i risultati visualizzati nel report/notebook, esegui:
```bash
python run_doc.py
```

Invece, per eseguire tutte le operazioni in sequenza senza valori predefiniti, esegui:
```bash
python main.py
```

Per testare in maniera separata i moduli:
- Operazioni di caricamento, preprocessing e discretizzazione:
    ```bash
    python -m src.preprocessing
    ```
- Addestramento e valutazione di apprendimento supervisionato:
    ```bash
    python -m src.model
    ```
- Addestramento e inferenza su una Belief Network:
    ```bash
    python -m src.belief_network
    ```
- Visualizzazione delle raccomandazioni che il sistema fornisce:
    ```bash
    python -m src.recommendation
    ```

## Contributori
- [Alessandro Pellegrino](https://github.com/ale-pell)
- [Kevin Saracino](https://github.com/kelvinsrcn)
