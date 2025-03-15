from src.model_training import *
from sklearn.model_selection import train_test_split
from src.belief_network import BeliefNetwork
from src.data_processing import preprocess_data, load_and_clean_data, discretize_features
from src.recommendation import *
from src.logger import Logger
from random import randint
from src.utils import plot_confusion_matrix, plot_BN, plot_comparision_models
import pickle
import sys

def load_data():
    try:
        df = load_and_clean_data('data/processed_data.csv')
    except FileNotFoundError:
        df = load_and_clean_data('data/raw_data.csv')
        df = preprocess_data(df)
        df.to_csv('data/processed_data.csv', index=False)

    return df

def train_and_test_all_models(df, test_size=0.2, max_attempts=100):
    models = init_models()
    param_grids = get_param_grids()
    successful_models = {}
    successful_evaluations = {}
    
    for model_name, model in models.items():
        print(f'\nTraining del modello {model_name}...')
        
        for attempt in range(max_attempts):
            try:
                random_state = randint(1, 1000)
                print(f'Suddivisione dei dati: Tentativo {attempt + 1}/{max_attempts} con random state: {random_state}')
                
                x_train, x_test, y_train, y_test = split_data(df, random_state, test_size=test_size)
                
                result = iterate_training(model_name, model, param_grids[model_name], x_train, y_train, show_info=True)
                
                print('\nValutazione sui dati di test:')
                
                evaluation = evaluate_model(result['model'], x_test, y_test)
                
                if evaluation['f1'] > result['validation_score']:
                    print(' - Il modello non è riuscito ad generalizzare sui dati di training, ritento con una nuova suddivisione del dataset...\n')
                else:
                    print(f' - Modello {model_name} ha generalizzato con successo!')
                    for key, value in evaluation.items():
                        print(f'{key}:\n\t{value}')
                    
                    plot_confusion_matrix(evaluation['confusion_matrix'], save_as_image=True, save_path=f'result_system/img/{model_name}_confusion_matrix.jpg')
                    
                    successful_models[model_name] = result['model']
                    successful_evaluations[model_name] = evaluation
                    break
            except ValueError as e:
                if attempt == max_attempts - 1:
                    print(f'Il modello {model_name} non è riuscito a generalizzare correttamente, con {max_attempts} tentativi.')
                continue
        
        if model_name not in successful_models:
            print(f'Il Modello {model_name} non è riuscito a generalizzare correttamente, con {max_attempts} tentativi.')
    
    if successful_evaluations:
        print('\nModelli che hanno generalizzato con successo:')
        for model_name in successful_evaluations:
            print(f' - {model_name}')
        plot_comparision_models(successful_evaluations, save_as_image=True, save_path='result_system/img/comparision_models.jpg')
    else:
        print('\nNessun modello è riuscito a generalizzare correttamente')
    
    return successful_models

def train_and_test_belief_networks(df_train, df_test):
    print('\nTraining Belief Network...')
    networks = {
        'K2': BeliefNetwork(df_train),
        'BIC': BeliefNetwork(df_train),
        'EXP': BeliefNetwork(df_train)
    }
    # Addestramento delle Belief Networks
        # apprendimento della struttura tramite HillClimbingSearch con metrica K2
    networks['K2'].learn_structure()
    networks['K2'].learn_parameters()
    plot_BN('Belief Network tramite K2', networks['K2'], save_as_image=True, 
            save_path='result_system/img/BN_K2.jpg')
    networks['K2'].save_to_file('result_system/BN_K2.pkl')
    
        # apprendimento della struttura tramite HillClimbingSearch con metrica BIC
    networks['BIC'].learn_structure(scoring_method='bic')
    networks['BIC'].learn_parameters()
    plot_BN('Belief Network tramite BIC', networks['BIC'], save_as_image=True, 
            save_path='result_system/img/BN_BIC.jpg')
    networks['BIC'].save_to_file('result_system/BN_BIC.pkl')
    
        # struttura della rete fornita dall'esperto
    expert_structure = [
        ('Weight', 'BMI'),
        ('Height', 'BMI'),
        ('family_history', 'Genetic_and_Behavioral_Risk'),
        ('FAVC', 'Genetic_and_Behavioral_Risk'),
        ('Genetic_and_Behavioral_Risk', 'Weight'),
        ('Genetic_and_Behavioral_Risk', 'Obesity'),
        ('BMI', 'Obesity'),
        ('CAEC', 'Obesity'),
        ('CH2O', 'Obesity'),
        ('FCVC', 'Obesity'),
        ('FAF', 'NCP'),
        ('CAEC', 'NCP'),
        ('family_history', 'NCP'),
        ('Age', 'TUE'),
        ('Age', 'CAEC'),
        ('Age', 'CH2O'),
        ('NCP', 'CH2O'),
        ('TUE', 'FAF'),
        ('Age', 'FAF'),
        ('Age', 'FCVC'),
    ]
    
    networks['EXP'].set_edges_model(expert_structure)
    networks['EXP'].learn_parameters()
    plot_BN('Belief Network Expert', networks['EXP'], save_as_image=True, 
            save_path='result_system/img/BN_EXP.jpg')
    networks['EXP'].save_to_file('result_system/BN_EXP.pkl')
    
    print('\nTesting Belief Network...')
    accuracies = {
        'K2': networks['K2'].evaluate_bn(df_test),
        'BIC': networks['BIC'].evaluate_bn(df_test),
        'EXP': networks['EXP'].evaluate_bn(df_test)
    }
    
    for name, acc in accuracies.items():
        print(f" - Accuratezza della Belief Network BN_{name}: {acc}")
    
    return networks

def generate_recommendations(BN, family_history):
    print('\nGenerazione di raccomandazioni di abitudini per essere normopeso...')
    recommendations = CSP_Recommendation(family_history)
    solutions = recommendations.get_n_best_solutions(5, BN)
    
    if solutions:
        print('Raccomandazioni per essere normopeso:')
        for i, solution in enumerate(solutions, start=1):
            print(f'Raccomandazione n.ro {i}')
            solution.pop('Obesity')
            probability = BN.infer(['Obesity'], solution).values[1]
            solution.pop('family_history')
            final_report = generate_report(solution)
            print(final_report)
            print(f'Probabilità di essere normopeso (Secondo BN_BIC): {(probability * 100):.2f}%\n')
    else:
        print('Nessuna raccomandazione possibile.')

def main():
    sys.stdout = Logger('main.log')

    # Caricamento e pre-elaborazione del dataset
    print('Caricamento dataset...\n')
    df = load_data()
    
    # Visualizzazione struttura dataset
    print('Struttura dataset:')
    df.info()
    
    # Addestramento e valutazione dei modelli di classificazione
    models = train_and_test_all_models(df)
    
    # Salvataggio dei modelli di classificazione che sono riusciti a generalizzare
    with open('models/models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Preparazione dei dati per l'addestramento della rete bayesiana
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
    
    # Separazione del dataset in train e test per l'addestramento della rete bayesiana
    random_state = randint(1, 1000)
    print(f'\nSeparazione dati con random state: {random_state}')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
    
    # Addestramento e valutazione delle reti bayesiane
    networks = train_and_test_belief_networks(df_train, df_test)
    
    # Generazione di raccomandazioni per essere normopeso
    generate_recommendations(networks['BIC'], 1)

if __name__ == "__main__":
    main()
