from src.model_training import *
from sklearn.model_selection import train_test_split
from src.belief_network import BeliefNetwork
from src.data_processing import preprocess_data, load_and_clean_data, discretize_features, get_label_encoders
from src.recommendation import *
from src.logger import Logger
from src.utils import *
import pickle
import sys

def load_data():
    try:
        df = pd.read_csv('data/processed_data.csv')
    except FileNotFoundError:
        df = load_and_clean_data('data/raw_data.csv')
        df = preprocess_data(df)
        df.to_csv('data/processed_data.csv', index=False)

    return df

def train_test_models_with_fixed_seeds(df, seed_models):
    models_trained = {}
    evaluation_models = {}
    start_models = init_models()
    param_grids = get_param_grids()
    x_train, x_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    for model_name, seed in seed_models.items():
        print(f'\nAddestramento del modello: {model_name}')
        models_trained[model_name] = train_model(
            model=start_models[model_name],
            param_grid=param_grids[model_name],
            x_train=x_train,
            y_train=y_train,
            random_state=seed,
            show_info=True,
            cv_external_folds=5,
            cv_internal_folds=5
        )
        
        print(f'Valutazione del modello: {model_name}')
        evaluation_models[model_name] = evaluate_model(models_trained[model_name]['model'], x_test, y_test)
        for key, value in evaluation_models[model_name].items():
            print(f'{key}:\n\t{value}')
            if key == 'confusion_matrix':
                plot_confusion_matrix(value, save_as_image=True, save_path=f'reports/img/confusion_matrix_{model_name}.jpg')
        
    return models_trained, evaluation_models

def train_belief_networks(df_train, df_test):
    networks = {
        'K2': BeliefNetwork(df_train),
        'BIC': BeliefNetwork(df_train),
        'EXP': BeliefNetwork(df_train)
    }
    
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
        ('Age', 'FCVC')
    ]

    # Train and evaluate networks
    for name, network in networks.items():
        if name == 'K2':
            network.learn_structure()
        elif name == 'BIC':
            network.learn_structure(scoring_method='bic')
        else:
            network.set_edges_model(expert_structure)
            
        network.learn_parameters()
        plot_BN(f'Belief Network {name}', network, save_as_image=True,
                save_path=f'reports/img/BN_{name}.jpg')
        network.save_to_file(f'reports/models/BN_{name}.pkl')
        
        acc = network.evaluate_bn(df_test)
        print(f'\nAccuratezza della Belief Network BN_{name}: {acc}')
        
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
    sys.stdout = Logger('run_doc.log')

    # Caricamento e pre-elaborazione del dataset
    print('Caricamento dataset...\n')
    df = load_data()
    
    # Visualizzazione struttura dataset
    print('Struttura dataset:')
    df.info()

    # Addestramento di modelli di classificazione con seed fissi per il fold esterno
    seed_models = {
        'decision_tree': 448,
        'kNN': 49,
        'logistic_regression': 976,
        'random_forest': 69
    }
    models_trained, evaluation_models = train_test_models_with_fixed_seeds(df, seed_models)

    # Visualizzazione dei risultati
    obesity_label = get_label_encoders()['Obesity'].get_categorical_values()
    plot_decision_tree(models_trained['decision_tree']['model'], feature_names= df.columns.drop(['Obesity']), class_names=obesity_label, save_as_image=True, save_path='reports/img/decision_tree.png')
    plot_comparision_models(evaluation_models, save_as_image=True, save_path='reports/img/comparision_models.jpg')

    # Salvataggio dei modelli
    with open('reports/models/models.pkl', 'wb') as f:
        pickle.dump(models_trained, f)

    # Operazioni di descretizzazione per l'apprendimento di BN
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
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Addestramento Belief Networks
    networks = train_belief_networks(df_train, df_test)
    
    # Generazione di raccomandazioni per essere normopeso
    generate_recommendations(networks['BIC'], 1)

if __name__ == "__main__":
    main()
