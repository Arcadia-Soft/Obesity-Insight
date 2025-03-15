from constraint import Problem
from src.data_processing import get_label_encoders

class CSP_Recommendation:
    """
    Sistema di raccomandazione basato su Constraint Satisfaction Problem.
    """    
    def __init__(self, family_history):
        self._problem = Problem()
        self._obesity_goal_level = 1

        # Definizione variabili e domini
        variables = [
            'family_history', 'FAVC', 'FCVC', 'NCP', 'CAEC',
            'CH2O', 'FAF', 'TUE', 'Obesity'
        ]
        
        domains = {
            'family_history': [family_history],
            'FAVC': [0, 1],
            'FCVC': [1, 2, 3],
            'NCP': [1, 2, 3, 4],
            'CAEC': [0, 1, 2, 3],
            'CH2O': [1, 2, 3],
            'FAF': [0, 1, 2, 3],
            'TUE': [0, 1, 2],
            'Obesity': [0, 1, 2, 3, 4, 5, 6]
        }

        for var in variables:
            self._problem.addVariable(var, domains[var])

        # Definizione vincoli
        self._add_constraints()

    def _add_constraints(self):
        """
        Aggiunge i vincoli al problema CSP.
        """
        constraints = {
            ('FAVC', 'FCVC'): lambda favc, fcvc: fcvc in [2, 3] if favc == 1 else True,
            
            ('NCP', 'CH2O'): lambda ncp, ch2o: (
                ch2o > 2 if ncp in [3, 4] else ch2o > 1
            ),
            
            ('NCP', 'CAEC'): lambda ncp, caec: (
                caec in [0, 1] if ncp in [3, 4]
                else caec in [2, 3] if ncp == 1
                else caec in [0, 1, 2]
            ),
            
            ('TUE', 'FAF'): lambda tue, faf: (
                faf >= 2 if tue > 1 else faf >= 1
            ),
            
            ('CAEC', 'CH2O'): lambda caec, ch2o: (
                ch2o in [2, 3] if caec in [2, 3] else ch2o >= 1
            ),
            
            ('FAF', 'CH2O'): lambda faf, ch2o: (
                ch2o in [2, 3] if faf in [2, 3] else True
            ),
            
            ('CAEC', 'FAVC'): lambda caec, favc: (
                favc == 0 if caec == 3 else True
            ),
            
            ('family_history', 'FAF'): lambda fh, faf: (
                faf >= 2 if fh == 1 else True
            ),
            
            ('NCP',): lambda ncp: ncp > 1
        }
        
        for vars, constraint in constraints.items():
            self._problem.addConstraint(constraint, vars)

        # Vincolo obiettivo
        self._problem.addConstraint(
            lambda obesity: obesity == self._obesity_goal_level,
            ['Obesity']
        )
    
    def solve_csp(self):
        """
        Risolve il problema CSP.
        
        Returns
        -------
        list[dict[str, int]]
            Lista delle soluzioni trovate
        """
        return self._problem.getSolutions()
    
    def get_n_best_solutions(self, n, belief_network):
        """
        Trova le n migliori soluzioni secondo la rete bayesiana.
        
        Parameters
        ----------
        n : int
            Numero di soluzioni da restituire
        belief_network : BeliefNetwork
            Rete bayesiana per valutare le soluzioni
            
        Returns
        -------
        optional[list[dict[str, int]]]
            Le n migliori soluzioni ordinate per probabilità
        """
        solutions = self.solve_csp()
        if not solutions:
            return None
            
        return sorted(
            solutions,
            key=lambda sol: belief_network.infer(
                variables=['Obesity'],
                evidence={k: v for k, v in sol.items() if k != 'Obesity'}
            ).values[self._obesity_goal_level],
            reverse=True
        )[:n]

def generate_report(features):
    """
    Genera un report leggibile delle raccomandazioni.
    
    Parameters
    ----------
    features : dict[str, int]
        Dizionario delle feature e relativi valori
        
    Returns
    -------
    str
        Report formattato delle raccomandazioni
    """
    label_encoders = get_label_encoders()
    return "\n".join([
        f" - Si consiglia per {feature}: {label_encoders[feature].inverse_transform([val])[0]}"
        for feature, val in features.items()
    ])

if __name__ == "__main__":
    
    # Esempio di utilizzo
    from sklearn.model_selection import train_test_split
    from src.data_processing import discretize_features
    from src.belief_network import BeliefNetwork
    import pandas as pd
    
    # Apprendimento BN per la valutazione delle raccomandazioni
    print("Apprendimento BN per la valutazione delle raccomandazioni...\n")
    df = pd.read_csv('data/processed_data.csv')
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
    BN = BeliefNetwork(df_train)
    BN.learn_structure(scoring_method='bic')
    BN.learn_parameters()
    
    # Soluzione CSP
    family_history = 1
    csp_recommendation = CSP_Recommendation(family_history)
    solutions = csp_recommendation.get_n_best_solutions(5, BN)
    
    if solutions:
        print("Raccomandazioni per essere normopeso:")
        for i, solution in enumerate(solutions, start=1):
            print(f"Raccomandazione n.ro {i}")
            solution.pop('Obesity')
            probability = BN.infer(['Obesity'], solution).values[1]
            solution.pop('family_history')
            final_report = generate_report(solution)
            print(final_report)
            print(f"Probabilità di essere normopeso (Secondo BN_BIC): {(probability*100):.2f}%\n")
    else:
        print("Nessuna raccomandazione possibile.")