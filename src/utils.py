from sklearn.tree import plot_tree
from src.belief_network import BeliefNetwork
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

def plot_decision_tree(model, feature_names, class_names, save_as_image=False, save_path='reports/img/decision_tree.jpg'):
    """
    Plotta l'albero di decisione.
    
    Parameters
    ----------
    model : sklearn.tree.DecisionTreeClassifier
        Il modello DecisionTreeClassifier da visualizzare.
    feature_names : list
        Lista dei nomi delle features.
    class_names : list
        Lista dei nomi delle classi.
    save_as_image : bool, default=False
        Se True, salva l'immagine del grafico.
    save_path : str, default=None
        Percorso di salvataggio dell'immagine. Se None, viene salvata in 'reports/decision_tree.jpg'.
    """
    plt.figure(figsize=(120, 60))
    plot_tree(model, feature_names = feature_names, class_names = class_names, filled=True, fontsize=15)
    
    if save_as_image:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()

def plot_comparision_models(result_models, save_as_image=False, save_path='reports/img/comparision_models.jpg'):
    """
    Crea un grafico a barre per confrontare le prestazioni dei modelli ottimizzati. Usando le metriche di accuratezza e F1-score.
    
    Parameters
    ----------
    result_models : dict
        Dizionario contenente i risultati di valutazione per ogni modelli con le metriche di accuratezza e F1-score.
    """
    results_df = pd.DataFrame(result_models).T
    plt.figure(figsize=(10, 7))
    bar_width = 0.4
    index = np.arange(len(results_df))
    
    # Plot bars
    accuracy_bars = plt.bar(index, results_df['accuracy'], width=bar_width, label='Accuracy', alpha=0.7, color='blue')
    f1_bars = plt.bar(index + bar_width, results_df['f1'], width=bar_width, label='F1 Score',alpha=0.7, color='orange')

    # Add value labels
    for bars, values in [(accuracy_bars, results_df['accuracy']), (f1_bars, results_df['f1'])]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{value:.6f}', ha='center', va='bottom')

    plt.title('Confronto delle Prestazioni dei Modelli Ottimizzati')
    plt.ylabel('Score')
    plt.xlabel('Modello')
    plt.xticks(ticks=[p + bar_width / 2  for p in index], labels=[model for model in result_models])
    plt.legend(title='Metrica', loc='best')
    plt.tight_layout()
    
    if save_as_image:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()

def plot_BN(title, model, save_as_image=False, save_path=None):
    """
    Crea un grafico per visualizzare la struttura del modello bayesiano. appreso
    
    Parameters
    ----------
    title : str
        Titolo del grafico
    model : src.BayesianNetwork
        Modello bayesiano appreso
    save_as_image : bool
        Se True, salva il grafico come immagine
    save_path : str, optional
        Percorso di salvataggio del grafico, nel caso non viene fornito è salvato in 'reports/img/BN_{metodoliga creazione}.jpg'
        
    Raises
    ------
    ValueError
        Se il modello non ha una struttura definita o è stato fornito un modello non valido.
    """
    if not isinstance(model, BeliefNetwork):
        raise ValueError("Modello fornito non valido")
    
    edges = model.get_edges()
    if not edges:
        raise ValueError("Il modello non ha una struttura definita")
    
    G = nx.DiGraph(model.get_edges())

    node_colors = {node: plt.cm.tab20(i) for i, node in enumerate(G.nodes())}

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2)

    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=list(node_colors.values()))

    for node in G.nodes():
        edges = G.out_edges(node)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[node_colors[node]] * len(edges), arrows=True, arrowsize=20)

    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title(title, pad=20)
    
    if save_as_image:
        if save_path is None:
            save_path = f'reports/img/BN_{model.get_creation_mode().upper()}.jpg'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()
    
def plot_confusion_matrix(confusion_matrix, save_as_image=False, save_path='reports/img/confusion_matrix.jpg'):
    """
    Crea un grafico per visualizzare la matrice di confusione.
    
    Parameters
    ----------
    confusion_matrix : np.ndarray di dimensione (n_classes, n_classes)
        Matrice di confusione da visualizzare.
    save_as_image : bool
        Se True, salva il grafico come immagine
    save_path : str, optional
        Percorso di salvataggio del grafico
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    if save_as_image:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()