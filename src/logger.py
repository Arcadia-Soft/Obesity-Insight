import sys
import atexit
from pathlib import Path

class Logger:
    """
    Logger personalizzato che scrive sia su console che su file.
    
    Attributes
    ----------
    terminal : TextIO
        Stream di output originale
    log : TextIO
        File di log
    closed : bool
        Stato del file di log
    """
    
    def __init__(self, filename):
        """
        Inizializza il logger.
        
        Parameters
        ----------
        filename : str
            Percorso del file di log
            
        Raises
        ------
        IOError
            Se non è possibile creare o aprire il file di log
        """
        self.terminal = sys.stdout
        self.log_path = Path(filename)
        
        # Crea la directory se non esiste
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.log = open(filename, 'w', encoding='utf-8')
            self.closed: bool = False
            atexit.register(self.close)
        except IOError as e:
            raise IOError(f"Impossibile aprire il file di log: {e}")

    def write(self, message) -> None:
        """
        Scrive il messaggio sia su console che su file.
        
        Parameters
        ----------
        message : str
            Messaggio da scrivere
        """
        self.terminal.write(message)
        if not self.closed:
            self.log.write(message)

    def flush(self):
        """
        Forza la scrittura dei buffer su disco.
        """
        self.terminal.flush()
        if not self.closed:
            self.log.flush()
        
    def close(self):
        """
        Chiude il file di log in modo sicuro.
        """
        if not self.closed:
            self.log.close()
            self.closed = True
            
    def __enter__(self):
        """
        Supporto per context manager.
        
        Returns
        -------
        Logger
            Istanza corrente
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Chiude il logger all'uscita dal context manager.
        """
        self.close()
