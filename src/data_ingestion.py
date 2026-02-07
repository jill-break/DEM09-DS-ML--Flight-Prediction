import pandas as pd
import logging
import os
from datetime import datetime

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self._setup_logging()

    def _setup_logging(self):
        """Initializes logging to both console and a file."""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = f"{log_dir}/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Logging initialized.")

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from the specified path."""
        try:
            if not os.path.exists(self.file_path):
                self.logger.error(f"File not found at {self.file_path}")
                raise FileNotFoundError
            
            self.df = pd.read_csv(self.file_path)
            self.logger.info(f"Successfully loaded data from {self.file_path}")
            self.logger.info(f"Dataset Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def run_initial_audit(self):
        """Performs a health check on the dataset and logs findings."""
        if self.df is None:
            self.logger.warning("No data loaded to audit.")
            return

        self.logger.info("--- Starting Data Audit ---")
        
        # Checking for missing values
        null_counts = self.df.isnull().sum()
        self.logger.info(f"Missing Values per Column:\n{null_counts[null_counts > 0]}")

        # Checking for duplicates
        duplicate_count = self.df.duplicated().sum()
        self.logger.info(f"Duplicate Rows Found: {duplicate_count}")

        # Summary Statistics for Numerical Columns
        num_stats = self.df.describe().to_string()
        self.logger.info(f"Numerical Summary Statistics:\n{num_stats}")

        # Cardinality of Categorical Columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_vals = self.df[col].nunique()
            self.logger.info(f"Column '{col}' has {unique_vals} unique values.")

        self.logger.info("--- Data Audit Complete ---") 