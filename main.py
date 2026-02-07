from src.data_ingestion import DataLoader

def main():
    # Define paths
    RAW_DATA_PATH = "data/01-raw/Flight_Price_Dataset_of_Bangladesh.csv"
    
    # Initialize and Run Phase 1
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    loader.run_initial_audit()

if __name__ == "__main__":
    main()