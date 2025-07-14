import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ig")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ig.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_url)
        logger.debug('Data loaded from CSV: %s', data_url)
        return data
    except pd.errors.ParserError as e:
        logger.error("Parsing error loading data: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading data: %s", e)
        raise

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['Stage_fear'] = data['Stage_fear'].map({'No': 0, 'Yes': 1})
        data['Drained_after_socializing'] = data['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

        lb = LabelEncoder()
        data['pre'] = lb.fit_transform(data['Personality'])
        data.drop(columns=['Personality'], inplace=True)

        col_means = data.mean(numeric_only=True, skipna=True)
        data = data.fillna(col_means)
        logger.debug("Preprocessing complete, nulls filled with column means")
        return data
    except KeyError as e:
        logger.error("Missing column during preprocessing: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise

def save_data(train: pd.DataFrame, test: pd.DataFrame, data_path: str):
    try:
        raw_data = os.path.join(data_path, 'raw')
        os.makedirs(raw_data, exist_ok=True)
        train.to_csv(os.path.join(raw_data, 'train.csv'), index=False)
        test.to_csv(os.path.join(raw_data, 'test.csv'), index=False)
        logger.debug("Train and test data saved successfully to %s", raw_data)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = r"C:\Users\lenovo\Desktop\mlops\MLops-complte-pipeline\p1.csv"
        data = load_data(data_path)
        final_df = preprocess_data(data)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, "./data")
    except Exception as e:
        logger.error("Failed to complete data ingestion pipeline: %s", e)
        print("Error:", e)

if __name__ == '__main__':
    main()
