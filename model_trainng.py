import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

data_dir='logs'
os.makedirs(data_dir,exist_ok=True)

logger=logging.getLogger('model_traing')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(data_dir,'model_traing.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(filepath:str)->pd.DataFrame:
    try:
        data=pd.read_csv(filepath)
        logger.debug('sucessfully loaded%s',filepath)
        return data
    except Exception as e:
        logger.error("unecpeed error %s",e)
def train_model(train_data:pd.DataFrame,y_train:pd.DataFrame)->LogisticRegression:
    try:
        tf=LogisticRegression(train_data,y_train)
        logger.debug("succesfully trained the model %s",tf)
        return tf
    except Exception as e:
        logger.error("unexcpped eroor is occuur",e)
def main():
    file_path='./data/raw/train.csv'
    data=load_data(file_path)
    y_train='./data/raw/test.csv'
    model=train_model(data,y_train)
    logger.debug("sucesfully model is trained and ready")
    
if __name__=='__main__':
    main()



