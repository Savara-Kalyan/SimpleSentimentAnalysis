from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(config):
    print("Preparing data...")
    # load data
    df = pd.read_csv(config.data.csv_file_path)

    df['label'] = pd.factorize(df['sentiment'])[0]

    train_df , test_df = train_test_split(df, test_size = config.data.test_set_ratio, stratify=df["sentiment"], random_state = config.data.random_state)

    train_df.to_csv(config.data.train_csv_file_path, index=False)
    test_df.to_csv(config.data.test_csv_file_path, index=False)
 
    print("Data prepared!")


if __name__ == "__main__":

    config = OmegaConf.load("./params.yaml")
    prepare_data(config)
