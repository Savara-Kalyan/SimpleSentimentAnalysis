from omegaconf import OmegaConf
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train(config):
    print("Training model...")

    train_inputs = joblib.load(config.features.train_features_path)
    train_labels = pd.read_csv(config.data.train_csv_file_path)["label"].values

    penalty = config.train.penalty
    C = config.train.C
    solver = config.train.solver

    model = LogisticRegression(penalty = penalty, C = C, solver = solver)
    model.fit(train_inputs, train_labels)

    joblib.dump(model, config.train.model_path)

    print("Model trained!")



if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    train(config)