import pandas as pd
import joblib
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score

def test(config):
    print("Testing model...")

    test_inputs = joblib.load(config.features.test_features_path)
    test_df = pd.read_csv(config.data.test_csv_file_path)
    test_labels = test_df["label"].values
    class_names = test_df["sentiment"].unique().tolist()

    model = joblib.load(config.train.model_path)

    metric_name = config.test.metric

    metric = {
        "accuracy": accuracy_score,
        "f1": f1_score
    }[metric_name]

    predictions = model.predict(test_inputs)

    results = metric(test_labels, predictions)
    result_dict = {metric_name:float(results)}

    OmegaConf.save(result_dict, config.test.results_save_path)

    print(f"Model {metric_name}: {results}")

    print("Model tested!")


if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    test(config)