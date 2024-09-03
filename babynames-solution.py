import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_dataset(fname:str) -> pd.DataFrame:
    names_df = pd.read_csv(fname, header=None)
    names_df.columns = ["name", "gender"]

    # verifiying that there are no missing values in the dataframe and that only two genders are in the dataframe
    assert names_df["name"].isna().sum() == 0 and names_df["gender"].isna().sum() == 0
    assert names_df["name"].nunique() == len(names_df) and names_df["gender"].nunique() == 2

    return names_df

def tokenize_names(names:list[str], tokenizer:BertTokenizer) -> torch.Tensor:
    return tokenizer(names, padding=True, truncation=True, return_tensors='pt')

def get_embeddings(names:list[str], model:BertModel, tokenizer:BertTokenizer) -> np.ndarray:
    tokenized_names = tokenize_names(names, tokenizer)
    with torch.no_grad():
        outputs = model(**tokenized_names)
    return outputs.last_hidden_state[:, 0, :].numpy()

def encode_label(names_df:pd.DataFrame, label_name:str, label_encoder) -> pd.DataFrame:
    names_df[f"encoded_{label_name}"] = label_encoder.fit_transform(names_df[label_name])
    return names_df

def train(embeddings:np.ndarray, labels:pd.Series, classifier) -> tuple[RandomForestClassifier, float, float]:
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return classifier, accuracy, f1

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    classifier = RandomForestClassifier()
    encoder = LabelEncoder()

    print("loading dataset...")
    names_df = load_dataset("babynames-clean.csv")
    print("encoding labels...")
    names_df = encode_label(names_df, "gender", encoder)
    print("loading embeddings...")
    embeddings = get_embeddings(names_df["name"].to_list(), model, tokenizer)
    labels = names_df["encoded_gender"]
    print("training classifier...")
    trained_classifier, accuracy, f1 = train(embeddings, labels, classifier)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 score: {f1 * 100:.2f}%")