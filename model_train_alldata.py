from datasets import Dataset
import mysqlconnector as mysqlconn
import torch
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import globals_nlp


dataset_utter = None


def load_utterances():
    print("loading utterances...")
    global dataset_utter
    try:
        query = "SELECT * FROM user_utterances;"
        df_utter = mysqlconn.read_df_sqlalchemy(query)
        # print(df_utter.head())
        dataset_utter = Dataset.from_pandas(df_utter)
    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))
load_utterances()


# Tokenize the whole dataset
def tokenize(batch):
    return globals_nlp.tokenizer_xlmr_banking(batch["utterance"], padding=True, truncation=True)


def tokenize_dataset():
    try:
        print("tokenizing the dataset ...")
        # applying the extract_hidden_states() function has added a new hidden_state
        # dataset_encoded = dataset_utter.map(tokenize, batched=True, batch_size=None)
        dataset_encoded = dataset_utter.map(tokenize, batched=True)
        # column to our dataset:
        # print(dataset_encoded.column_names)
        return dataset_encoded
    except Exception as e:
        print("Error: ", e)
# tokenize_dataset()


#  we’ll use the map() method of DatasetDict to extract all the hidden states in one go
def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(globals_nlp.device) for k,v in batch.items()
                if k in globals_nlp.tokenizer_xlmr_banking.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = globals_nlp.model_xlmr_banking(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


def extract_hidden_states_dataset():
    print("Extracting hidden states...")
    
    dataset_encoded = tokenize_dataset()
    #  convert the input_ids and attention_mask columns to the "torch" format
    dataset_encoded.set_format("torch", columns=["input_ids", 
                                              "attention_mask", "label"])
    
    #  convert the input_ids and attention_mask columns to the "torch" format
    dataset_encoded.set_format("torch", columns=["input_ids", 
                                              "attention_mask", "label"])
    
    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)
    # print(dataset_hidden.column_names)
    return dataset_hidden
# extract_hidden_states_dataset()


def create_feature_matrix():
    print("create_feature_matrix...")
    
    dataset_hidden = extract_hidden_states_dataset()
    X_train = np.array(dataset_hidden["hidden_state"])
    y_train = np.array(dataset_hidden["label"])
    # X_valid = np.array(dataset_hidden["test"]["hidden_state"])
    # y_valid = np.array(dataset_hidden["test"]["label"])
    # X_train.shape, X_valid.shape
    # print(X_train)
    return X_train, y_train


def save_model(clf):
    print("save_model...")
    
    import pickle
    filename = 'Data/UDModel_banks.sav'
    pickle.dump(clf, open(filename, 'wb'))


def train_classifier():
    print("train_classifier ...")
    X_train, y_train = create_feature_matrix()

    #Linear Support Vector Machine
    clf = Pipeline([
        ('clf', SGDClassifier(loss='log_loss', penalty='l2',alpha=1e-3, random_state=42, max_iter=3000, tol=None)),
        ])

    clf.fit(X_train, y_train)

    save_model(clf)

# train_classifier()