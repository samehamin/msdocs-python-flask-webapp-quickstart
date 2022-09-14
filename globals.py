from transformers import AutoModel, AutoTokenizer 
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import pipeline
import torch


def init():
    global device
    global tokenizer_xlmr_banking
    global model_xlmr_banking
    global model_xlmr_banking_seq
    global pipe_xlmr_banking
    global classifier_smallTalk
    global tokenizer_NER
    global clfr_NER
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # loading the banking domain model for head training
    tokenizer_xlmr_banking = AutoTokenizer.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
    model_xlmr_banking = AutoModel.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
    model_xlmr_banking = model_xlmr_banking.to(device)

    # loading the banking domain model for pipeline classification
    model_xlmr_banking_seq = AutoModelForSequenceClassification.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
    model_xlmr_banking_seq = model_xlmr_banking_seq.to(device)
    pipe_xlmr_banking = pipeline("text-classification", model_xlmr_banking_seq, tokenizer=tokenizer_xlmr_banking)

    # loading the small talk pipeline 
    classifier_smallTalk = pipeline("zero-shot-classification",
                        model="joeddav/xlm-roberta-large-xnli")

    # loading the NER model
    tokenizer_NER = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
    clfr_NER = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
