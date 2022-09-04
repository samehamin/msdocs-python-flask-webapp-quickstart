from transformers import AutoModel, AutoTokenizer 
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
import torch


def init():
    global device
    global tokenizer_xlmr_banking
    global model_xlmr_banking
    global model_xlmr_banking_seq
    global pipe_xlmr_banking
    global classifier_smallTalk
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_xlmr_banking = AutoTokenizer.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')

    model_xlmr_banking = AutoModel.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
    model_xlmr_banking = model_xlmr_banking.to(device)

    model_xlmr_banking_seq = AutoModelForSequenceClassification.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
    model_xlmr_banking_seq = model_xlmr_banking_seq.to(device)

    pipe_xlmr_banking = pipeline("text-classification", model_xlmr_banking_seq, tokenizer=tokenizer_xlmr_banking)

    classifier_smallTalk = pipeline("zero-shot-classification",
                        model="joeddav/xlm-roberta-large-xnli")

