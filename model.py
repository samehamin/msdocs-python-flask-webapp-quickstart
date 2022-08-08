from inspect import getmodule
import requests
import json
from transformers import AutoModel, AutoTokenizer 
import torch
import pickle


def getLabels():
    INTENTS = ['datetime_query', 'iot_hue_lightchange', 'transport_ticket', 'takeaway_query', 'qa_stock',
            'general_greet', 'recommendation_events', 'music_dislikeness', 'iot_wemo_off', 'cooking_recipe',
            'qa_currency', 'transport_traffic', 'general_quirky', 'weather_query', 'audio_volume_up',
            'email_addcontact', 'takeaway_order', 'email_querycontact', 'iot_hue_lightup',
            'recommendation_locations', 'play_audiobook', 'lists_createoradd', 'news_query',
            'alarm_query', 'iot_wemo_on', 'general_joke', 'qa_definition', 'social_query',
            'music_settings', 'audio_volume_other', 'calendar_remove', 'iot_hue_lightdim',
            'calendar_query', 'email_sendemail', 'iot_cleaning', 'audio_volume_down',
            'play_radio', 'cooking_query', 'datetime_convert', 'qa_maths', 'iot_hue_lightoff',
            'iot_hue_lighton', 'transport_query', 'music_likeness', 'email_query', 'play_music',
            'audio_volume_mute', 'social_post', 'alarm_set', 'qa_factoid', 'calendar_set',
            'play_game', 'alarm_remove', 'lists_remove', 'transport_taxi', 'recommendation_movies',
            'iot_coffee', 'music_query', 'play_podcasts', 'lists_query']

    return INTENTS


def getModel():
    # load the model from disk

    filename = "./Data" + '/intent-classification-ar.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def predict(pred_text):
    #define the tokenizer
    model_ckpt = "aubmindlab/bert-base-arabertv2"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # tokenizing the text
    pred_inputs = tokenizer(pred_text, return_tensors="pt")
    # print(pred_inputs)

    # define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    pred_inputs = {k:v.to(device) for k,v in pred_inputs.items()}
    with torch.no_grad():
        pred_hidden_state = model(**pred_inputs).last_hidden_state[:,0].cpu().numpy()

    # print(pred_hidden_state)
    classifier = getModel()
    pred_label = classifier.predict(pred_hidden_state)

    labels = getLabels()
    # print(labels[pred_label[0]])
    # print(labels)
    return labels[pred_label[0]]


# dataset = pd.read_csv('Data/Salary_Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)

# pickle.dump(regressor, open('model.pkl','wb'))

# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[1.8]]))

