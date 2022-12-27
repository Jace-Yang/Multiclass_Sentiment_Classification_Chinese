import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from classifier_utils import StackedClassifier, SentimentClassifier
from data_load_utils import convert_text_to_token, LABEL_DICT, TOKENIZER, SEQ_LENGTH

from transformers import logging
logging.set_verbosity_error()

class DeployedClassifier:
    def __init__(self):
        # Model Function
        self.softmax = nn.Softmax(dim=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STACK_WEIGHTS = pd.read_excel('result/model/model_weight.xlsx').set_index('Model').to_dict()['weight']
        MODELS_PATHS_UNITS = {
            'BERT-wwm': ('hfl/chinese-bert-wwm-ext', 768),
            'RoBERTa-wwm': ('hfl/chinese-roberta-wwm-ext', 768),
            'RoBERTa-wwm-large': ('hfl/chinese-roberta-wwm-ext-large', 1024),
            '3L RoBERTa-wwm': ('hfl/rbt3', 768),
            '3L RoBERTa-wwm-large': ('hfl/rbtl3', 1024),
        }
        self.LABEL_NAME_DICT = {v: k for k, v in LABEL_DICT.items()}

        # Load base model
        BEST_MODEL_FOLDER = 'result/model/'  # Path to save best model
        TESTING = False
        BASE_MODELS = {}
        for model_name in MODELS_PATHS_UNITS.keys():
            print('--- Loading', model_name, '---')
            # Initialize model
            sentiment_classifier = SentimentClassifier(num_classes=6, 
                                                       model_name=model_name, 
                                                       pretrain_path=MODELS_PATHS_UNITS[model_name][0], 
                                                       hidden_size=MODELS_PATHS_UNITS[model_name][1]).to(self.device)
            # Load model parameters
            model_path = f'{BEST_MODEL_FOLDER}best_{model_name}.pth' if not TESTING else f'{BEST_MODEL_FOLDER}best_testing_{model_name}.pth'
            sentiment_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            BASE_MODELS[model_name] = sentiment_classifier

        # Load model weight
        self.stacked_classifier = StackedClassifier(BASE_MODELS, STACK_WEIGHTS, self.device)

    def pred(self, word):
        cur_ids, cur_type, cur_mask = convert_text_to_token(TOKENIZER, word, seq_length=SEQ_LENGTH)
        cur_ids, cur_type, cur_mask = torch.LongTensor(np.array([cur_ids])).to(self.device), torch.LongTensor(np.array([cur_type])).to(self.device), torch.LongTensor(np.array([cur_mask])).to(self.device)
        y_proba_stacked = self.stacked_classifier(cur_ids, token_type_ids=cur_type, attention_mask=cur_mask).squeeze()
        y_pred_stacked = self.LABEL_NAME_DICT[y_proba_stacked.argmax(dim=0).item()]
        y_all_pred_stacked = [(self.LABEL_NAME_DICT[i], y_proba.item()) for i, y_proba in enumerate(y_proba_stacked)]
        return y_all_pred_stacked, y_pred_stacked