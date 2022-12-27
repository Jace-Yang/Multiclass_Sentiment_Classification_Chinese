import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import logging
from classifier_utils import StackedClassifier, SentimentClassifier
logging.set_verbosity_error()

SEQ_LENGTH = 128
BATCH_SIZE = 8
LABEL_DICT = {'fear':0, 'neutral':1, 'sad':2, 'surprise':3, 'angry':4, 'happy':5} # Mapping label code and meaning
TOKENIZER = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext") # Hugging face BertTokenizer to load pretrain model

def convert_text_to_token(tokenizer, sentence, seq_length):
    """Tokenize sentence

    Args:
        tokenizer (PreTrainedTokenizer): a pretrained tokenizer with special token set to 
            {'unk_token': '[UNK]', 'sep_token': '[SEP]', 
             'pad_token': '[PAD]', 'cls_token': '[CLS]', 
             'mask_token': '[MASK]'}
        sentence (str): 
        seq_length (int): length of maximum input sentence accepted
    
    Returns: tuple(word_ids, segments, attention_masks)
        word_ids (list): tokenized sentence
        segments (list): label segmentation of original sentence and padding
        attention_masks (list): label whether the word is masked
    """ 
    tokens = tokenizer.tokenize(sentence) # Tokenize the sentence
    tokens = ["[CLS]"] + tokens + ["[SEP]"] # Add [CLS] before token and [SEP] after token
    word_ids = tokenizer.convert_tokens_to_ids(tokens) # Generate list of word id
    segments = [0] * len(word_ids) # Label whether it is segmented
    attention_masks = [1] * len(word_ids) # Label whether the word is masked
    # Chop or pad the sentence into a single length - seq_length
    if len(word_ids) < seq_length: # Padding
        length_to_pad = seq_length - len(word_ids)
        word_ids += [0] * length_to_pad # [0] is the index of word "PAD" in the vocabulary table
        segments += [1] * length_to_pad # [1] denotes that this part of words are PAD
        attention_masks += [0] * length_to_pad # Change attention mask of PAD part as [0]
    else: # Chopping
        word_ids = word_ids[:seq_length]
        segments = segments[:seq_length]
        attention_masks = attention_masks[:seq_length]
    assert len(word_ids) == len(segments) == len(attention_masks)
    return word_ids, segments, attention_masks

class DeployedClassifier:
    def __init__(self):
        # Model Fcuntion
        self.softmax = nn.Softmax(dim=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STACK_WEIGHTS = pd.read_excel('result/model/model_weight.xlsx').set_index('Model').to_dict()['weight']
        MODELS_PATHS_UNITS = {
            'BERT': ('bert-base-chinese', 768),
            'BERT-wwm': ('hfl/chinese-bert-wwm-ext', 768),
            'RoBERTa': ('uer/chinese_roberta_L-12_H-768', 768),
            'RoBERTa-wwm': ('hfl/chinese-roberta-wwm-ext', 768),
            'RoBERTa-wwm-large': ('hfl/chinese-roberta-wwm-ext-large', 1024),
            'Re-trained RoBERTa-wwm': ('hfl/rbt3', 768),
            'Re-trained RoBERTa-wwm-large': ('hfl/rbtl3', 1024),
        }
        self.LABEL_NAME_DICT = {0:'fear', 1:'neutral', 2:'sad', 3:'surprise', 4:'angry', 5:'happy'}
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
        cur_ids, cur_type, cur_mask = torch.LongTensor(np.array([cur_ids])).to(self.device), torch.LongTensor(np.array([cur_type])).to(self.device), torch.LongTensor(np.array([cur_mask])).to(self.device) # 数据构造成tensor形式
        # y_probas = []
        # for sentiment_classifier in self.sentiment_classifiers:
        #     with torch.no_grad():
        #         y_ = sentiment_classifier(cur_ids, token_type_ids=cur_type, attention_mask=cur_mask).squeeze()
        #         y_proba = self.softmax(y_)
        #         y_probas.append(y_proba)
        y_proba_stacked = self.stacked_classifier(cur_ids, token_type_ids=cur_type, attention_mask=cur_mask).squeeze()
        y_pred_stacked = self.LABEL_NAME_DICT[y_proba_stacked.argmax(dim=0).item()]
        y_all_pred_stacked = [(self.LABEL_NAME_DICT[i], y_proba.item()) for i, y_proba in enumerate(y_proba_stacked)]
        return y_all_pred_stacked, y_pred_stacked