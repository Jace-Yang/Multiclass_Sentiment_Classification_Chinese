import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

LABEL_DICT = {'fear':0, 'neutral':1, 'sad':2, 'surprise':3, 'angry':4, 'happy':5} # Mapping label code and meaning
DEVELOPMENT_SET_PATH = 'data/usual_train.txt'
TEST_SET_PATH = 'data/usual_test_labeled.txt'
BATCH_SIZE = 8
SEQ_LENGTH = 128
TOKENIZER = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext") # Hugging face BertTokenizer to load pretrain model
TESTING = False

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

def genDataLoader(data_type):
    '''Construct dataset loader

    Args:
        data_type (str): 'train' in training, 'val' in validating, 'test' in testing
    '''
    if data_type == 'test':
        with open(TEST_SET_PATH, encoding='utf8') as file:
            data = json.load(file)
    else:
        with open(DEVELOPMENT_SET_PATH, encoding='utf8') as file:
            data = json.load(file)
            # TESTING_STAGE
            if TESTING:
                dev_set, _ = train_test_split(data, train_size=320, random_state=4995)
                train_set, val_set = train_test_split(dev_set, test_size=0.2, random_state=4995)
            else:
                train_set, val_set = train_test_split(data, test_size=5000, random_state=4995)
            data = train_set if data_type == 'train' else val_set
    ids_pool = []
    segments_pool = []
    masks_pool = []
    target_pool = []
    count = 0
    # Process all the sentences
    for each in data:
        cur_ids, cur_type, cur_mask = convert_text_to_token(TOKENIZER, each['content'], seq_length = SEQ_LENGTH)
        ids_pool.append(cur_ids)
        segments_pool.append(cur_type)
        masks_pool.append(cur_mask)
        cur_target = LABEL_DICT[each['label']]
        target_pool.append([cur_target])
        count += 1
        if count % 2000 == 0:
            print(f'Processed {count} sentences for {data_type}')
    # Construct Data Generater
    data_gen = TensorDataset(torch.LongTensor(np.array(ids_pool)),
                             torch.LongTensor(np.array(segments_pool)),
                             torch.LongTensor(np.array(masks_pool)),
                             torch.LongTensor(np.array(target_pool)))
    sampler = RandomSampler(data_gen)
    loader = DataLoader(data_gen, sampler=sampler, batch_size=BATCH_SIZE)
    return loader