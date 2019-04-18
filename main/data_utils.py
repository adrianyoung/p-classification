import re
import os
import logging
import operator
import codecs
import random
from langconv import *
from tqdm import tqdm
from pprint import pprint
from collections import Counter
from utils import get_config_values, load_json, load_pickle, save_pickle, logger
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile, common_texts
from gensim.corpora import LowCorpus
from gensim.corpora import Dictionary
import numpy as np
from tflearn.data_utils import pad_sequences
tqdm.pandas()

KOREAN_TOKEN = '[KOREAN]'
HEBREW_TOKEN = '[HEBREW]'
ARABIC_TOKEN = '[ARABIC]'
JAPAN_TOKEN = '[JAPAN]'
ZRH_TOKEN = '[ZRH]'
THAI_TOKEN = '[THAI]'
LATIN_TOKEN = '[LATIN]'
GREECE_TOKEN = '[GREECE]'
YAMEINIYA_TOKEN = '[YAMEINIYA]'
INDIA_TOKEN = '[INDIA]'
GELUJIYA_TOKEN = '[GELUJIYA]'
SPACE_TOKEN = '[SPACE]'
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'

korean = ['[\uac00-\ud7af]']
hebrew = ['[\u0590-\u05ff]', '[\ufb00-\ufb4f]']
arabic = ['[\u0600-\u06ff]','[\u0750-\u077f]','[\ufb50-\ufdff]','[\ufe70-\ufeff]','[\u08a0-\u08ff]']
japan = ['[\u3040-\u309f]', '[\u30a0-\u30ff]', '[\u31f0-\u31ff]']
zrh = ['[\u3400-\u4dbf]', '[\uf900-\ufaff]']
thai = ['[\u0e00-\u0e7f]']
latin = ['[\u0080-\u00ff]','[\u0100-\u017f]','[\u0180-\u024f]','[\u1e00-\u1eff]','[\u2c60-\u2c7f]','[\ua720-\ua7ff]']
greece = ['[\u0370-\u03ff]']
yameiniya = ['[\u0530-\u058f]']
india = ['[\u0900-\u097f]','[\u0a80-\u0aff]','[\u0a00-\u0a7f]']
gelujiya = ['[\u10a0-\u10ff]']
space = ['\s']

TOKEN_LIST = [
    PAD_TOKEN, UNKNOWN_TOKEN, KOREAN_TOKEN, HEBREW_TOKEN, ARABIC_TOKEN, JAPAN_TOKEN,
    ZRH_TOKEN, THAI_TOKEN, LATIN_TOKEN, GREECE_TOKEN, YAMEINIYA_TOKEN, INDIA_TOKEN,
    GELUJIYA_TOKEN, SPACE_TOKEN
]

TOKEN_REGEXPS = {
    KOREAN_TOKEN:korean, HEBREW_TOKEN:hebrew, ARABIC_TOKEN:arabic, JAPAN_TOKEN:japan,
    ZRH_TOKEN:zrh, THAI_TOKEN:thai, LATIN_TOKEN:latin, GREECE_TOKEN:greece, YAMEINIYA_TOKEN:yameiniya,
    INDIA_TOKEN:india, GELUJIYA_TOKEN:gelujiya, SPACE_TOKEN:space
}

# build corpus for gibbsLDA
def save_for_corpus(data_list):
    file_corpus = get_config_values('corpus','text')
    if not os.path.exists(file_corpus):
        with codecs.open(file_corpus, 'w') as fp:
            total = sum([len(dataset) for dataset in data_list])
            fp.write((str(total) + '\n'))
            for dataset in tqdm(data_list):
                for line in dataset:
                    for word in line['postag']:
                        fp.write((Converter('zh-hans').convert(word['word'].strip().replace(' ', '')) + ' '))
                    if line['postag'] != None:
                        fp.write('\n')
    else:
        logger.info('corpus already done...')

# build for char dict
def deal_with_text(data_list, mode='full'):

    if len(data_list) == 1 and mode == 'train':
        cache_text = get_config_values('cache', 'text_train')
    elif len(data_list) == 1 and mode == 'dev':
        cache_text = get_config_values('cache', 'text_dev')
    elif len(data_list) == 2 and mode == 'mix':
        cache_text = get_config_values('cache', 'text_mix')
    elif len(data_list) == 3 and mode == 'full':
        cache_text = get_config_values('cache', 'text_full')
    else:
        logger.warn('Found data format wrong when dealing with text...')

    if not os.path.exists(cache_text):
        logger.info("dealing with text...")
        text = []
        for dataset in tqdm(data_list):
            text.extend([Converter('zh-hans').convert(line['text']) for line in dataset])
        save_pickle(cache_text, text)
    else:
        logger.info("loading with text...")
        text = load_pickle(cache_text)
    logger.info("text total num: {0}".format(len(text)))
    return text

# build (word, postag, length) for each text
def deal_with_postag(data_list, mode='full'):

    if len(data_list) == 1 and mode == 'train':
        cache_postag = get_config_values('cache', 'postag_train')
    elif len(data_list) == 1 and mode == 'dev':
        cache_postag = get_config_values('cache', 'postag_dev')
    elif len(data_list) == 2 and mode == 'mix':
        cache_postag = get_config_values('cache', 'postag_mix')
    elif len(data_list) == 3 and mode == 'full':
        cache_postag = get_config_values('cache', 'postag_full')
    else:
        logger.warn('Found data format wrong when dealing with postag...')

    if not os.path.exists(cache_postag):
        logger.info("dealing with postag...")
        postag = []
        for dataset in tqdm(data_list):
            for line in dataset:
                postag.append([[Converter('zh-hans').convert(word['word'].strip().replace(' ', '')),
                                word['pos'],len(word['word'])] for word in line['postag']])
        save_pickle(cache_postag, postag)
    else:
        logger.info("loading with postag...")
        postag = load_pickle(cache_postag)
    logger.info("postag total num: {0}".format(len(postag)))
    logger.info("postag 5: {0}".format(postag[:5]))
    return postag

# build (label, objects, postion) for each text
def deal_with_spo(data_list, mode='full'):

    if len(data_list) == 1 and mode == 'train':
        cache_spo = get_config_values('cache', 'spo_train')
    elif len(data_list) == 1 and mode == 'dev':
        cache_spo = get_config_values('cache', 'spo_dev')
    elif len(data_list) == 2 and mode == 'full':
        cache_spo = get_config_values('cache', 'spo_full')
    else:
        logger.warn('Found data format wrong when dealing with spo...')

    if not os.path.exists(cache_spo):
        logger.info("dealing with spo...")
        spos = []
        for dataset in tqdm(data_list):
            for line in dataset:
                pairs = []
                # position
                func = lambda x,y: list(re.search(re.escape(x.lower()), y.lower()).span())

                # true classes
                for spo in line['spo_list']:
                    pairs.append([
                    spo['predicate'],
                    [spo['object_type'], spo['subject_type']],
                    func(spo['object'], line['text']) + func(spo['subject'], line['text']),
                    ])
                    #list(re.search(re.escape(spo['object'].lower()), line['text'].lower()).span())
                    #    + list(re.search(re.escape(spo['subject'].lower()), line['text'].lower()).span()),

                # NA classes
                data = {}
                # {data:{obj/sub_type, pos, co-occur}}
                for spo in line['spo_list']:
                    if spo['object'] not in data.keys():
                        data[spo['object']] = {}
                        data[spo['object']]['type'] = spo['object_type']
                        data[spo['object']]['pos'] = func(spo['object'], line['text'])
                        data[spo['object']]['co-occur'] = []
                    if spo['subject'] not in data.keys():
                        data[spo['subject']] = {}
                        data[spo['subject']]['type'] = spo['subject_type']
                        data[spo['subject']]['pos'] = func(spo['subject'], line['text'])
                        data[spo['subject']]['co-occur'] = []

                for spo in line['spo_list']:
                    for key in data.keys():
                        if (spo['object'] == key) or (spo['subject']) == key:
                            data[key]['co-occur'].append(1)
                        else:
                            data[key]['co-occur'].append(0)

                # [('',{}),('',{}),('',{})]
                data = list(data.items())

                # judge by co-occurance
                for idx1 in range(len(data)-1):
                    for idx2 in range(idx1+1, len(data)):
                        co1 = np.array(data[idx1][1]['co-occur'])
                        co2 = np.array(data[idx2][1]['co-occur'])
                        if 2 not in co1+co2:
                            pairs.append([
                            'NA',
                            [data[idx1][1]['type'], data[idx2][1]['type']],
                            func(data[idx1][0], line['text']) + func(data[idx2][0], line['text']),
                            ])

                spos.append(pairs)

                # spos.append(
                #    [[
                #    spo['predicate'],
                #    [spo['object_type'], spo['subject_type']],
                #    list(re.search(re.escape(spo['object'].lower()), line['text'].lower()).span())
                #        + list(re.search(re.escape(spo['subject'].lower()), line['text'].lower()).span()),
                #    ] for spo in line['spo_list']]
                #    )
                # except:
                #    logger.info('what the fuck is that ??? text:{0} ; spo:{1}'.format(line['text'], line['spo_list']))

        save_pickle(cache_spo, spos)
    else:
        logger.info("loading with spo...")
        spos = load_pickle(cache_spo)
    logger.info("spo total num: {0}".format(len(spos)))
    logger.info("spo 5: {0}".format(spos[:5]))
    return spos

def build_postag_dict(postags):
    cache_postag = get_config_values('cache','postag')
    if not os.path.exists(cache_postag):
        logger.info('dealing with postag...')
        postag = Counter()
        for line in postags:
            for tag in line:
                postag.update(tag[1])
        save_pickle(cache_postag, postag)
    else:
        logger.info('loading with postag data...')
        postag = load_pickle(cache_postag)
    logger.info('postag total num: {0}'.format(len(dict(postag))))
    logger.info('postag frequent postag: {0}'.format(postag.most_common()[:10]))
    return postag

# build segm dict
def build_segm_dict(data_list):
    cache_segm = get_config_values('cache', 'segm')
    if not os.path.exists(cache_segm):
        logger.info("dealing with segm...")
        segm = Counter()
        for dataset in tqdm(data_list):
            for line in dataset:
                segm.update([Converter('zh-hans').convert(word['word'].strip().replace(' ', '')) for word in line['postag']])
        save_pickle(cache_segm, segm)
    else:
        logger.info("loading with segn...")
        segm = load_pickle(cache_segm)
    logger.info('segm total num: {0}'.format(len(dict(segm))))
    logger.info('segm frequent segm: {0}'.format(segm.most_common()[:10]))
    return segm

# build char dict
def build_char_dict(text):
    cache_char = get_config_values('cache', 'char')
    if not os.path.exists(cache_char):
        logger.info('dealing with char data...')
        char = Counter()
        for line in tqdm(text):
            char.update(line)
        save_pickle(cache_char, char)
    else:
        logger.info('loading with char data...')
        char = load_pickle(cache_char)
    logger.info('char total num: {0}'.format(len(dict(char))))
    logger.info('char frequent char: {0}'.format(char.most_common()[:10]))
    return char

# check coverage of vocab with embeddings index
def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
    logger.info('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    logger.info('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    text_clean(sorted_x)
    logger.info('Found oov total num: {0}'.format(len(sorted_x)))
    logger.info('Found top 10 oov: {0}'.format(sorted_x[:2000]))
    return sorted_x

# clean oovs with differnt languages
def text_clean(sorted_x):

    def clean(lang, regexps, oovs):
        oov_lang = []
        for oov in oovs[:]:
            for regexp in regexps:
                found = re.search(regexp, oov[0])
                if found != None:
                    oov_lang.append(oov[0])
                    oovs.remove(oov)
                    break
        logger.info('Found {0} oov total {1} : {2}'.format(lang, len(oov_lang), oov_lang))

    clean('korean', korean, sorted_x)
    clean('hebrew', hebrew, sorted_x)
    clean('arabic', arabic, sorted_x)
    clean('japan', japan, sorted_x)
    clean('zrh', zrh, sorted_x)
    clean('thai', thai, sorted_x)
    clean('latin', latin, sorted_x)
    clean('greece', greece, sorted_x)
    clean('yameiniya', yameiniya, sorted_x)
    clean('gelujiya', gelujiya, sorted_x)
    clean('india', india, sorted_x)
    clean('space', space, sorted_x)

# build postag mapping
class Postag_vocab(object):
    def __init__(self, data_list):
        self.data_list = data_list
        self._postag_to_id = {}
        self._id_to_postag = {}
        self._count = 0
        self.create_vocab()

    def create_vocab(self):

        postags = deal_with_postag(self.data_list, mode='full')
        postags = build_postag_dict(postags)

        for postag in [PAD_TOKEN, UNKNOWN_TOKEN]:
            self._postag_to_id[postag] = self._count
            self._id_to_postag[self._count] = postag
            self._count += 1

        for postag in postags.most_common():
            self._postag_to_id[postag] = self._count
            self._id_to_postag[self._count] = postag
            self._count += 1

    def postag2id(self, postag):
        return self._postag_to_id[postag]

    def id2postag(self, postag_id):
        return self._id_to_postag[postag_id]

    def size(self):
        return self._count

# build schemas mapping
class Schema_vocab(object):
    def __init__(self):
        self._object_to_id = {}
        self._id_to_object = {}
        self._label_to_id = {}
        self._id_to_label = {}
        self._count_object = 0
        self._count_label = 0
        self.read_schemas()
        self.create_mapping()

    def read_schemas(self):
        filename = get_config_values('dataset','schemas')
        self.schemas = load_json(filename)

    def create_mapping(self):

        for schemas in self.schemas:
            l = schemas['predicate']
            self._label_to_id[l] = self._count_label
            self._id_to_label[self._count_label] = l
            self._count_label += 1

        for schemas in self.schemas:
            for o in [schemas['object_type'], schemas['subject_type']]:
                if o not in self._object_to_id.keys():
                    self._object_to_id[o] = self._count_object
                    self._id_to_object[self._count_object] = o
                    self._count_object += 1

    def label2id(self, label):
        if label == 'NA':return 'NA'
        else:return self._label_to_id[label]

    def id2label(self, label_id):
        return self._id_to_label[label_id]

    def object2id(self, obj):
        return self._object_to_id[obj]

    def id2obj(self, obj_id):
        return self._id_to_object[obj_id]

    def label_size(self):
        return self._count_label

    def object_size(self):
        return self._count_object

# build char vocab
class Char_vocab(object):
    def __init__(self, data_list, index, max_size):
        self._char_to_id = {}
        self._id_to_char = {}
        self._count = 0
        self.data_list = data_list
        self.embeddings_index = index.vocab.keys()
        self.max_size = max_size
        self.create_vocab()

    def create_vocab(self):

        for c in TOKEN_LIST :
            self._char_to_id[c] = self._count
            self._id_to_char[self._count] = c
            self._count += 1

        text = deal_with_text(self.data_list, mode='full')
        char = build_char_dict(text)

        for c in char.most_common():

            c = c[0]

            if c in TOKEN_LIST:
                raise Exception('Special tokens shouldn\'t be in the vocab file, but {0} is'.format(w))

            if c in self._char_to_id:
                raise Exception('Duplicated char in vocabulary file: {0}'.format(w))

            if c in self.embeddings_index:
                self._char_to_id[c] = self._count
                self._id_to_char[self._count] = c
                self._count += 1

            if self.max_size != 0 and self._count >= self.max_size:
                break

    def char2id(self, char):
        if char not in self.embeddings_index:
            for token, regexps in TOKEN_REGEXPS.items():
                for regexp in regexps:
                    if re.search(regexp, char):
                        return self._char_to_id[token]
            return self._char_to_id[UNKNOWN_TOKEN]
        return self._char_to_id[char]

    def id2obj(self, char_id):
        return self._id_to_char[char_id]

    def size(self):
        return self._count

# build word vocab
class Word_vocab(object):
    def __init__(self, data_list, index, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self.data_list = data_list
        self.embeddings_index = index.vocab.keys()
        self.max_size = max_size
        self.create_vocab()

    def create_vocab(self):

        for c in [PAD_TOKEN, UNKNOWN_TOKEN]:
            self._word_to_id[c] = self._count
            self._id_to_word[self._count] = c
            self._count += 1

        word = build_segm_dict(self.data_list)

        for c in word.most_common():

            c = c[0]

            if c in [PAD_TOKEN, UNKNOWN_TOKEN]:
                raise Exception('Special tokens shouldn\'t be in the vocab file, but {0} is'.format(w))

            if c in self._word_to_id:
                raise Exception('Duplicated word in vocabulary file: {0}'.format(w))

            if c in self.embeddings_index:
                self._word_to_id[c] = self._count
                self._id_to_word[self._count] = c
                self._count += 1

            if self.max_size != 0 and self._count >= self.max_size:
                break

    def word2id(self, word):
        if word not in self.embeddings_index:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2obj(self, word_id):
        return self._id_to_word[word_id]

    def size(self):
        return self._count

def pos_mapping(hps):
    pos_map = {}
    pos_map[PAD_TOKEN] = 0
    pos_map[0] = 1
    count = 2
    for index in range(1, hps.distance):
        pos_map[index] = count
        count += 1
        pos_map[-index] = count
        count += 1
    return pos_map

def char2id(text, vocab):
    return np.array([ vocab.char2id(c) for c in text ])

def word2id(postag, vocab):
    return np.array([ vocab.word2id(w[0]) for w in postag ])

def mix2id(postag, text, vocab):
    try:
        return np.concatenate([[ vocab.word2id(w[0])] * w[2] for w in postag])
    except:
        return np.concatenate([ [vocab.word2id(UNKNOWN_TOKEN)] * len(text)])

def postag2id(postag, text, vocab):
    try:
        return np.concatenate([[ vocab.postag2id(w[1])] * w[2] for w in postag])
    except:
        return np.concatenate([ [vocab.postag2id(UNKNOWN_TOKEN)] * len(text)])

def label2onehot(label, vocab):
    res = np.zeros(vocab.label_size(), dtype=np.int32)
    if vocab.label2id(label) == 'NA':
        return res
    res[vocab.label2id(label)] = 1
    return res

def obj2id(objs, vocab):
    res = np.zeros(2, dtype=np.int32)
    res[0] = vocab.object2id(objs[0])
    res[1] = vocab.object2id(objs[1])
    return res

def distance2entity(text, poss, pos_map, pad_id, hps):
    pos_list =[[],[],[],[]]
    for pos_cur in range(len(text)):
        for index, pos_obj in enumerate(poss):
            pos_list[index].append(pos_map[pos_cur-pos_obj])
        if pos_cur + 1 == hps.sequence_length: break
    pos_list = pad_sequences(pos_list, maxlen=hps.sequence_length, value=pad_id)
    return np.array(pos_list)

def distance_entitys(poss, pos_map):
    pos2entity = [[],[]]
    pos_entity1 = [poss[0],poss[1]]
    pos_entity2 = [poss[2],poss[3]]

    for pos1 in pos_entity1:
        for pos2 in pos_entity2:
            pos2entity[0].append(pos_map[pos1-pos2])
            pos2entity[1].append(pos_map[pos2-pos1])

    return np.array(pos2entity)

def Process(data_list, mode='train'):
    text = deal_with_text(data_list, mode)
    postag = deal_with_postag(data_list, mode)
    spo_list = deal_with_spo(data_list, mode)
    raw_list = list(zip(text, postag))
    data_list = []
    for spos, raw in zip(spo_list, raw_list):
        for spo in spos:
            data_list.append([raw[0], raw[1], spo])
    return data_list

def lexical2id(ids, poss, pad_id, hps):

    length = len(ids)
    window = hps.entity_window
    bound = {'left':0, 'right':length}

    left_part = window // 2
    right_part = window - left_part

    center = (poss[1]-poss[0]) // 2 + poss[0]
    right_index = center + right_part
    right = right_index if right_index < bound['right'] else bound['right']
    left_index = center - left_part
    left = left_index if left_index >= bound['left'] else bound['left']

    l1 = np.pad(ids[left:right], (0, window-(right-left)), 'constant')

    center = (poss[3]-poss[2]) // 2 + poss[2]
    right = center + right_part
    right = right if right < bound['right'] else bound['right']
    left = center - left_part
    left = left if left >= bound['left'] else bound['left']

    l2 = np.pad(ids[left:right], (0, window-(right-left)), 'constant')

    return l1, l2

def Batch(data_list,char_vocab,word_vocab,schemas_vocab,pos_map,postag_vocab,hps):

    random.shuffle(data_list)
    label_sentence, char_sentence, mix_sentence, objects_ids, relative_position, entitys_position, lexical, postag_sentence = [],[],[],[],[],[],[],[]
    for cnt, data in enumerate(data_list):

        label, char, mix, objs, pos1, pos2, lex, postag = Example(
           data[0], data[1], data[2][0], data[2][1], data[2][2], schemas_vocab, char_vocab, word_vocab, pos_map, postag_vocab, hps
        )
        label_sentence.append(label)
        char_sentence.append(char)
        mix_sentence.append(mix)
        objects_ids.append(objs)
        relative_position.append(pos1)
        entitys_position.append(pos2)
        lexical.append(lex)
        postag_sentence.append(postag)

        if (cnt+1) % hps.batch_size == 0:

            data_dict={}
            data_dict['label_sentence'] = np.array(label_sentence)
            data_dict['char_sentence'] = char_sentence
            data_dict['postag_sentence'] = postag_sentence
            data_dict['mix_sentence'] = mix_sentence
            data_dict['objects_ids'] = objects_ids
            data_dict['relative_position'] = relative_position
            data_dict['entitys_position'] = np.array(entitys_position)
            data_dict['lexical'] = np.array(lexical)

            yield data_dict

            label_sentence.clear()
            char_sentence.clear()
            postag_sentence.clear()
            mix_sentence.clear()
            objects_ids.clear()
            relative_position.clear()
            entitys_position.clear()
            lexical.clear()

def Example(text, postag, label, objs, poss, schemas_vocab, char_vocab, word_vocab, pos_map, postag_vocab, hps):

    """ get ids of special tokens """
    pad_id = 0
    """ process the label """
    label_sentence = label2onehot(label, schemas_vocab)
    """ process the char """
    char_ids = char2id(text, char_vocab)
    char_sentence = char_ids if char_ids.shape[0] < hps.sequence_length else char_ids[:hps.sequence_length]
    char_sentence = np.pad(char_sentence, (0, hps.sequence_length-len(char_sentence)), 'constant')
    """ process the mix """
    mix_ids = mix2id(postag, text, word_vocab)
    mix_sentence = mix_ids if mix_ids.shape[0] < hps.sequence_length else mix_ids[:hps.sequence_length]
    mix_sentence = np.pad(mix_sentence, (0, hps.sequence_length-len(mix_sentence)), 'constant')
    """ process the object """
    objects_ids = obj2id(objs, schemas_vocab)
    """ process the postion """
    relative_position = distance2entity(text, poss, pos_map, pad_id, hps)
    entitys_position = distance_entitys(poss, pos_map)
    """ process lexical feature """
    lchar1, lchar2 = lexical2id(char_ids, poss, pad_id, hps)
    lmix1, lmix2 = lexical2id(mix_ids, poss, pad_id, hps)
    """ process the postag """
    postag_ids = postag2id(postag, text, postag_vocab)
    postag_sentence = postag_ids if postag_ids.shape[0] < hps.sequence_length else postag_ids[:hps.sequence_length]
    postag_sentence = np.pad(postag_sentence, (0, hps.sequence_length-len(postag_sentence)), 'constant')

    return label_sentence, char_sentence, mix_sentence, objects_ids, relative_position, entitys_position, [lchar1, lchar2, lmix1, lmix2], postag_sentence

def full_data():

    file_train = get_config_values('dataset', 'train')
    file_dev = get_config_values('dataset', 'dev')
    file_test = get_config_values('dataset', 'test')
    data_train = load_json(file_train)
    data_dev = load_json(file_dev)
    data_test = load_json(file_test)

    return [data_train, data_dev, data_test]

def build_vocab(hps, mode, data_list=None, embeddings_index=None):

    logger.info('dealing with {0} data...'.format(mode))
    if mode == 'char':
        vocab = Char_vocab(data_list, embeddings_index, hps.vocab_size_c)
    elif mode == 'word':
        vocab = Word_vocab(data_list, embeddings_index, hps.vocab_size_w)
    elif mode == 'postag':
        vocab = Postag_vocab(data_list)
    else:
        vocab = Schema_vocab()
    return vocab

def main():

    #1.load raw_data
    file_train = get_config_values('dataset', 'train')
    file_dev = get_config_values('dataset', 'dev')
    file_test = get_config_values('dataset', 'test')
    data_train = load_json(file_train)
    data_dev = load_json(file_dev)
    data_test = load_json(file_test)
    #2.using all the data build vocab
    text = deal_with_text([data_train,data_dev,data_test], mode='full')
    postag = deal_with_postag([data_train,data_dev,data_test], mode='full')
    spo = deal_with_spo([data_train,data_dev], mode='full')
    char = build_char_dict(text)
    segm = build_segm_dict([data_train,data_dev,data_test])
    # save_for_corpus([data_train,data_dev,data_test])
    #3.loading the embeddings
    w2v_char = get_config_values('vector', 'w2v_char')
    # w2v_segm1 = get_config_values('vector', 'w2v_segm1')
    # w2v_segm2 = get_config_values('vector', 'w2v_segm2')
    twe_segm = get_config_values('vector', 'twe_segm')
    embeddings_index1 = KeyedVectors.load_word2vec_format(w2v_char, binary=False)
    # embeddings_index2 = KeyedVectors.load_word2vec_format(w2v_segm1, binary=False)
    # embeddings_index3 = KeyedVectors.load_word2vec_format(w2v_segm2, binary=False)
    embeddings_index4 = KeyedVectors.load_word2vec_format(twe_segm, binary=False)
    #4.build vocab
    char_vocab = Char_vocab([data_train,data_dev,data_test], embeddings_index1, 10000)
    word_vocab = Word_vocab([data_train,data_dev,data_test], embeddings_index4, 500000)
    postag_vocab = Postag_vocab([data_train,data_dev,data_test])
    schemas_vocab = Schema_vocab()
    logger.info('vocab char num: {0}'.format(char_vocab.size()))
    logger.info('vocab word num: {0}'.format(word_vocab.size()))
    logger.info('vocab postag num: {0}'.format(postag_vocab.size()))
    logger.info('schemas object num: {0}'.format(schemas_vocab.object_size()))
    logger.info('schemas label num: {0}'.format(schemas_vocab.label_size()))
    #5.checking the coverage of embeddings
    # oov1 = check_coverage(dict(char), embeddings_index1)
    # oov2 = check_coverage(dict(segm), embeddings_index2)
    # oov3 = check_coverage(dict(segm), embeddings_index3)
    # oov4 = check_coverage(dict(segm), embeddings_index4)

if __name__ == '__main__':
    main()
