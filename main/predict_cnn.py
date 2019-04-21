# -*- coding: utf-8 -*-
import os
import math
import gc
import random
import pprint
import tensorflow as tf
import numpy as np
from data_utils import *
from utils import *
from cnn import CNN
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from tqdm import tqdm

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("ckpt_dir",get_config_values('model','cnn'),"checkpoint location for the model")
tf.app.flags.DEFINE_string("log_path",get_config_values('model','log'),"path of summary log.")

tf.app.flags.DEFINE_integer("sequence_length", 150,"the max length of a sentence in documents")
tf.app.flags.DEFINE_integer("entity_window", 15, "the window of the entity and its context")
tf.app.flags.DEFINE_integer("distance", 300, "the distance from char to entity")
tf.app.flags.DEFINE_integer("num_classes", 50,"relation type total number")

tf.app.flags.DEFINE_string("char_embedding_model_path",get_config_values('vector','w2v_char'),"char's vocabulary and vectors")
tf.app.flags.DEFINE_string("word_embedding_model_path",get_config_values('vector','glv_segm'),"word's vocabulary and vectors")

tf.app.flags.DEFINE_boolean("is_training", False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("embed_size", 300,"word embedding size")
tf.app.flags.DEFINE_integer("embed_size_p", 10,"position embedding size")
tf.app.flags.DEFINE_integer("sentence_size", 300,"the size of the sentence level embedding")

tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("decay_step", 5000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.3, "Rate of decay for learning rate.")

tf.app.flags.DEFINE_integer("first_decay_steps", 2000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("t_mul", 2.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("m_mul", 0.40, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("alpha", 0.00005, "Rate of decay for learning rate.")

tf.app.flags.DEFINE_integer("is_frozen_step", 400, "how many steps before fine-tuning the embedding.")

tf.app.flags.DEFINE_boolean("use_highway_flag", False,"using highway network or not.")
tf.app.flags.DEFINE_integer("highway_layers", 1,"How many layers in highway network.")

tf.app.flags.DEFINE_boolean("use_ranking_loss", True,"using ranking loss or not.")
tf.app.flags.DEFINE_float("lm", 1.0,"lambda in ranking loss")
tf.app.flags.DEFINE_float("margin_plus", 2.5,"margin value for postive in ranking loss")
tf.app.flags.DEFINE_float("margin_minus", 0.5,"margin value for negative in ranking loss")

tf.app.flags.DEFINE_boolean("use_dev", True,"using dev dataset.")
tf.app.flags.DEFINE_boolean("use_test", False,"using test dataset.")

tf.app.flags.DEFINE_integer("k_fold", 5, "K-fold Cross Vaildation")
tf.app.flags.DEFINE_integer("num_epochs", 5,"number of epochs to run.")

tf.app.flags.DEFINE_integer("vocab_size_c", 7533,"vocab size for char")
tf.app.flags.DEFINE_integer("vocab_size_w", 422901,"vocab size for word")

filter_sizes = [6,7,8,9]
feature_map = [75,75,75,75]
vocab_size = [7533,422901,600,28,18]

def main(_):

    data_list = full_data()
    char_embedding = KeyedVectors.load_word2vec_format(FLAGS.char_embedding_model_path, binary=False)
    word_embedding = KeyedVectors.load_word2vec_format(FLAGS.word_embedding_model_path, binary=False)
    schemas_vocab = build_vocab(FLAGS, 'schemas')
    char_vocab = build_vocab(FLAGS,'char',data_list,char_embedding)
    word_vocab = build_vocab(FLAGS,'word',data_list,word_embedding)
    postag_vocab = build_vocab(FLAGS,'postag',data_list)
    pos_map = pos_mapping(FLAGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    if FLAGS.use_dev:
        file_dev = get_config_values('dataset', 'dev')
        data_dev = load_json(file_dev)
        data_list = Process([data_dev], mode='dev')

    if FLAGS.use_test:
        file_test = get_config_values('dataset', 'test')
        data_test = load_json(file_test)
        data_list = Process([data_test], mode='test')

    logits = np.array([])
    for K in range(FLAGS.k_fold):
        with tf.Session(config=config) as sess:
            # instantiate model
            Model = CNN(FLAGS.is_training,FLAGS.num_classes,vocab_size,FLAGS.batch_size,FLAGS.embed_size,FLAGS.embed_size_p,FLAGS.learning_rate,FLAGS.decay_step,FLAGS.decay_rate,FLAGS.entity_window,
                    FLAGS.sequence_length,filter_sizes,feature_map,FLAGS.use_highway_flag,FLAGS.highway_layers,FLAGS.sentence_size,FLAGS.use_ranking_loss,FLAGS.lm,FLAGS.margin_plus,FLAGS.margin_minus,
                    FLAGS.first_decay_steps, FLAGS.t_mul, FLAGS.m_mul, FLAGS.alpha)
            # initialize saver
            saver = tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir+"Model{}/checkpoint".format(K)):
                logger.info("Restoring Variables from Checkpoint.")
                save_path=FLAGS.ckpt_dir+"Model{0}/Model{0}-5F.ckpt-{1}".format(K,FLAGS.num_epochs-1)
                saver.restore(sess,save_path)
            else:
                logger.info("Can't load model checkpoint...stoping...")
                return

            data = Batch(data_list, char_vocab, word_vocab, schemas_vocab, pos_map, postag_vocab, FLAGS)
            f1_score, p_score, r_score, confusion_matrix, logits, _ = do_eval(sess, data, Model)
            print ("Model%d\tf1_score:%.4f\tprecision_score:%.4f\t recall_score:%.4f" % (K, f1_score, p_score, r_score))
            print ("Model %d\tConfusion matrix:" % (K))
            pprint (confusion_matrix)

        logits += logits * 0.7
        del Model
        gc.collect()
        tf.reset_default_graph()

        with tf.Session(config=config) as sess:
            # instantiate model
            Model = CNN(FLAGS.is_training,FLAGS.num_classes,vocab_size,FLAGS.batch_size,FLAGS.embed_size,FLAGS.embed_size_p,FLAGS.learning_rate,FLAGS.decay_step,FLAGS.decay_rate,FLAGS.entity_window,
                    FLAGS.sequence_length,filter_sizes,feature_map,FLAGS.use_highway_flag,FLAGS.highway_layers,FLAGS.sentence_size,FLAGS.use_ranking_loss,FLAGS.lm,FLAGS.margin_plus,FLAGS.margin_minus,
                    FLAGS.first_decay_steps, FLAGS.t_mul, FLAGS.m_mul, FLAGS.alpha)
            # initialize saver
            saver = tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir+"Model{}/checkpoint".format(K)):
                logger.info("Restoring Variables from Checkpoint.")
                save_path=FLAGS.ckpt_dir+"Model{0}/Model{0}-5F.ckpt-{1}".format(K,FLAGS.num_epochs-2)
                saver.restore(sess,save_path)
            else:
                logger.info("Can't load model checkpoint...stoping...")
                return

            data = Batch(data_list, char_vocab, word_vocab, schemas_vocab, pos_map, postag_vocab, FLAGS)
            f1_score, p_score, r_score, confusion_matrix, logits, labels = do_eval(sess, data, Model)
            print ("Model%d\tf1_score:%.4f\tprecision_score:%.4f\t recall_score:%.4f" % (K, f1_score, p_score, r_score))
            print ("Model %d\tConfusion matrix:" % (K))
            pprint (confusion_matrix)

        logits += logits * 0.3
        del Model
        gc.collect()
        tf.reset_default_graph()

    logits = logits / K
    threshold, _ = best_threshold(logits, labels)

# grid search for best threshold to detect NA
def best_threshold(logits, labels):
    # idx, cur, max, y_pred
    tmp = [0,0,0,[]]
    delta = 0
    y_pred = []
    y_true = []

    true = np.argsort(labels)[:,-1]
    index_NA = np.argwhere((np.sum((np.array(labels)==0), axis=1) == FLAGS.num_classes))
    for index in index_NA:
        true[index] = FLAGS.num_classes
    y_true.extend(true)

    for tmp[0] in tqdm(np.arange(0.0, 0.501, 0.001)):

        pred = np.argsort(logits)[:,-1]
        index_NA = np.argwhere((np.sum((logits<tmp[0]), axis=1) == FLAGS.num_classes))
        for index in index_NA:
            pred[index] = FLAGS.num_classes
        y_pred.extend(pred)

        tmp[1] = f1_score(y_true, y_pred, labels = [ i for i in range(FLAGS.num_classes + 1)], average='macro')
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
            tmp[3] = y_pred

        y_pred.clear()

    print ('best_threshold is {:.4f} with F1_score: {:.4f}'.format(delta, tmp[2]))

    return delta, tmp[3]

def do_predict_dev(sess, test_data, Model):
    pass

def do_eval(sess, valid_data, Model):
    logits_collection = []
    labels_collection = []
    iteration = 0
    for batch in valid_data:

        feed_dict={}

        feed_dict[Model.dropout_keep_prob] = 1.0
        feed_dict[Model.dropout_keep_prob_spatial] = 1.0
        feed_dict[Model.input_x_c] = batch['char_sentence']
        feed_dict[Model.input_x_w] = batch['mix_sentence']
        feed_dict[Model.input_x_t] = batch['postag_sentence']
        feed_dict[Model.input_x_p] = batch['relative_position']
        feed_dict[Model.input_x_p_l1] = batch['entitys_position'][:,0,:]
        feed_dict[Model.input_x_p_l2] = batch['entitys_position'][:,1,:]
        feed_dict[Model.input_x_o] = batch['objects_ids']
        feed_dict[Model.input_x_c_l1] = batch['lexical'][:,0,:]
        feed_dict[Model.input_x_c_l2] = batch['lexical'][:,1,:]
        feed_dict[Model.input_x_w_l1] = batch['lexical'][:,2,:]
        feed_dict[Model.input_x_w_l2] = batch['lexical'][:,3,:]
        feed_dict[Model.input_y] = batch['label_sentence']
        feed_dict[Model.tst] = FLAGS.is_training

        logits=sess.run([Model.logits],feed_dict)
        logits_collection.append(logits[0][0])
        labels_collection.append(batch['label_sentence'][0])

        iteration += 1
        if iteration % 1000 == 0:
            print ("Model has processed {} examples".format(iteration))

    y_true = []
    y_pred = []
    # ranking loss

    print (np.argsort(labels_collection))
    true = np.argsort(labels_collection)[:,-1]
    index_NA = np.argwhere((np.sum((np.array(labels_collection)==0), axis=1) == FLAGS.num_classes))
    #print (index_NA)
    for index in index_NA:
        true[index] = FLAGS.num_classes
    y_true.extend(true)


    pred = np.argsort(logits_collection)[:,-1]
    print (pred.shape)
    index_NA = np.argwhere((np.sum((np.array(logits_collection)<0), axis=1) == FLAGS.num_classes))
    for index in index_NA:
        pred[index] = FLAGS.num_classes
    y_pred.extend(pred)

    # evaluation
    f1 = f1_score(true, pred,labels = [ i for i in range(FLAGS.num_classes + 1)], average='macro')
    p = precision_score(true, pred,labels = [ i for i in range(FLAGS.num_classes + 1)], average='macro')
    r = recall_score(true, pred,labels = [ i for i in range(FLAGS.num_classes + 1)], average='macro')
    c = confusion_matrix(true, pred, labels = [ i for i in range(FLAGS.num_classes + 1)] )

    # set zero
    def setzero(array):
        if array < 0:
            return 0.0
        else:
            return array

    func = np.vectorize(setzero)
    logits = np.array(logits_collection)
    logits = func(logits)

    return f1, p, r, c, logits, labels_collection

if __name__ == "__main__":
    tf.app.run()
