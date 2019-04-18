# -*- coding: utf-8 -*-
import os
import math
import gc
import random
import tensorflow as tf
import numpy as np
from data_utils import *
from utils import *
from cnn import CNN
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("ckpt_dir",get_config_values('model','ckpt'),"checkpoint location for the model")
tf.app.flags.DEFINE_string("log_path",get_config_values('model','log'),"path of summary log.")

tf.app.flags.DEFINE_integer("sequence_length", 150,"the max length of a sentence in documents")
tf.app.flags.DEFINE_integer("entity_window", 15, "the window of the entity and its context")
tf.app.flags.DEFINE_integer("distance", 300, "the distance from char to entity")
tf.app.flags.DEFINE_integer("num_classes", 50,"relation type total number")

tf.app.flags.DEFINE_string("char_embedding_model_path",get_config_values('vector','w2v_char'),"char's vocabulary and vectors")
tf.app.flags.DEFINE_string("word_embedding_model_path",get_config_values('vector','glv_segm'),"word's vocabulary and vectors")

tf.app.flags.DEFINE_boolean("is_training", True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("embed_size", 300,"word embedding size")
tf.app.flags.DEFINE_integer("embed_size_p", 10,"position embedding size")
tf.app.flags.DEFINE_integer("sentence_size", 300,"the size of the sentence level embedding")

tf.app.flags.DEFINE_integer("batch_size", 160, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("decay_step", 5000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.3, "Rate of decay for learning rate.")

tf.app.flags.DEFINE_integer("is_frozen_step", 400, "how many steps before fine-tuning the embedding.")

tf.app.flags.DEFINE_boolean("use_highway_flag", False,"using highway network or not.")
tf.app.flags.DEFINE_integer("highway_layers", 1,"How many layers in highway network.")

tf.app.flags.DEFINE_boolean("use_ranking_loss", True,"using ranking loss or not.")
tf.app.flags.DEFINE_float("lm", 1.0,"lambda in ranking loss")
tf.app.flags.DEFINE_float("margin_plus", 2.5,"margin value for postive in ranking loss")
tf.app.flags.DEFINE_float("margin_minus", 0.5,"margin value for negative in ranking loss")

tf.app.flags.DEFINE_integer("num_epochs", 10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1,"Validate every validate_every epochs.")

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
    tf.set_random_seed(2019)
    with tf.Session(config=config) as sess:
        # instantiate model
        Model = CNN(FLAGS.is_training,FLAGS.num_classes,vocab_size,FLAGS.batch_size,FLAGS.embed_size,FLAGS.embed_size_p,FLAGS.learning_rate,FLAGS.decay_step,FLAGS.decay_rate,FLAGS.entity_window,
                FLAGS.sequence_length,filter_sizes,feature_map,FLAGS.use_highway_flag,FLAGS.highway_layers,FLAGS.sentence_size,FLAGS.use_ranking_loss,FLAGS.lm,FLAGS.margin_plus,FLAGS.margin_minus)
        # initialize saver
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            logger.info("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            summary_writer = tf.summary.FileWriter(logdir=FLAGS.log_path, graph=sess.graph)
        else:
            logger.info('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(logdir=FLAGS.log_path, graph=sess.graph)
            assign_pretrained_embedding(sess,Model,char_vocab,vocab_size[0],char_embedding,'char',FLAGS)
            assign_pretrained_embedding(sess,Model,word_vocab,vocab_size[1],word_embedding,'word',FLAGS)
            assign_position_embedding(sess,Model,vocab_size[2],FLAGS)
            assign_objects_embedding(sess,Model,vocab_size[3],FLAGS)
            assign_postags_embedding(sess,Model,vocab_size[4],FLAGS)
            del char_embedding
            del word_embedding
            gc.collect()
        curr_epoch=sess.run(Model.epoch_step)

        batch_size=FLAGS.batch_size
        iteration=0
        file_train = get_config_values('dataset', 'train')
        data_train = load_json(file_train)
        data_list = Process([data_train], mode='train')
        Range = int (len(data_list) * 0.8)
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            train_data = Batch(data_list[:Range], char_vocab,word_vocab,schemas_vocab,pos_map,postag_vocab,FLAGS)
            loss, counter =  0.0, 0
            for batch in tqdm(train_data):
                iteration=iteration+1
                #if epoch==0 and counter==0:
                #    logger.info("train_batch char {0}".format(batch['char_sentence']))
                #    logger.info("train_batch mix {0}".format(batch['mix_sentence']))
                #    logger.info("train_batch lexical {0}".format(batch['lexical']))
                feed_dict={}

                feed_dict[Model.dropout_keep_prob] = 0.5
                feed_dict[Model.dropout_keep_prob_spatial] = 0.1
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

                train_op = Model.train_op_frozen if FLAGS.is_frozen_step > iteration and epoch == 0 else Model.train_op
                curr_loss,lr,_,_,summary,logits=sess.run([Model.loss_val,Model.learning_rate,train_op,Model.global_increment,Model.merge,Model.logits],feed_dict)
                summary_writer.add_summary(summary, global_step=iteration)
                loss,counter=loss+curr_loss,counter+1
                if counter %50==0:
                    print ("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))
                if iteration % 5000 == 0:
                    valid_data = Batch(data_list[Range:], char_vocab, word_vocab, schemas_vocab, pos_map, postag_vocab, FLAGS)
                    eval_loss, f1_score, p_score, r_score = do_eval(sess, valid_data, Model)
                    print ("Epoch %d Validation Loss:%.3f\t F1_score:%.3f" % (epoch, eval_loss, f1_score))
                    print ("Epoch %d Validation precision_score:%.3f\t recall_score:%.3f" % (epoch, p_score, r_score))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)
            #epoch increment
            logger.info("going to increment epoch counter....")
            sess.run(Model.epoch_increment)
            if epoch % FLAGS.validate_every==0:
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)
        summary_writer.close()

def do_eval(sess, valid_data, Model):
    y_true = []
    y_pred = []
    eval_loss, eval_counter= 0.0, 0
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
        feed_dict[Model.tst] = not FLAGS.is_training

        curr_eval_loss,logits=sess.run([Model.loss_val,Model.logits],feed_dict)

        # y_true.extend(np.argsort(batch['label_sentence'])[:,-1])
        # y_pred.extend(np.argsort(logits)[:,-1])

        true = np.argsort(batch['label_sentence'])[:,-1]
        index_NA = np.argwhere((np.sum((batch['label_sentence']==0), axis=1) == FLAGS.num_classes))
        for index in index_NA:
            true[index] = FLAGS.num_classes
        y_true.extend(true)

        pred = np.argsort(logits)[:,-1]
        index_NA = np.argwhere((np.sum((logits<0), axis=1) == FLAGS.num_classes))
        for index in index_NA:
            pred[index] = FLAGS.num_classes
        y_pred.extend(pred)

        eval_loss += curr_eval_loss
        eval_counter += 1

    f1 = f1_score(y_true, y_pred, average='macro')
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')

    return eval_loss/float(eval_counter), f1, p, r

def assign_pretrained_embedding(sess, Model, vocab, vocab_size, embedding_index, mode, FLAGS):
    embedding_model_path = FLAGS.char_embedding_model_path if mode == 'char' else FLAGS.word_embedding_model_path
    logger.info("using pre-trained word emebedding.started.model_path:{0}".format(embedding_model_path))
    count_exist = 0
    count_not_exist = 0
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size, dtype=np.float32)  # assign empty for first word:'PAD'
    embedding_mean = embedding_index.vectors.mean(axis=0)
    for i in range(1, vocab_size):  # loop each word
        word = vocab.id2obj(i)
        embedding = None
        try:
            embedding = embedding_index[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = embedding_mean
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    Embedding = Model.Embedding_c if mode == 'char' else Model.Embedding_w
    t_assign_embedding = tf.assign(Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    logger.info("word. exists embedding: {0} ;word not exist embedding: {1}".format(count_exist,count_not_exist))
    logger.info("using pre-trained word embedding.ended...")

def assign_objects_embedding(sess, Model, vocab_size, FLAGS):
    logger.info("initializing the objects embedding...")
    object_embedding = tf.random_uniform([vocab_size, FLAGS.embed_size],-0.1,0.1)
    t_assign_embedding = tf.assign(Model.Embedding_o, object_embedding)
    sess.run(t_assign_embedding)
    logger.info("assign object embedding.ended...")

def assign_position_embedding(sess, Model, vocab_size, FLAGS):
    logger.info("initializing the position embedding...")
    position_embedding = tf.random_uniform([vocab_size, FLAGS.embed_size_p],-0.1,0.1)
    t_assign_embedding = tf.assign(Model.Embedding_p, position_embedding)
    sess.run(t_assign_embedding)
    logger.info("assign position embedding.ended...")

def assign_postags_embedding(sess, Model, vocab_size, FLAGS):
    logger.info("initializing the postag embedding...")
    postag_embedding = tf.random_uniform([vocab_size, FLAGS.embed_size_p],-0.1,0.1)
    t_assign_embedding = tf.assign(Model.Embedding_t, postag_embedding)
    sess.run(t_assign_embedding)
    logger.info("assign postag embedding.ended...")

if __name__ == "__main__":
    tf.app.run()
