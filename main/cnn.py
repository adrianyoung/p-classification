# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as py

class CNN:
    def __init__(self,is_training,num_classes,vocab_size,batch_size,embed_size,embed_size_p,learning_rate,decay_step,decay_rate,entity_window,sequence_length,filter_sizes,
                 feature_map,use_highway_flag,highway_layers,sentence_size,use_ranking_loss,lm,margin_plus,margin_minus,clip_gradients=5.0):
        """init all hyperparameter"""
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        """Basic"""
        self.embed_size = embed_size
        self.embed_size_p = embed_size_p
        self.sentence_size = sentence_size
        self.vocab_size_c = vocab_size[0]
        self.vocab_size_w = vocab_size[1]
        self.vocab_size_p = vocab_size[2]
        self.vocab_size_o = vocab_size[3]
        self.vocab_size_t = vocab_size[4]
        self.num_classes = num_classes

        """learning_rate"""
        self.is_training = is_training
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        self.decay_step = decay_step
        self.decay_rate = decay_rate

        """Overfit"""
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.dropout_keep_prob_spatial = tf.placeholder(tf.float32, name='dropout_keep_prob_spatial')
        self.clip_gradients = clip_gradients

        """Lexical"""
        self.entity_window = entity_window

        """CNN"""
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.feature_map = feature_map

        """Highway Network"""
        self.use_highway_flag = use_highway_flag
        self.highway_layers = highway_layers

        """Ranking Loss"""
        self.use_ranking_loss = use_ranking_loss
        self.lm = tf.constant(lm)
        self.margin_plus = tf.constant(margin_plus)
        self.margin_minus = tf.constant(margin_minus)

        """Input"""
        self.tst = tf.placeholder(tf.bool, name='is_training_flag')
        self.input_x_c = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_c')
        self.input_x_w = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_w')
        self.input_x_t = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_t')
        self.input_x_p = tf.placeholder(tf.int32, [None, 4, self.sequence_length], name='input_x_p')
        self.input_x_p_l1 = tf.placeholder(tf.int32, [None, 4], name='input_x_p_l1')
        self.input_x_p_l2 = tf.placeholder(tf.int32, [None, 4], name='input_x_p_l2')
        self.input_x_o = tf.placeholder(tf.int32, [None, 2], name='input_x_o')
        self.input_x_c_l1 = tf.placeholder(tf.int32, [None, self.entity_window], name='input_x_c_l1')
        self.input_x_c_l2 = tf.placeholder(tf.int32, [None, self.entity_window], name='input_x_c_l2')
        self.input_x_w_l1 = tf.placeholder(tf.int32, [None, self.entity_window], name='input_x_w_l1')
        self.input_x_w_l2 = tf.placeholder(tf.int32, [None, self.entity_window], name='input_x_w_l2')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y')

        """Count"""
        self.global_step = tf.Variable(0, trainable=False, name='Global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.global_increment = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1)))

        """Process"""
        self.instantiate_weights()

        """Logits"""
        self.logits = self.inference()

        if not self.is_training:
            return

        if self.use_ranking_loss:
            self.loss_val = self.loss_ranking()
        else:
            self.loss_val = self.loss_softmax()

        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_rate, staircase=True)
        self.train_op = self.train()
        self.train_op_frozen = self.train_frozen()
        self.merge = tf.summary.merge_all()

    def instantiate_weights(self):
        """ define weights """
        with tf.name_scope("Embedding"):
            self.Embedding_c = tf.get_variable("Embedding_c",shape=[self.vocab_size_c, self.embed_size],initializer=self.initializer)
            self.Embedding_w = tf.get_variable("Embedding_w",shape=[self.vocab_size_w, self.embed_size],initializer=self.initializer)
            self.Embedding_p = tf.get_variable("Embedding_p",shape=[self.vocab_size_p, self.embed_size_p],initializer=self.initializer)
            self.Embedding_o = tf.get_variable("Embedding_o",shape=[self.vocab_size_o, self.embed_size],initializer=self.initializer)
            self.Embedding_t = tf.get_variable("Embedding_t",shape=[self.vocab_size_t, self.embed_size_p],initializer=self.initializer)

        with tf.name_scope("MLP"):
            self.W_projection = tf.get_variable("W_projection",shape=[self.embed_size*4 + self.embed_size_p*8 + self.sentence_size, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes], initializer=self.initializer)

    def lexical_features(self):
        """
        L1: entity1 and its context with entity_window
        L2: entity2 and its context with entity_window
        L3: the PF of entity1 to entity2
        L4: the PF of entity2 to entity1
        L5: the hypernym of entity1 and entity2
        """

        self.embedded_char_l1 = tf.nn.embedding_lookup(self.Embedding_c, self.input_x_c_l1)
        self.embedded_word_l1 = tf.nn.embedding_lookup(self.Embedding_w, self.input_x_w_l1)
        self.embedded_char_l2 = tf.nn.embedding_lookup(self.Embedding_c, self.input_x_c_l2)
        self.embedded_word_l2 = tf.nn.embedding_lookup(self.Embedding_w, self.input_x_w_l2)

        self.embedded_l1 = tf.add(self.embedded_char_l1, self.embedded_word_l1)
        self.embedded_l2 = tf.add(self.embedded_char_l2, self.embedded_word_l2)
        L1 = tf.reduce_mean(self.embedded_l1, axis=1)
        L2 = tf.reduce_mean(self.embedded_l2, axis=1)

        self.embedded_pos_entity1 = tf.nn.embedding_lookup(self.Embedding_p, self.input_x_p_l1)
        self.embedded_pos_entity2 = tf.nn.embedding_lookup(self.Embedding_p, self.input_x_p_l2)
        L3 = tf.reshape(self.embedded_pos_entity1, shape=[-1, self.embed_size_p*4])
        L4 = tf.reshape(self.embedded_pos_entity2, shape=[-1, self.embed_size_p*4])

        self.embedded_objs = tf.nn.embedding_lookup(self.Embedding_o, self.input_x_o)
        L5 = tf.reshape(self.embedded_objs, [-1, self.embed_size*2])

        features = tf.concat([L1,L3,L2,L4,L5], axis=1)

        with tf.name_scope("dropout"):
            features = tf.nn.dropout(features, keep_prob=self.dropout_keep_prob)
        return features

    def sentence_features(self):
        """
        embedding-->conv-->bn-->max-pooling
        """
        embedded_all_expanded = tf.expand_dims(self.embedded_all, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-{0}".format(filter_size)):
                filter = tf.get_variable("filter-{0}".format(filter_size), [filter_size, self.embed_size+5*self.embed_size_p, 1, self.feature_map[i]], initializer=self.initializer)
                conv = tf.nn.conv2d(embedded_all_expanded, filter, strides=[1,1,1,1], padding="VALID", name="conv")
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.tst)
                b = tf.get_variable("b-{0}".format(filter_size), [self.feature_map[i]])
                h = tf.nn.relu(tf.nn.bias_add(conv,b),"relu")
                pooled = tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1],strides=[1,1,1,1], padding='VALID',name="pool")
                pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.sentence_size])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)
        return h_drop

    def embedding(self):
        self.embedded_char = tf.nn.embedding_lookup(self.Embedding_c, self.input_x_c)
        self.embedded_word = tf.nn.embedding_lookup(self.Embedding_w, self.input_x_w)
        self.embedded_tag = tf.nn.embedding_lookup(self.Embedding_t, self.input_x_t)
        embedded_poss = []
        for idx in range(self.batch_size):
            self.embedded_pos_part = tf.nn.embedding_lookup(self.Embedding_p, self.input_x_p[idx:idx+1])
            self.embedded_pos_part = tf.reshape(self.embedded_pos_part, [-1, 4*self.embed_size_p])
            embedded_poss.append(self.embedded_pos_part)
        self.embedded_pos = tf.stack(embedded_poss, axis=0)
        # WF+PF
        self.embedded_mix = tf.add(self.embedded_char, self.embedded_word)
        # self.embedded_dropout = tf.nn.dropout(self.embedded_mix, self.dropout_keep_prob_spatial, noise_shape=[1,1,tf.shape(self.embedded_mix)[2]])
        self.embedded_all = tf.concat([self.embedded_mix, self.embedded_tag, self.embedded_pos], axis=2)

    def inference(self):
        """
        compute graph:
        1.Embedding-->2.Features Layer-->3.Concat[sentence, lexical]-->4.MLP

        1.1 Word Embedding ( WF )
        1.2 Position Embedding ( PF )
        1.3 Object Embedding ( L5 )

        2.1 sentence level
        WF+PF-->CNN-->max pooling-->sentence_features
        2.2 lexical level
        Conact[avg(L1) + avg(L2) +L3 + L4 + L5]-->lexical_features

        """
        # init embedding
        self.embedding()
        # get features
        sentence = self.sentence_features()
        lexical = self.lexical_features()
        # concat
        h = tf.concat([sentence, lexical], axis=1)

        # MLP
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
            tf.summary.histogram("logits_value", logits)
        return logits

    def loss_softmax(self, l2_lambda=0.00001):
        with tf.name_scope("loss_softmax"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_ranking(self, l2_lambda=0.00001):
        with tf.name_scope("loss_ranking"):
            loss = self.ranking_loss()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def ranking_loss(self):
        lm = self.lm
        m_plus = self.margin_plus
        m_minus = self.margin_minus
        labels = self.input_y
        logits = self.logits
        batch_size = self.batch_size

        L = tf.constant(0.0)
        i = tf.constant(0)
        cond = lambda i, L: tf.less(i, batch_size)

        def loop_body(i, L):
            #judge NA class
            zeros = tf.zeros_like(labels[i,:], dtype=labels.dtype)
            conditon = tf.equal(tf.reduce_sum(labels[i,:]), tf.reduce_sum(zeros))
            # tf.scalar_mul(labels[i,:], zeros)
            is_NA = tf.cond(conditon, lambda:tf.constant(0.0, dtype=tf.float32), lambda:tf.constant(1.0, dtype=tf.float32))
            # cplus = labels[i] #positive class label index
            _, cplus_indices = tf.nn.top_k(labels[i,:], k=1) #positive class label index
            cplus = cplus_indices[0]
            #taking most informative negative class, use 2nd argmax
            _, cminus_indices = tf.nn.top_k(logits[i,:], k=2)
            cminus = tf.cond(tf.equal(cplus, cminus_indices[0]), lambda: cminus_indices[1], lambda: cminus_indices[0])
            splus = logits[i,cplus] #score for gold class
            sminus = logits[i,cminus] #score for negative class
            l = is_NA * tf.log((1.0+tf.exp((lm*(m_plus-splus))))) + \
                tf.log((1.0+tf.exp((lm*(m_minus+sminus)))))
            return [tf.add(i, 1), tf.add(L,l)]

        _, L = tf.while_loop(cond, loop_body, loop_vars=[i,L])
        nbatch = tf.to_float(batch_size)
        loss = L/nbatch
        return loss

    def train_frozen(self):
        with tf.name_scope("train_op_frozen"):
            learning_rate = self.learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.99)
            tvars = [tvar for tvar in tf.trainable_variables() if 'embedding' not in tvar.name]
            gradients, variables = zip(*optimizer.compute_gradients(self.loss_val, tvars))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def train(self):
        with tf.name_scope("train_op"):
            learning_rate = self.learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.99)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
