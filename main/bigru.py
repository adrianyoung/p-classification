# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as py

class BiGRU:
    def __init__(self,is_training,num_classes,vocab_size,batch_size,embed_size,embed_size_p,learning_rate,decay_step,decay_rate,entity_window,sequence_length,
                 use_highway_flag,highway_layers,use_ranking_loss,lm,margin_plus,margin_minus,clip_gradients=5.0):
        """init all hyperparameter"""
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.initializer_rnn = tf.initializers.orthogonal()

        """Basic"""
        self.embed_size = embed_size
        self.embed_size_p = embed_size_p
        self.hidden_size_gru = 64
        self.sequence_length = sequence_length
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
        self.input_x_c1 = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_c1')
        self.input_x_c2 = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_c2')
        self.input_x_c3 = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_c3')
        self.input_x_w1 = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_w1')
        self.input_x_w2 = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_w2')
        self.input_x_w3 = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x_w3')
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
        with tf.name_scope("embedding"):
            self.Embedding_c = tf.get_variable("Embedding_c",shape=[self.vocab_size_c, self.embed_size],initializer=self.initializer)
            self.Embedding_w = tf.get_variable("Embedding_w",shape=[self.vocab_size_w, self.embed_size],initializer=self.initializer)
            self.Embedding_p = tf.get_variable("Embedding_p",shape=[self.vocab_size_p, self.embed_size_p],initializer=self.initializer)
            self.Embedding_o = tf.get_variable("Embedding_o",shape=[self.vocab_size_o, self.embed_size],initializer=self.initializer)
            self.Embedding_t = tf.get_variable("Embedding_t",shape=[self.vocab_size_t, self.embed_size_p],initializer=self.initializer)

        with tf.name_scope("gru_weights"):
            # GRU parameters:update gate related
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size*3+4*self.embed_size_p, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.U_z = tf.get_variable("U_z", shape=[self.hidden_size_gru, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size_gru], initializer=tf.zeros_initializer())
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size*3+4*self.embed_size_p, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.U_r = tf.get_variable("U_r", shape=[self.hidden_size_gru, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size_gru], initializer=tf.zeros_initializer())

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size*3+4*self.embed_size_p, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.U_h = tf.get_variable("U_h", shape=[self.hidden_size_gru, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size_gru], initializer=tf.zeros_initializer())

            self.W_z_b = tf.get_variable("W_z_b", shape=[self.embed_size*3+4*self.embed_size_p, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.U_z_b = tf.get_variable("U_z_b", shape=[self.hidden_size_gru, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.b_z_b = tf.get_variable("b_z_b", shape=[self.hidden_size_gru], initializer=tf.zeros_initializer())

            self.W_r_b = tf.get_variable("W_r_b", shape=[self.embed_size*3+4*self.embed_size_p, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.U_r_b = tf.get_variable("U_r_b", shape=[self.hidden_size_gru, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.b_r_b = tf.get_variable("b_r_b", shape=[self.hidden_size_gru], initializer=tf.constant_initializer(-1.0))

            self.W_h_b = tf.get_variable("W_h_b", shape=[self.embed_size*3+4*self.embed_size_p, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.U_h_b = tf.get_variable("U_h_b", shape=[self.hidden_size_gru, self.hidden_size_gru], initializer=self.initializer_rnn)
            self.b_h_b = tf.get_variable("b_h_b", shape=[self.hidden_size_gru], initializer=tf.zeros_initializer())

            self.init_state_h_gru = tf.get_variable("init_state_h_gru", shape=[self.hidden_size_gru], initializer=self.initializer)

        with tf.name_scope("attention_weights"):
            self.W_w_attention_gru = tf.get_variable("W_w_attention_gru", shape=[self.hidden_size_gru * 2, self.embed_size + self.embed_size_p*4], initializer=self.initializer)
            self.W_b_attention_gru = tf.get_variable("W_b_attention_gru", shape=[self.embed_size + self.embed_size_p*4],initializer=self.initializer)

        with tf.name_scope("MLP"):
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size_gru*4+self.embed_size*4+self.embed_size_p*8, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes], initializer=self.initializer)

    def attention_context(self):
        """
        L1: entity1 and its context with entity_window dims 300
        L2: entity2 and its context with entity_window dims 300
        L3: the PF of entity1 to entity2 dims 40
        L4: the PF of entity2 to entity1 dims 40
        L5: the hypernym of entity1 dims 300
        L6: the hypernym of entity2 dims 300
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
        embedded_objs_splitted = tf.split(self.embedded_objs, 2, axis=1)
        embedded_objs_squeeze = [tf.squeeze(x, axis=1) for x in embedded_objs_splitted]
        self.embedded_obj1 = tf.reshape(embedded_objs_squeeze[0], shape=[-1, self.embed_size])
        self.embedded_obj2 = tf.reshape(embedded_objs_squeeze[1], shape=[-1, self.embed_size])

        self.L1_context = tf.concat([L1, L3], axis=1)
        self.L2_context = tf.concat([L2, L4], axis=1)

    def embedding(self):
        """
        WF: Word Features
            ADD[CHAR, WORD] dims 300
        PF: Position Features dims 40
        """
        self.embedded_char1 = tf.nn.embedding_lookup(self.Embedding_c, self.input_x_c1)
        self.embedded_word1 = tf.nn.embedding_lookup(self.Embedding_w, self.input_x_w1)
        self.embedded_mix1 = tf.add(self.embedded_char1, self.embedded_word1)

        self.embedded_char2 = tf.nn.embedding_lookup(self.Embedding_c, self.input_x_c2)
        self.embedded_word2 = tf.nn.embedding_lookup(self.Embedding_w, self.input_x_w2)
        self.embedded_mix2 = tf.add(self.embedded_char2, self.embedded_word2)

        self.embedded_char3 = tf.nn.embedding_lookup(self.Embedding_c, self.input_x_c3)
        self.embedded_word3 = tf.nn.embedding_lookup(self.Embedding_w, self.input_x_w3)
        self.embedded_mix3 = tf.add(self.embedded_char3, self.embedded_word3)

        self.embedded_mix = tf.concat([self.embedded_mix1, self.embedded_mix2, self.embedded_mix3], axis=2)

        self.embedded_dropout = tf.nn.dropout(self.embedded_mix, self.dropout_keep_prob_spatial, noise_shape=[1,1,tf.shape(self.embedded_mix)[2]])
        # self.embedded_tag = tf.nn.embedding_lookup(self.Embedding_t, self.input_x_t)
        embedded_poss = []
        for idx in range(self.batch_size):
            self.embedded_pos_part = tf.nn.embedding_lookup(self.Embedding_p, self.input_x_p[idx:idx+1])
            self.embedded_pos_part = tf.reshape(self.embedded_pos_part, [-1, 4*self.embed_size_p])
            embedded_poss.append(self.embedded_pos_part)
        self.embedded_pos = tf.stack(embedded_poss, axis=0)
        # WF+PF
        # self.embedded_all = tf.concat([self.embedded_mix, self.embedded_tag, self.embedded_pos], axis=2)
        self.embedded_all = tf.concat([self.embedded_dropout, self.embedded_pos], axis=2)

    def attention_computation_gru(self, hidden_state):
        # MLP
        hidden_state_ = tf.reshape(hidden_state, shape=[-1, self.hidden_size_gru*2])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_, self.W_w_attention_gru) + self.W_b_attention_gru)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length, self.embed_size + self.embed_size_p*4])

        #compute similiarity
        sim1 = []
        sim2 = []
        for idx in range(self.batch_size):
            sim1.append(tf.multiply(hidden_representation[idx], self.L1_context[idx]))
            sim2.append(tf.multiply(hidden_representation[idx], self.L2_context[idx]))
        hidden_state_context_similiarity_l1 = tf.stack(sim1, axis=0)
        hidden_state_context_similiarity_l2 = tf.stack(sim2, axis=0)

        #get logits for each position
        attention_logits_l1 = tf.reduce_sum(hidden_state_context_similiarity_l1, axis=2)
        attention_logits_l2 = tf.reduce_sum(hidden_state_context_similiarity_l2, axis=2)

        #get max logit for each position
        attention_logits_max_l1 = tf.reduce_max(attention_logits_l1, axis=1, keepdims=True)
        attention_logits_max_l2 = tf.reduce_max(attention_logits_l2, axis=1, keepdims=True)

        #get possibility distribution
        p_attention_l1 = tf.nn.softmax(attention_logits_l1 - attention_logits_max_l1)
        p_attention_l2 = tf.nn.softmax(attention_logits_l2 - attention_logits_max_l2)

        #expand dims for matrix computation
        p_attention_expanded_l1 = tf.expand_dims(p_attention_l1, axis=2)
        p_attention_expanded_l2 = tf.expand_dims(p_attention_l2, axis=2)

        #get attention context
        L1_GRU_ATTENTION = tf.reduce_sum(tf.multiply(p_attention_expanded_l1, hidden_state), axis=1)
        L2_GRU_ATTENTION = tf.reduce_sum(tf.multiply(p_attention_expanded_l2, hidden_state), axis=1)

        return L1_GRU_ATTENTION, L2_GRU_ATTENTION

    def inference(self):
        """
        compute graph:
        1.Embedding-->2.BiGRU-->3.MLP

            (3-grams)
        1.1 Word Embedding
        1.2 Char Embedding
        1.3 Position Embedding

        2.1 BiGRU

        3.1 CONCAT [GRU_MAXPOOLING, GRU_MEANPOOLING, L1_CONTEXT, L2_CONTEXT, OBJ1, OBJ2]
        3.2 MLP
        """
        self.embedding()
        self.attention_context()

        BiGRU_hidden_states = self.BiGRU(self.embedded_all)
        BiGRU_hidden_states_splitted = tf.split(BiGRU_hidden_states, self.sequence_length, axis=1)
        BiGRU_hidden_states = tf.contrib.layers.layer_norm(BiGRU_hidden_states, scope='ln_gru')
        # GRU_LASTSTATE = tf.squeeze(BiGRU_hidden_states_splitted[-1], axis=1)
        # L1_GRU_ATTENTION, L2_GRU_ATTENTION = self.attention_computation_gru(BiGRU_hidden_states)
        GRU_MAXPOOLING = tf.keras.backend.max(BiGRU_hidden_states, axis=1)
        GRU_MEANPOOLING = tf.keras.backend.mean(BiGRU_hidden_states, axis=1)

        sentence_representation = tf.concat([GRU_MAXPOOLING, GRU_MEANPOOLING, self.L1_context, self.L2_context, self.embedded_obj1, self.embedded_obj2], axis=1)
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(sentence_representation, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('output'):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
            tf.summary.histogram("logits_value", logits)
        return logits

    def gru_single_step(self, Xt, h_t_minus_1):
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r)
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) + r_t * ( tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def gru_single_step_backward(self, Xt, h_t_minus_1):
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_b) + tf.matmul(h_t_minus_1, self.U_z_b) + self.b_z_b)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_b) + tf.matmul(h_t_minus_1, self.U_r_b) + self.b_r_b)
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_b) + r_t * ( tf.matmul(h_t_minus_1, self.U_h_b)) + self.b_h_b)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def BiGRU(self, representation):
        representation_splitted = tf.split(representation, self.sequence_length, axis=1)
        representation_squeeze = [tf.squeeze(x, axis=1) for x in representation_splitted]
        #h_t = tf.ones((self.batch_size, self.hidden_size_gru))
        h_t = tf.stack([ self.init_state_h_gru for _ in range(self.batch_size) ], axis=0)

        h_t_forward_list = []
        h_t_backward_list = []

        for time_step, Xt in enumerate(representation_squeeze):
            h_t = self.gru_single_step(Xt,h_t)
            h_t_forward_list.append(h_t)

        for time_step, Xt in enumerate(reversed(representation_squeeze)):
            h_t = self.gru_single_step_backward(Xt,h_t)
            h_t_backward_list.append(h_t)

        bigru_hidden_state = [ tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(h_t_forward_list, h_t_backward_list)]
        bigru_hidden_state = tf.stack(bigru_hidden_state, axis=1)

        return bigru_hidden_state

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
            tf.summary.scalar('ranking_loss', loss)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            tf.summary.scalar('l2_loss', l2_losses)
            loss = loss + l2_losses
            tf.summary.scalar('total_loss', loss)
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
            tvars = [tvar for tvar in tf.trainable_variables() if ('Embedding_c' not in tvar.name) and ('Embedding_w' not in tvar.name)]
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
