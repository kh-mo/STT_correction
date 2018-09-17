import os
import time
import random
import numpy as np
import make_train_set
import tensorflow as tf
from copy import deepcopy
from itertools import chain

# 학습용 데이터
tr_source = make_train_set.tr_input
tr_target = make_train_set.tr_output
ts_source = make_train_set.ts_input
ts_target = make_train_set.ts_output
pad_token = [key for key, value in make_train_set.c_util.special.items() if value == '<pad>']

# 배치 함수
def feed_dict(source, target, pad_token, batch_size):
    idx = [random.randint(0, len(source)-1) for p in range(0, batch_size)]
    source_for_batch = deepcopy(source)
    target_for_batch = deepcopy(target)

    batch_x = [source_for_batch[i] for i in idx]
    batch_y = [target_for_batch[i] for i in idx]

    batch_x_length = [len(i) for i in batch_x] # batch_x_length == fedic_encoder_seq_len_except_pad
    batch_y_length = [len(i)-1 for i in batch_y] # batch_y_length == fedic_decoder_seq_len_except_pad

    for batch_x_idx, input_sentence in enumerate(batch_x):
        if len(input_sentence) == max(batch_x_length):
            pass
        else:
            input_sentence += pad_token*(max(batch_x_length) - len(input_sentence))
            batch_x[batch_x_idx] = input_sentence

    target_label = []
    for batch_y_idx, output_sentence in enumerate(batch_y):
        if (len(output_sentence)-1) == max(batch_y_length):
            batch_y[batch_y_idx] = output_sentence[:-1]
            target_label.append(output_sentence[1:])
        else:
            output_sentence += pad_token*(max(batch_y_length) - len(output_sentence) + 1)
            batch_y[batch_y_idx] = output_sentence[:-1]
            target_label.append(output_sentence[1:])

    return {encoder_x: batch_x, source_sequence_lengths: batch_x_length,
            decoder_x: batch_y, decoder_lengths: batch_y_length, y: target_label}

# perplexity 계산 함수
def get_perplexity(input):
    # PPL(w1....wn)
    # = P(w1...wn)^-1/n
    # = ((1/P(w1))^-1/n).......
    px = tf.reduce_max(tf.nn.softmax(input), axis=2)
    reversed_px = tf.reciprocal(px)
    number_of_word = tf.shape(px)[1]
    sqrt_n = tf.reciprocal(tf.cast(tf.fill([number_of_word,], number_of_word), dtype=tf.float32))
    result = tf.reduce_prod(tf.pow(reversed_px, sqrt_n), axis=1)
    return result

# 하이퍼파라미터
number_of_document = None # batch_size와 동일
number_of_encoder_word = None
number_of_decoder_word = None
word_embedding_size = 200
lstm_hidden_size = 200
voca_size = make_train_set.c_util.voca_size
max_gradient_norm = 0.1
starter_learning_rate = 1e-4
num_encoder_layer = 4
num_decoder_layer = 4

# label
y = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_decoder_word], name="y")

# encoder
encoder_x = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_encoder_word], name="encoder_x")
source_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[number_of_document], name="encoder_length")
embedding_matrix = tf.get_variable(name="embeding_matrix", shape=[voca_size, word_embedding_size],
                                   dtype=tf.float32, initializer=tf.truncated_normal_initializer())
encoder_emb_inp = tf.nn.embedding_lookup(params=embedding_matrix, ids=encoder_x, name="encoder_emb_inp")
encoder_cells = [tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size,
                                         initializer=tf.contrib.layers.variance_scaling_initializer(),name="encoder_cell"+str(n)) for n in range(num_encoder_layer)]
stacked_encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
encoder_outputs, encoder_final_hidden_state = tf.nn.dynamic_rnn(cell=stacked_encoder_cell, inputs=encoder_emb_inp,
                                                                sequence_length=source_sequence_lengths, dtype=tf.float32,
                                                                scope="encoder_LSTM")

# decoder
decoder_x = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_decoder_word], name="decoder_x")
decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[number_of_document], name="decoder_length")
decoder_emb_inp = tf.nn.embedding_lookup(params=embedding_matrix, ids=decoder_x, name="decoder_embedded_x")
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)
decoder_cells = [tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size,
                                         initializer=tf.contrib.layers.variance_scaling_initializer()) for n in range(num_decoder_layer)]
stacked_decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
projection_layer = tf.layers.Dense(voca_size)
decoder = tf.contrib.seq2seq.BasicDecoder(stacked_decoder_cell, helper, encoder_final_hidden_state, output_layer=projection_layer)
decoder_outputs, decoder_final_hidden_state, decoder_final_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, scope="decoder_for_logit")
logits = decoder_outputs.rnn_output

# loss
crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, voca_size), logits=logits)
train_loss = tf.reduce_sum(crossent)

# calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm) # gradient exploding 방지

# optimization
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.9, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

# model save
saver = tf.train.Saver()

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tensorboard
train_writer = tf.summary.FileWriter(os.getcwd() + "/save/train", sess.graph)

start_time = time.time()
for i in range(1):
    _, print_loss = sess.run([update_step,train_loss], feed_dict(tr_source, tr_target, pad_token, batch_size=256))
    if i%500 == 0 :
        ts_loss, perplexity, lr, show_sample = sess.run([train_loss, tf.reduce_mean(get_perplexity(logits)), learning_rate, tf.argmax(tf.nn.softmax(logits), axis=2)],
                                        feed_dict(ts_source, ts_target, pad_token, batch_size=len(ts_source)))
        # save
        saver.save(sess, os.getcwd()+"/save/"+str(num_encoder_layer)+"enc_"+str(num_decoder_layer)+"dec_"+str(i)+"iter_"+str(perplexity)+"plx_"+str(ts_loss)+"loss_model.ckpt")
        show_time = time.time() - start_time
        print("iteration :", i, "소요시간 :", round(show_time, 3), "loss :", ts_loss, "perplexity :", perplexity,
              "learning_rate :", lr, "sentence :", make_train_set.c_util.compose_sentence(list(show_sample[0])))

end_time = time.time() - start_time
print(" 총 소요시간 :", round(end_time, 3), "초")
