import os
import time
import random
import numpy as np
import make_train_set
import tensorflow as tf
from copy import deepcopy
from itertools import chain

# 학습용 데이터
source = make_train_set.input
target = make_train_set.output
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
y = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_decoder_word])

# encoder
encoder_x = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_encoder_word])
source_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[number_of_document])
embedding_matrix = tf.get_variable(name="embeding_matrix", shape=[voca_size, word_embedding_size],
                                   dtype=tf.float32, initializer=tf.truncated_normal_initializer())
encoder_emb_inp = tf.nn.embedding_lookup(params=embedding_matrix, ids=encoder_x, name="encoder_emb_inp")
encoder_cells = [tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size,
                                         initializer=tf.contrib.layers.variance_scaling_initializer()) for n in range(num_encoder_layer)]
stacked_encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
encoder_outputs, encoder_final_hidden_state = tf.nn.dynamic_rnn(cell=stacked_encoder_cell, inputs=encoder_emb_inp,
                                                                sequence_length=source_sequence_lengths, dtype=tf.float32,
                                                                scope="encoder_LSTM")

# decoder
decoder_x = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_decoder_word])
decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[number_of_document])
decoder_emb_inp = tf.nn.embedding_lookup(params=embedding_matrix, ids=decoder_x, name="decoder_embedded_x")
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)
decoder_cells = [tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size,
                                         initializer=tf.contrib.layers.variance_scaling_initializer()) for n in range(num_decoder_layer)]
stacked_decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
projection_layer = tf.layers.Dense(voca_size)
decoder = tf.contrib.seq2seq.BasicDecoder(stacked_decoder_cell, helper, encoder_final_hidden_state, output_layer=projection_layer)
decoder_outputs, decoder_final_hidden_state, decoder_final_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
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

start_time = time.time()
for i in range(20000):
    _, print_loss = sess.run([update_step,train_loss], feed_dict(source, target, pad_token, batch_size=256))
    if i%500 == 0 :
        tr_loss, perplexity, lr, show_sample = sess.run([train_loss, tf.exp(train_loss), learning_rate, tf.argmax(tf.nn.softmax(logits), axis=2)],
                                        feed_dict(source, target, pad_token, batch_size=1))
        # save
        saver.save(sess, os.getcwd()+"/save/"+str(num_encoder_layer)+"enc_"+str(num_decoder_layer)+"dec_"+str(i)+"iter_"+str(perplexity)+"plx_"+str(tr_loss)+"loss_model.ckpt")
        show_time = time.time() - start_time
        print("iteration :", i, "소요시간 :", round(show_time, 3), "loss :", tr_loss, "perplexity :", perplexity,
              "learning_rate :", lr, "sentence :", make_train_set.c_util.compose_sentence(list(show_sample[0])))

end_time = time.time() - start_time
print(" 총 소요시간 :", round(end_time, 3), "초")







# inference
inf_text = [['<sos>', '나', ' ', '는', ' ', '자랑스러워', '<eos>']]
fedic_inf_encoder_seq_len_except_pad = [len(i) for i in inf_text]
fedic_inf_encoder_x = []
for sentence in inf_text:
    fedic_inf_encoder_x.append([dic.index(i) for i in sentence])
fedic_inf_decoder_seq_len_except_pad = [1]
fedic_inf_decoder_x = [[dic.index('<sos>')]]

result = []
next_word = '<sos>'

for count in range(50):
    next_word = sess.run(tf.argmax(logits, axis=2), feed_dict={encoder_x: fedic_inf_encoder_x,
                                                               source_sequence_lengths: fedic_inf_encoder_seq_len_except_pad,
                                                               decoder_x: fedic_inf_decoder_x,
                                                               decoder_lengths: fedic_inf_decoder_seq_len_except_pad})

    fedic_inf_decoder_x = [list(chain(*([[dic.index('<sos>')]] + next_word.tolist())))]
    fedic_inf_decoder_seq_len_except_pad = [len(fedic_inf_decoder_x[0])]
    next_word = dic[fedic_inf_decoder_x[0][-1]]
    if next_word == '<eos>':
        break

for inf_sen in fedic_inf_decoder_x:
    for inf_word in inf_sen:
        result.append(dic[inf_word])

result

## class화 (to do...)
