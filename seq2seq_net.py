import numpy as np
import tensorflow as tf
from itertools import chain

# 학습용 데이터
source = [['<sos>', '타임', ' ', '아웃', '<eos>'],
          ['<sos>', '너무', ' ', '자랑스러워', '<eos>'],
          ['<sos>', '판다', '랑', ' ', '나', '는', ' ', '오늘', ' ', '못', ' ', '갈', ' ', '거', ' ', '같아', '<eos>']]

target = [['<sos>', 'time', ' ', 'out', '<eos>'],
          ['<sos>', 'i', ' ', 'am', ' ', 'proud', ' ', 'of', ' ', 'you', '<eos>'],
          ['<sos>', 'pan', ' ', 'and', ' ', 'i', ' ', 'can\'t', ' ', 'make', ' ', 'it', ' ', 'today', '<eos>']]

dic = list(set(list(chain(*source)) + list(chain(*target)))) + ["<pad>"]

# 배치 구성
fedic_encoder_seq_len_except_pad = [len(i) for i in source]
fedic_encoder_x = []
for sentence in source:
    if len(sentence) == max(fedic_encoder_seq_len_except_pad):
        fedic_encoder_x.append([dic.index(i) for i in sentence])
    else:
        sentence += ['<pad>']*(max(fedic_encoder_seq_len_except_pad) - len(sentence))
        fedic_encoder_x.append([dic.index(i) for i in sentence])

fedic_decoder_seq_len_except_pad = [len(i)-1 for i in target]
fedic_decoder_x = []
real_label = []
for sentence in target:
    if (len(sentence)-1) == max(fedic_decoder_seq_len_except_pad):
        fedic_decoder_x.append([dic.index(i) for i in sentence[:-1]])
        real_label.append([dic.index(i) for i in sentence[1:]])
    else:
        sentence += ['<pad>']*(max(fedic_decoder_seq_len_except_pad) - len(sentence) + 1)
        fedic_decoder_x.append([dic.index(i) for i in sentence[:-1]])
        real_label.append([dic.index(i) for i in sentence[1:]])

# 파라미터 설정
number_of_document = None
number_of_encoder_word = None
number_of_decoder_word = None
lstm_hidden_size = 200
voca_size = len(dic)
word_embedding_size = 30
max_gradient_norm = 0.1
learning_rate = 1e-4

# label
y = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_decoder_word])

# encoder
encoder_x = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_encoder_word])
source_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[number_of_document])
embedding_matrix = tf.get_variable(name="embeding_matrix", shape=[voca_size, word_embedding_size],
                                   dtype=tf.float32, initializer=tf.truncated_normal_initializer())
encoder_emb_inp = tf.nn.embedding_lookup(params=embedding_matrix, ids=encoder_x, name="encoder_emb_inp")
encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size,
                                       initializer=tf.contrib.layers.variance_scaling_initializer())
encoder_outputs, encoder_final_hidden_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_emb_inp,
                                                                sequence_length=source_sequence_lengths, dtype=tf.float32,
                                                                scope="encoder_LSTM")

# decoder
decoder_x = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_decoder_word])
decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[number_of_document])
decoder_emb_inp = tf.nn.embedding_lookup(params=embedding_matrix, ids=decoder_x, name="decoder_embedded_x")
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)
decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size, initializer=tf.contrib.layers.variance_scaling_initializer())
projection_layer = tf.layers.Dense(voca_size)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_final_hidden_state, output_layer=projection_layer)
decoder_outputs, decoder_final_hidden_state, decoder_final_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
logits = decoder_outputs.rnn_output

# loss
crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y,voca_size), logits=logits)
train_loss = tf.reduce_sum(crossent)

# calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm) # gradient exploding 방지

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2000):
    _, print_loss = sess.run([update_step,train_loss], feed_dict={encoder_x:fedic_encoder_x,
                                                                  source_sequence_lengths:fedic_encoder_seq_len_except_pad,
                                                                  decoder_x:fedic_decoder_x,
                                                                  decoder_lengths:fedic_decoder_seq_len_except_pad,
                                                                  y:real_label})
    if i%500 == 0 :
        a1, a2 = sess.run([tf.argmax(tf.nn.softmax(logits), axis=2), train_loss], feed_dict={encoder_x: fedic_encoder_x,
                                                                                             source_sequence_lengths: fedic_encoder_seq_len_except_pad,
                                                                                             decoder_x: fedic_decoder_x,
                                                                                             decoder_lengths: fedic_decoder_seq_len_except_pad, y: real_label})
        print_list = []
        for i in a1:
            sen = ""
            for j in i:
                sen += dic[j]+" "
            print_list.append(sen)

        print("sentence : ", print_list, "loss : ", a2)

# model save
saver = tf.train.Saver()
saver.save(sess, "/save/model.ckpt")

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
