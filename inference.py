import make_train_set
import tensorflow as tf
from itertools import chain
import character_util as c_util

model_save_path = "C:/Users/Administrator/PycharmProjects/STT_correction/save/4enc_4dec_0iter_54.910683plx_129752.555loss_model.ckpt"
graph_save_path = "C:/Users/Administrator/PycharmProjects/STT_correction/save/4enc_4dec_0iter_54.910683plx_129752.555loss_model.ckpt.meta"
inf_text = "시험이 끝난 후 친구의 답을 마쳐 보유했다"

# restore
sess = tf.Session()
saver = tf.train.import_meta_graph(graph_save_path) # 그래프 생성
saver.restore(sess, model_save_path) # 가중치 불러오기

# inference
inf_char, inf_idx = make_train_set.c_util.decompose_sentence(inf_text)

inf_encoder_x = [inf_idx]
inf_encoder_seq_len_except_pad = [len(inf_char)]
inf_decoder_seq_len_except_pad = [1]
inf_decoder_x = [make_train_set.sos_token]

for count in range(50):
    next_word = sess.run(tf.get_default_graph().get_tensor_by_name('decoder_for_logit/transpose_1:0'),
                         feed_dict={'encoder_x:0': inf_encoder_x, 'encoder_length:0': inf_encoder_seq_len_except_pad,
                                    'decoder_x:0': inf_decoder_x, 'decoder_length:0': inf_decoder_seq_len_except_pad})
    inf_decoder_x = [make_train_set.sos_token + list(*chain(next_word.tolist()))]
    inf_decoder_seq_len_except_pad = [len(inf_decoder_x[0])]
    next_word = [inf_decoder_x[0][-1]]
    if next_word == make_train_set.eos_token:
        break

make_train_set.c_util.compose_sentence(inf_decoder_x[0])
