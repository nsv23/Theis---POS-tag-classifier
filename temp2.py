import tensorflow as tf
from tensorflow.contrib import rnn
import temp1
import numpy as np

# General variables
itr = 1000   # 198 full read
reset_point = 198
start = 0
batch_size_counter = 5
batch_size = 5
sent_max_len = 10
word_max_len = 15
char_codes = 91
char_embed_size = 50
word_embed_size = 300
sent_embed_size = 600
no_of_classes = 45
learning_rate = 0.001

# Placeholder (They work)
char_id = tf.placeholder(dtype=tf.int32, shape=[None, word_max_len])
word_id = tf.placeholder(dtype=tf.int32, shape=[None, sent_max_len])
y = tf.placeholder(dtype=tf.int32, shape=[None, sent_max_len])

# Operations
with tf.name_scope("CharacterLayer"):
    with tf.variable_scope("CharacterLayer"):
        char_embeddings = tf.Variable(tf.truncated_normal(shape=[char_codes, char_embed_size]))
        char_lookup = tf.nn.embedding_lookup(char_embeddings, char_id)
        # print(char_lookup.get_shape())
        char_train = tf.unstack(value=char_lookup, axis=1)
        # print(char_train)
        char_lstm_cell = rnn.BasicLSTMCell(word_embed_size, forget_bias=1)
        output_words, _ = rnn.static_rnn(cell=char_lstm_cell, inputs=char_train, dtype=tf.float32)
        # print(output_words[-1].get_shape())

with tf.name_scope("WordLayer"):
    with tf.variable_scope("WordLayer"):
        batch_word_lookup = tf.nn.embedding_lookup(output_words[-1], word_id)
        word_train = tf.unstack(batch_word_lookup, axis=1)
        word_lstm_cell = rnn.BasicLSTMCell(sent_embed_size, forget_bias=1)
        output_sent, _ = rnn.static_rnn(word_lstm_cell, word_train, dtype=tf.float32)
        # print(output_sent[-1].get_shape())
        output_sent = tf.concat(output_sent, axis=0)
        # output_sent = tf.reshape(output_sent, [-1, sent_embed_size])
        # print(output_sent)                                                                # shape=(12550, 600)

with tf.name_scope("ForwardLayer"):
    h_layer_weights = tf.Variable(tf.random_normal([sent_embed_size, no_of_classes]))
    h_layer_bias = tf.Variable(tf.zeros([no_of_classes]))
    predicted_output = tf.matmul(output_sent, h_layer_weights) + h_layer_bias
    # print(predicted_output)                                                               # shape=(12550, 45)

with tf.name_scope("CostFunction"):
    # y_reshape = tf.reshape(y, [-1, 1])                                                  # (12550,1)*
    y_reshape = tf.reshape(y, [-1])                                                       # Tensor("CostFunction/Reshape:0", shape=(12550,), dtype=int32)
    yy =  tf.reshape(y, [-1, 1])
    yyz = tf.reshape(y, [-1])
    # print(y_reshape)
    weights = tf.cast(tf.where(y_reshape > 0, tf.ones_like(y_reshape), tf.zeros_like(y_reshape)), tf.float32)
    y_reshape = tf.one_hot(y_reshape, depth=4)
    # print(y_reshape)                                                                       # Tensor("CostFunction/one_hot:0", shape=(12550, 45), dtype=float32) # shape=(12550, 1, 45)*
    # y_reshape = tf.unstack(y_reshape, axis=1)
    # print(y_reshape)                                                                     # shape=(12550, 45)*
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_output, labels=y_reshape) * weights)

with tf.name_scope("Optimizer"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# with tf.name_scope("Accuracy"):
#     # print(predicted_output.get_shape())
#     # print(y_reshape)
#     correct_prediction = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(y_reshape, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Graph executions
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    char_id_batch, word_id_batch, pos_id_batch = temp1.retrieve_batch_sent(start, batch_size_counter, sent_max_len, word_max_len)
    feed_dict = {char_id: char_id_batch, word_id: word_id_batch, y: pos_id_batch}
    c, w, o = sess.run((yy, yyz, y_reshape), feed_dict=feed_dict)
    print(c)
    print('\n')
    print(w)
    print('\n')
    print(o)

    # for i in range(1, itr):
    #     if i%reset_point != 0:
    #         char_id_batch, word_id_batch, pos_id_batch = temp1.retrieve_batch_sent(start, batch_size_counter, sent_max_len, word_max_len)
    #         feed_dict = {char_id: char_id_batch, word_id: word_id_batch, y: pos_id_batch}
    #         # char_id_batch = np.ndarray(char_id_batch)
    #         # print("\nprinting char ids")
    #         # print(char_id_batch)
    #         # print((np.shape(char_id_batch)))
    #         # print("\nprinting word ids")
    #         # print(word_id_batch)
    #         # print((np.shape(word_id_batch)))
    #         # print("\nprinting pos ids")
    #         # print(pos_id_batch)
    #         # print((np.shape(pos_id_batch)))
    #         start = batch_size_counter
    #         batch_size_counter = batch_size + batch_size_counter
    #         _, ls = sess.run((train_op, loss), feed_dict=feed_dict)
    #         print('Loss at iteration %d is: %.3f' % (i, ls))
    #         # if (i % 10) == 0:
    #         #     acc_result = sess.run(accuracy, feed_dict=feed_dict)
    #         #     print('iteration: ', i, ' ', "Accuracy: ", i, ' - ', acc_result)
    #     else:
    #         # print("hello")
    #         start = 0
    #         batch_size_counter = 251
    #         batch_size = 251
