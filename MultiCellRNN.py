import tensorflow as tf
from tensorflow.contrib import rnn
import Reader
import numpy as np

# General variables
itr = 10000   # 198 full read
reset_point = 198
start = 0
batch_size_counter = 251
batch_size = 251
sent_max_len = 40
word_max_len = 25
char_codes = 91
char_embed_size = 50
word_embed_size = 500
sent_embed_size = 300
no_of_classes = 45
learning_rate = 0.01
char_layer_multiRnn = 2
word_layer_multiRnn = 3
neurons_hidden_layer_1 = 400

# Placeholder
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

        char_stacked_rnn = []
        for l in range(char_layer_multiRnn):
            char_stacked_rnn.append(rnn.BasicLSTMCell(word_embed_size, forget_bias=1))

        # char_lstm_cell = rnn.BasicLSTMCell(word_embed_size, forget_bias=1)
        char_multi_cell = rnn.MultiRNNCell(char_stacked_rnn)
        output_words, _ = rnn.static_rnn(cell=char_multi_cell, inputs=char_train, dtype=tf.float32)
        # print(output_words[-1].get_shape())

with tf.name_scope("WordLayer"):
    with tf.variable_scope("WordLayer"):
        batch_word_lookup = tf.nn.embedding_lookup(output_words[-1], word_id)
        word_train = tf.unstack(batch_word_lookup, axis=1)

        word_stacked_rnn = []
        for _ in range(word_layer_multiRnn):
            word_stacked_rnn.append(rnn.BasicLSTMCell(sent_embed_size, forget_bias=1))

        # word_lstm_cell = rnn.BasicLSTMCell(sent_embed_size, forget_bias=1)
        word_multi_cell = rnn.MultiRNNCell(word_stacked_rnn)
        output_sent, _ = rnn.static_rnn(cell=word_multi_cell, inputs=word_train, dtype=tf.float32)
        # print(output_sent[-1].get_shape())
        output_sent = tf.concat(output_sent, axis=0)
        # output_sent = tf.reshape(output_sent, [-1, sent_embed_size])
        # print(output_sent)                                                                # shape=(12550, 600)

with tf.name_scope("ForwardLayer"):
    h_layer_weights_1 = tf.Variable(tf.random_normal([sent_embed_size, neurons_hidden_layer_1]))
    h_layer_bias_1 = tf.Variable(tf.zeros([neurons_hidden_layer_1]))
    predicted_output_1 = tf.matmul(output_sent, h_layer_weights_1) + h_layer_bias_1

    h_layer_weights_2 = tf.Variable(tf.random_normal([neurons_hidden_layer_1, no_of_classes]))
    h_layer_bias_2 = tf.Variable(tf.zeros([no_of_classes]))
    predicted_output_2 = tf.matmul(predicted_output_1, h_layer_weights_2) + h_layer_bias_2
    # print(predicted_output_2)                                                               # shape=(12550, 45)

with tf.name_scope("CostFunction"):
    # y_reshape = tf.reshape(y, [-1, 1])                                                      # (1200,1)*
    y_reshape = tf.reshape(y, [-1])                                                       # Tensor("CostFunction/Reshape:0", shape=(1200,), dtype=int32)
    # print(y_reshape)
    weights = tf.cast(tf.where(y_reshape > 0, tf.ones_like(y_reshape), tf.zeros_like(y_reshape)), tf.float32)
    y_reshape = tf.one_hot(y_reshape, depth=no_of_classes)
    # print(y_reshape)                                                                       # Tensor("CostFunction/one_hot:0", shape=(1200, 45), dtype=float32) # shape=(12550, 1, 45)*
    # y_reshape = tf.unstack(y_reshape, axis=1)
    # print(y_reshape)                                                                     # shape=(12550, 45)*
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_output_2, labels=y_reshape) * weights)

with tf.name_scope("Optimizer"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("Accuracy"):
    # print(predicted_output_2.get_shape())
    # print(y_reshape)
    correct_prediction = tf.equal(tf.argmax(predicted_output_2, 1), tf.argmax(y_reshape, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Graph executions
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, itr):
        if i%reset_point != 0:
            char_id_batch, word_id_batch, pos_id_batch = Reader.retrieve_batch_sent(start, batch_size_counter, sent_max_len, word_max_len)
            feed_dict = {char_id: char_id_batch, word_id: word_id_batch, y: pos_id_batch}
            # char_id_batch = np.ndarray(char_id_batch)
            # print("\nprinting char ids")
            # print(char_id_batch)
            # print((np.shape(char_id_batch)))
            # print("\nprinting word ids")
            # print(word_id_batch)
            # print((np.shape(word_id_batch)))
            # print("\nprinting pos ids")
            # print(pos_id_batch)
            # print((np.shape(pos_id_batch)))
            start = batch_size_counter
            batch_size_counter = batch_size + batch_size_counter
            sess.run(train_op, feed_dict=feed_dict)
            # print(i)
            if (i % 10) == 0:
                _, cost, acc_result = sess.run((train_op, loss, accuracy), feed_dict=feed_dict)
                print('Cost and accuracy for iteration %d is %0.3f and %0.3f:' % (i, cost, acc_result))
        else:
            # print("hello")
            start = 0
            batch_size_counter = 251
            batch_size = 251
