import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell


######################model##########################
def weights(name, hidden_size, i):
    image_stdv = np.sqrt(1. / (2048))
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'in_image':
        w = tf.get_variable(name='w/in_image_'+ str(i),
                            shape=[2048, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'out_image':
        w = tf.get_variable(name='w/out_image_' + str(i),
                            shape=[hidden_size, 2048],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
    if name == 'image_hidden_state_out':
        w = tf.get_variable(name='w/image_hidden_state_out' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_in':
        w = tf.get_variable(name='w/image_hidden_state_in_' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    return w


def biases(name, hidden_size, i):
    image_stdv = np.sqrt(1. / (2048))
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'image_hidden_state_out':
        b = tf.get_variable(name='b/image_hidden_state_out' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'image_hidden_state_in':
        b = tf.get_variable(name='b/image_hidden_state_in' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
    if name == 'out_image':
        b = tf.get_variable(name='b/out_image_' + str(i), shape=[2048],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))

    return b


def message_pass(label, x, hidden_size, batch_size, num_category, graph):

    w_hidden_state = weights(label + '_hidden_state_out', hidden_size, 0)
    x_all = tf.reshape(tf.matmul(
        tf.reshape(x[:,0,:], [batch_size, hidden_size]),
        w_hidden_state),
                       [batch_size, hidden_size])
    for i in range(1, num_category):
        w_hidden_state = weights(label + '_hidden_state_out', hidden_size, i)
        x_all_ = tf.reshape(tf.matmul(
            tf.reshape(x[:, i, :], [batch_size, hidden_size]),
            w_hidden_state),
                           [batch_size, hidden_size])
        x_all = tf.concat([x_all, x_all_], 1)
    x_all = tf.reshape(x_all, [batch_size, num_category, hidden_size])
    x_all = tf.transpose(x_all, (0, 2, 1))  # [batch_size, hidden_size, num_category]

    x_ = x_all[0]
    graph_ = graph[0]
    x = tf.matmul(x_, graph_)
    for i in range(1, batch_size):
        x_ = x_all[i]
        graph_ = graph[i]
        x_ = tf.matmul(x_, graph_)
        x = tf.concat([x, x_], 0)
    x = tf.reshape(x, [batch_size, hidden_size, num_category])
    x = tf.transpose(x, (0, 2, 1))

    x_ = tf.reshape(tf.matmul(x[:, 0, :], weights(label + '_hidden_state_in', hidden_size, 0)),
                    [batch_size, hidden_size])
    for j in range(1, num_category):
        _x = tf.reshape(tf.matmul(x[:, j, :], weights(label + '_hidden_state_in', hidden_size, j)),
                        [batch_size, hidden_size])
        x_ = tf.concat([x_, _x], 1)
    x = tf.reshape(x_, [batch_size, num_category, hidden_size])

    return x


def GNN(label, data, batch_size, hidden_size, n_steps, num_category, graph):

    gru_cell = GRUCell(hidden_size)
    w_in = weights('in_' + label, hidden_size, 0)
    h0 = tf.reshape(tf.matmul(data[:,0,:], w_in), [batch_size, hidden_size])  #initialize h0 [batchsize, hidden_state]
    for i in range(1, num_category):
        w_in = weights('in_' + label, hidden_size, i)
        h0 = tf.concat([h0, tf.reshape(
                tf.matmul(data[:,i,:], w_in), [batch_size, hidden_size])
                          ], 1)
    h0 = tf.reshape(h0, [batch_size, num_category, hidden_size])  # h0: [batchsize, num_category, hidden_state]
    ini = h0
    h0 = tf.nn.tanh(h0)

    state = h0
    sum_graph = tf.reduce_sum(graph, reduction_indices=1)
    enable_node = tf.cast(tf.cast(sum_graph, dtype=bool), dtype=tf.float32)

    with tf.variable_scope("gnn"):
        for step in range(n_steps):
            if step > 0: tf.get_variable_scope().reuse_variables()
            x = message_pass(label, state, hidden_size, batch_size, num_category, graph)
            (x_new, state_new) = gru_cell(x[0], state[0])
            state_new = tf.transpose(state_new, (1, 0))
            state_new = tf.multiply(state_new, enable_node[0])
            state_new = tf.transpose(state_new, (1, 0))
            for i in range(1, batch_size):
                (x_, state_) = gru_cell(x[i], state[i])  # #input of GRUCell must be 2 rank, not 3 rank
                state_ = tf.transpose(state_, (1, 0))
                state_ = tf.multiply(state_, enable_node[i])
                state_ = tf.transpose(state_, (1, 0))
                state_new = tf.concat([state_new, state_], 0)
            state = tf.reshape(state_new, [batch_size, num_category, hidden_size])  # #restore: 2 rank to 3 rank
    return state, ini
