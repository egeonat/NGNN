import tensorflow as tf
import numpy as np
import json
from util.load_data_visual import (load_num_category, load_graph, load_train_data,
                                   load_train_size, load_fitb_data, load_test_size,
                                   load_auc_data)
from model.model_visual import GNN
from datetime import *
import os


##################load data###################
ftest = open('data/test_no_dup_new_100.json', 'r')
test_outfit_list = json.load(ftest)


def eval_cm_ggnn(batch_size, image_hidden_size, n_steps, G, num_category, i, beta):
    hidden_stdv = np.sqrt(1. / (image_hidden_size))
    if i == 0:
        with tf.variable_scope("cm_ggnn", reuse=None):
            w_conf_image = tf.get_variable(name='gnn/w/conf_image', shape=[image_hidden_size, 1],
                                           initializer=tf.random_normal_initializer(hidden_stdv))
            w_score_image = tf.get_variable(name='gnn/w/score_image', shape=[image_hidden_size, 1],
                                            initializer=tf.random_normal_initializer(hidden_stdv))
    else:
        with tf.variable_scope("cm_ggnn"):
            tf.get_variable_scope().reuse_variables()

    #################feed#######################
    image_pos = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    image_neg = tf.placeholder(tf.float32, [batch_size, num_category, 2048])
    graph_pos = tf.placeholder(tf.float32, [batch_size, num_category, num_category])
    graph_neg = tf.placeholder(tf.float32, [batch_size, num_category, num_category])

    ##################GGNN's output###################
    with tf.variable_scope("gnn_image", reuse=None):
        image_state_pos, _ = GNN('image', image_pos, batch_size, image_hidden_size, n_steps, num_category, graph_pos)  #output: [batch_size, num_category, 2048]
        tf.get_variable_scope().reuse_variables()
        image_state_neg, _ = GNN('image', image_neg, batch_size, image_hidden_size, n_steps, num_category, graph_neg)

    ##################predict positive###################
    for i in range(batch_size):
        image_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_pos[i], w_conf_image), [1, num_category]))
        image_score_pos = tf.reshape(tf.matmul(image_state_pos[i], w_score_image), [num_category, 1])
        image_score_pos = tf.maximum(0.01 * image_score_pos, image_score_pos)
        score_pos = tf.reshape(tf.matmul(image_conf_pos, image_score_pos), [1])

        image_conf_neg = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_neg[i], w_conf_image), [1, num_category]))
        image_score_neg = tf.reshape(tf.matmul(image_state_neg[i], w_score_image), [num_category, 1])
        image_score_neg = tf.maximum(0.01 * image_score_neg, image_score_neg)
        score_neg = tf.reshape(tf.matmul(image_conf_neg, image_score_neg), [1])

        if i == 0:
            s_pos = score_pos
            s_neg = score_neg
        else:
            s_pos = tf.concat([s_pos, score_pos], 0)
            s_neg = tf.concat([s_neg, score_neg], 0)

    s_pos = tf.reshape(s_pos, [batch_size, 1])
    s_neg = tf.reshape(s_neg, [batch_size, 1])

    s_pos_mean = tf.reduce_mean(s_pos)
    s_neg_mean = tf.reduce_mean(s_neg)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize the graph
        # 2017-03-02 if using tensorflow >= 0.12
        sess.run(init)
        saver.restore(sess, "multi_modal_1/cm_ggnn.ckpt")
        print("Restored model from ckpt")

        ######record######
        best_accuracy = 0.
        best_auc = 0.
        best_epoch = 0

        ############test############
        test_size_fitb = load_test_size()
        batches = int((test_size_fitb * 4) / batch_size)
        right = 0.
        for ii in range(batches):
            test_fitb = load_fitb_data(ii, batch_size, test_outfit_list)
            answer = sess.run([s_pos], feed_dict={image_pos: test_fitb[0],
                                                  graph_pos: test_fitb[1]})

            answer = np.asarray(answer[0])

            for j in range(batch_size / 4):
                a = []
                for k in range(j * 4, (j + 1) * 4):
                    a.append(answer[k][0])
                if np.argmax(a) == 0:
                    right += 1.

        accuracy = float(right / test_size_fitb)

        ####### AUC #######
        test_size_auc = load_test_size()
        batches = int((test_size_auc * 2) / batch_size)
        right = 0.
        for ii in range(batches):
            test_auc = load_auc_data(ii, batch_size, test_outfit_list)
            answer = sess.run([s_pos], feed_dict={image_pos: test_auc[0],
                                                  graph_pos: test_auc[1]})
            answer = np.asarray(answer[0])

            for j in range(batch_size / 2):
                a = []
                for k in range(j * 2, (j + 1) * 2):
                    a.append(answer[k][0])
                if np.argmax(a) == 0:
                    right += 1.

        auc = float(right / test_size_auc)
        print("Accuracy: ", "%d" % accuracy, " AUC: ", "%d" % auc)
    return best_accuracy


def look_enable_node(graph):
    if_enable = np.sum(graph, axis=1)
    index_list = []
    for index, value in enumerate(if_enable):
        if value > 0:
            index_list.append(index)
    return index_list


if __name__ == '__main__':
    num_category = load_num_category()
    G = load_graph()
    best_accuracy = 0.
    i = 0
    batch_size = 24
    image_hidden_size = 12
    n_steps = 3
    beta = 0.2
    accuracy = eval_cm_ggnn(batch_size, image_hidden_size, n_steps, G, num_category, i, beta)
