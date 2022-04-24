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
    # delete the current graph
    tf.reset_default_graph()

    # import the graph from the file
    imported_graph = tf.train.import_meta_graph('multi_modal_1/cm_ggnn.ckpt.meta')

    with tf.Session() as sess:
        imported_graph.restore(sess, "multi_modal_1/cm_ggnn.ckpt")
        print("Restored model from ckpt")

        ############test############
        test_size_fitb = load_test_size()
        batches = int((test_size_fitb * 4) / batch_size)
        right = 0.
        for ii in range(batches):
            test_fitb = load_fitb_data(ii, batch_size, test_outfit_list)
            answer = sess.run([score_pos], feed_dict={image_pos: test_fitb[0],
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
            answer = sess.run([score_pos], feed_dict={image_pos: test_auc[0],
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
