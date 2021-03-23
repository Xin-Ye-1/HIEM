#! /usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim

seed = 0


class Lowlevel_Network():
    def __init__(self,
                 window_size,
                 num_labels,
                 action_size,
                 history_steps,
                 scope='global'
                 ):
        with tf.variable_scope(scope):
            self.visions = tf.placeholder(shape=[None, history_steps * window_size * window_size, num_labels],
                                          dtype=tf.float32)
            self.depths = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)
            self.targets = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

            targets_expanded = tf.tile(tf.expand_dims(self.targets, 1),
                                       [1, history_steps * window_size * window_size, 1])
            masked_visions = tf.reduce_sum(self.visions * targets_expanded, axis=-1)
            masked_visions = slim.flatten(masked_visions)

            depths = slim.flatten(self.depths)

            hidden_visions = slim.fully_connected(inputs=masked_visions,
                                                  num_outputs=256,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                                  biases_initializer=tf.zeros_initializer(),
                                                  scope='vision_hidden')

            hidden_depths = slim.fully_connected(inputs=depths,
                                                 num_outputs=256,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 scope='depth_hidden')

            vision_depth_feature = tf.concat([hidden_visions, hidden_depths], 1)

            embed_feature = slim.fully_connected(inputs=vision_depth_feature,
                                                 num_outputs=256,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 scope='embed')
            # value estimation

            hidden_value = slim.fully_connected(inputs=embed_feature,
                                                num_outputs=20,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='value_hidden')

            self.qvalues = slim.fully_connected(inputs=hidden_value,
                                                num_outputs=action_size,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='qvalue')

            # Lowlevel training
            if not scope.startswith('global'):
                self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.target_qvalues = tf.placeholder(shape=[None], dtype=tf.float32)
                self.lowlevel_lr = tf.placeholder(dtype=tf.float32)

                actions_onehot = tf.one_hot(self.chosen_actions, action_size, dtype=tf.float32)
                qvalues_for_chosen_actions = tf.reduce_sum(self.qvalues * actions_onehot, axis=-1)
                self.qvalue_loss = 0.5 * tf.reduce_mean(tf.square(self.target_qvalues - qvalues_for_chosen_actions))

                local_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                gradients = tf.gradients(self.qvalue_loss, local_lowlevel_params)
                norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                lowlevel_trainer = tf.train.RMSPropOptimizer(learning_rate=self.lowlevel_lr)
                global_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/main')
                self.lowlevel_update = lowlevel_trainer.apply_gradients(zip(norm_gradients, global_lowlevel_params))





















