#!/usr/bin/env python
import sys

sys.path.append('..')
from utils.helper import *
from utils.offline_feature import *
import time
from replay_buffer import *
import json
import tensorflow.contrib.slim as slim
from graph import *
from scipy.special import softmax

flags = tf.app.flags
FLAGS = flags.FLAGS

id2class = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'id2class.json'), 'r'))
class2id = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'class2id.json'), 'r'))

cfg = json.load(open('../config.json', 'r'))


np.random.seed(12345)


class Worker():
    def __init__(self,
                 name,
                 envs,
                 scenes,
                 targets,
                 min_steps,
                 starting_points,
                 target_points,
                 highlevel_networks,
                 lowlevel_networks,
                 global_episodes,
                 global_frames):
        self.name = name
        self.envs = envs
        self.scenes = scenes
        self.targets = targets
        self.min_steps = min_steps
        self.starting_points = starting_points
        self.target_points = target_points
        self.highlevel_network_main, self.highlevel_network_target = highlevel_networks
        self.lowlevel_network_main, self.lowlevel_network_target, self.lowlevel_network_ex = lowlevel_networks

        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_multiple_target_graphs(from_scopes=['highlevel/global/main',
                                                                           'lowlevel/global/in/main',
                                                                           'lowlevel/global/ex'],
                                                              to_scopes=['highlevel/local_%d' % self.name,
                                                                         'lowlevel/local_%d/in' % self.name,
                                                                         'lowlevel/local_%d/ex' % self.name])
        self.update_target_ops = update_multiple_target_graphs(from_scopes=['highlevel/global/main',
                                                                            'lowlevel/global/in/main'],
                                                               to_scopes=['highlevel/global/target',
                                                                          'lowlevel/global/in/target'])

        self.saver = tf.train.Saver(max_to_keep=1)

    def _initialize_network(self,
                            sess,
                            testing=False):
        with sess.as_default():
            if FLAGS.load_model:
                print 'Loading model ...'
                if testing or FLAGS.continuing_training:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)

                else:
                    sess.run(tf.global_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    variable_to_restore = slim.get_variables_to_restore(exclude=['global_episodes', 'global_frames'])
                    variable_to_restore = [val for val in variable_to_restore if 'ex' not in val.name]
                    variable_to_restore = [val for val in variable_to_restore if 'termination' not in val.name]
                    temp_saver = tf.train.Saver(variable_to_restore)
                    temp_saver.restore(sess, ckpt.model_checkpoint_path)

                    # sess.run(tf.global_variables_initializer())
                    # ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    # variable_to_restore = slim.get_variables_to_restore(include=['lowlevel/global/in'])
                    # # variable_to_restore = [val for val in variable_to_restore if 'termination' not in val.name]
                    # var_list = {}
                    # for var in variable_to_restore:
                    #     var_list[var.name.replace('lowlevel/global/in/', 'global/').split(':')[0]] = var
                    # temp_saver = tf.train.Saver(var_list)
                    # temp_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


    def _get_learning_rate(self,
                           lr):
        if self.episode_count < 500000:
            e = 0
        elif self.episode_count < 300000:
            e = 1
        elif self.episode_count < 500000:
            e = 2
        else:
            e = 3
        return lr / (10 ** e)



    def _train_highlevel(self,
                         sess):
        # replay_buffer:
        # [0:vision, 1:depth, 2:valid_targets, 3:target_input, 4:subtarget 5:action, 6:ir, 7:er 8:sub_done 9:done,
        #  10:new_vision, 11:new_depth, 12:new_valid_targets]
        with sess.as_default():
            batch = self.replay_buffer.sample(FLAGS.batch_size)
            next_q_prime, next_terminations = sess.run([self.highlevel_network_target.q_values,
                                                        self.highlevel_network_target.terminations],
                                                       feed_dict={self.highlevel_network_target.visions:np.stack(batch[:, 10]),
                                                                  self.highlevel_network_target.depths:np.stack(batch[:, 11]),
                                                                  self.highlevel_network_target.targets:np.stack(batch[:, 3])})
            next_terminations = next_terminations[range(FLAGS.batch_size), batch[:, 4].astype('int32')]
            for i in range(FLAGS.batch_size):
                if batch[i, 4] not in batch[i, 12]:
                    next_terminations[i] = 1

            next_q = sess.run(self.highlevel_network_main.q_values,
                              feed_dict={self.highlevel_network_main.visions:np.stack(batch[:, 10]),
                                         self.highlevel_network_main.depths:np.stack(batch[:, 11]),
                                         self.highlevel_network_main.targets:np.stack(batch[:, 3])})
            next_valid_targets = batch[:, 12]
            subtarget_id = [next_valid_targets[i][np.argmax(next_q[i][next_valid_targets[i]])]
                            for i in range(FLAGS.batch_size)]
            # print next_terminations
            # print subtarget_id == batch[:, 4]
            # print [batch[i, 4] in batch[i, 12] for i in range(FLAGS.batch_size)]
            # print next_q_prime[range(FLAGS.batch_size), subtarget_id]
            # print next_q_prime[range(FLAGS.batch_size), subtarget_id] - next_q_prime[range(FLAGS.batch_size), batch[:, 4].astype('int32')]
            target_q_values = batch[:, 7] + (1 - batch[:, 9]) * FLAGS.gamma * \
                              (next_terminations*next_q_prime[range(FLAGS.batch_size), subtarget_id] +
                               (1-next_terminations)*next_q_prime[range(FLAGS.batch_size), batch[:, 4].astype('int32')])

            valid_targets = np.ones((FLAGS.batch_size, FLAGS.num_labels)) * (-np.inf)
            for i in range(FLAGS.batch_size):
                valid_targets[i][batch[i, 2]] = 0

            highlevel_lr = self._get_learning_rate(FLAGS.highlevel_lr)

            qvalue_loss, _, _ = sess.run([self.highlevel_network_main.qvalue_loss,
                                          self.highlevel_network_main.highlevel_update,
                                          self.highlevel_network_main.term_update],
                                         feed_dict={self.highlevel_network_main.visions:np.stack(batch[:, 0]),
                                                    self.highlevel_network_main.depths:np.stack(batch[:, 1]),
                                                    self.highlevel_network_main.targets:np.stack(batch[:, 3]),
                                                    self.highlevel_network_main.chosen_objects:batch[:, 4],
                                                    self.highlevel_network_main.target_q_values:target_q_values,
                                                    self.highlevel_network_main.valid_targets:valid_targets,
                                                    self.highlevel_network_main.highlevel_lr:highlevel_lr,
                                                    self.highlevel_network_main.termination_reg:FLAGS.termination_reg})

            # subtarget_input = np.zeros((FLAGS.batch_size, FLAGS.num_labels))
            # subtarget_input[range(FLAGS.batch_size), np.array(batch[:, 4].astype('int32'))] = 1
            #
            # lowlevel_lr = self._get_learning_rate(FLAGS.lowlevel_lr)
            #
            # ex_qvalue_loss, _ = sess.run([self.lowlevel_network_ex.qvalue_loss,
            #                               self.lowlevel_network_ex.lowlevel_update],
            #                              feed_dict={self.lowlevel_network_ex.visions: np.stack(batch[:, 0]),
            #                                         self.lowlevel_network_ex.depths: np.stack(batch[:, 1]),
            #                                         self.lowlevel_network_ex.targets: np.stack(batch[:, 3]),
            #                                         self.lowlevel_network_ex.subtargets: subtarget_input,
            #                                         self.lowlevel_network_ex.chosen_actions: batch[:, 5],
            #                                         self.lowlevel_network_ex.target_q_values: target_q_values,
            #                                         self.lowlevel_network_ex.lowlevel_lr: lowlevel_lr})

            return qvalue_loss#, ex_qvalue_loss

    def _train_ex_lowlevel(self,
                         sess):
        # replay_buffer:
        # [0:vision, 1:depth, 2:valid_targets, 3:target_input, 4:subtarget 5:action, 6:ir, 7:er 8:sub_done 9:done,
        #  10:new_vision, 11:new_depth, 12:new_valid_targets]
        with sess.as_default():
            batch = self.replay_buffer.sample(FLAGS.batch_size)
            # batch_size = FLAGS.batch_size
            batch_id = [i for i in range(FLAGS.batch_size) if batch[i][4] != FLAGS.num_labels-1]
            if len(batch_id) == 0:
                return 0
            batch = batch[np.array(batch_id)]
            batch_size = len(batch_id)

            next_q_prime, next_terminations = sess.run([self.highlevel_network_target.q_values,
                                                        self.highlevel_network_target.terminations],
                                                       feed_dict={self.highlevel_network_target.visions: np.stack(
                                                           batch[:, 10]),
                                                                  self.highlevel_network_target.depths: np.stack(
                                                                      batch[:, 11]),
                                                                  self.highlevel_network_target.targets: np.stack(
                                                                      batch[:, 3])})
            next_terminations = next_terminations[range(batch_size), batch[:, 4].astype('int32')]
            for i in range(batch_size):
                if batch[i, 4] not in batch[i, 12]:
                    next_terminations[i] = 1

            next_q = sess.run(self.highlevel_network_main.q_values,
                              feed_dict={self.highlevel_network_main.visions: np.stack(batch[:, 10]),
                                         self.highlevel_network_main.depths: np.stack(batch[:, 11]),
                                         self.highlevel_network_main.targets: np.stack(batch[:, 3])})
            next_valid_targets = batch[:, 12]
            subtarget_id = [next_valid_targets[i][np.argmax(next_q[i][next_valid_targets[i]])]
                            for i in range(batch_size)]
            target_eq_values = batch[:, 7] + (1 - batch[:, 9]) * FLAGS.gamma * \
                              (next_terminations * next_q_prime[range(batch_size), subtarget_id] +
                               (1 - next_terminations) * next_q_prime[
                                   range(batch_size), batch[:, 4].astype('int32')])

            subtarget_input = np.zeros((batch_size, FLAGS.num_labels))
            subtarget_input[range(batch_size), np.array(batch[:,4].astype('int32'))] = 1
            # next_q = sess.run(self.lowlevel_network_main.qvalues,
            #                   feed_dict={self.lowlevel_network_main.visions: np.stack(batch[:, 10]),
            #                              self.lowlevel_network_main.depths: np.stack(batch[:, 11]),
            #                              self.lowlevel_network_main.subtargets: subtarget_input})
            #
            # target_iq_values = batch[:, 6] + (1 - batch[:, 8]) * FLAGS.gamma * np.max(next_q, axis=-1)
            # target_iq_values = sess.run(self.lowlevel_network_main.qvalues,
            #                             feed_dict={self.lowlevel_network_main.visions: np.stack(batch[:, 0]),
            #                                        self.lowlevel_network_main.depths: np.stack(batch[:, 1]),
            #                                        self.lowlevel_network_main.subtargets: subtarget_input})
            target_q_values = target_eq_values #+ 0.1*target_iq_values[range(batch_size), batch[:, 5].astype('int32')]

            lowlevel_lr = self._get_learning_rate(FLAGS.lowlevel_lr)

            qvalue_loss, _ = sess.run([self.lowlevel_network_ex.qvalue_loss,
                                       self.lowlevel_network_ex.lowlevel_update],
                                      feed_dict={self.lowlevel_network_ex.visions:np.stack(batch[:, 0]),
                                                 self.lowlevel_network_ex.depths:np.stack(batch[:, 1]),
                                                 self.lowlevel_network_ex.targets:np.stack(batch[:, 3]),
                                                 self.lowlevel_network_ex.subtargets: subtarget_input,
                                                 self.lowlevel_network_ex.chosen_actions:batch[:, 5],
                                                 self.lowlevel_network_ex.target_q_values:target_q_values,
                                                 self.lowlevel_network_ex.lowlevel_lr:lowlevel_lr})
            return qvalue_loss

    def _train_lowlevel(self,
                        sess):
        # replay_buffer:
        # [0:vision, 1:depth, 2:valid_targets, 3:target_input, 4:subtarget 5:action, 6:ir, 7:er 8:sub_done 9:done,
        #  10:new_vision, 11:new_depth, 12:new_valid_targets]
        with sess.as_default():
            batch = self.replay_buffer.sample(FLAGS.batch_size)

            subtarget_input = np.zeros((FLAGS.batch_size, FLAGS.num_labels))
            subtarget_input[range(FLAGS.batch_size), np.array(batch[:, 4].astype('int32'))] = 1

            next_q_prime = sess.run(self.lowlevel_network_target.qvalues,
                                    feed_dict={self.lowlevel_network_target.visions: np.stack(batch[:, 10]),
                                               self.lowlevel_network_target.depths: np.stack(batch[:, 11]),
                                               self.lowlevel_network_target.subtargets: subtarget_input})
            next_q = sess.run(self.lowlevel_network_main.qvalues,
                              feed_dict={self.lowlevel_network_main.visions: np.stack(batch[:, 10]),
                                         self.lowlevel_network_main.depths: np.stack(batch[:, 11]),
                                         self.lowlevel_network_main.subtargets: subtarget_input})

            target_qvalues = batch[:, 6] + (1 - batch[:, 8]) * FLAGS.gamma * \
                             (next_q_prime[range(FLAGS.batch_size), np.argmax(next_q, axis=-1)])

            lowlevel_lr = self._get_learning_rate(FLAGS.lowlevel_lr)
            qvalue_loss, _ = sess.run([self.lowlevel_network_main.qvalue_loss,
                                       self.lowlevel_network_main.lowlevel_update],
                                      feed_dict={self.lowlevel_network_main.visions: np.stack(batch[:, 0]),
                                                 self.lowlevel_network_main.depths: np.stack(batch[:, 1]),
                                                 self.lowlevel_network_main.subtargets: subtarget_input,
                                                 self.lowlevel_network_main.chosen_actions: batch[:, 5],
                                                 self.lowlevel_network_main.target_qvalues: target_qvalues,
                                                 self.lowlevel_network_main.lowlevel_lr: lowlevel_lr})
            return qvalue_loss


    def _plan_on_graph(self,
                       valid_options,
                       planning_results):
        trajectories, rewards = planning_results
        all_trajectories = [trajectories[o] for o in valid_options]
        all_rewards = [rewards[o] for o in valid_options]
        distribution = softmax(all_rewards)
        return all_trajectories, all_rewards, distribution


    def _run_training_episode(self,
                              sess,
                              env,
                              scene,
                              target,
                              min_steps,
                              starting_points,
                              target_points,
                              testing=False,
                              start_state=None):
        remove_background = np.ones(FLAGS.num_labels)
        remove_background[-1] = 0
        target_input = np.zeros(FLAGS.num_labels)
        target_input[int(class2id[target])] = 1


        if start_state is not None:
            state = env.start(start_state)
        else:
            num_starting_points = len(starting_points)
            if testing:
                #state = env.start(starting_points[np.random.choice(num_starting_points)])
                state = env.start((-6, 1, 0))
            else:
                scope = max(int(num_starting_points * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training else num_starting_points
                state = env.start(starting_points[np.random.choice(scope)])

        min_step = min_steps[str(state)][target]


        done = False
        sub_done = False

        subtarget = target
        subtarget_id = int(class2id[target])
        subtarget_input = target_input


        states_buffer = []

        disc_cumu_rewards = 0
        step_extrinsic_cumu_rewards = 0
        disc_extrinsic_cumu_rewards = 0
        disc_intrinsic_cumu_rewards = 0.0
        lowlevel_disc_rewards = 0

        episode_steps = 0
        action_steps = 0
        highlevel_steps = 0
        lowlevel_steps = 0
        avg_lowlevel_steps = 0.0
        avg_lowlevel_sr = 0.0

        gamma = 1
        highlevel_gamma = 1
        lowlevel_gamma = 1

        subtargets_buffer = []
        actions_buffer = []

        qvalue_losses = []
        ex_losses = []
        in_losses = []


        termination = False
        action = -1


        (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
        if testing:
            epsilon = ep_end
            alpha = 0
        else:
            ratio = max((anneal_steps - max(self.episode_count - FLAGS.replay_start_size, 0)) / float(anneal_steps), 0)
            epsilon = (ep_start - ep_end) * ratio + ep_end
            alpha = max((10000 - max(self.episode_count - 100, 0)) / float(10000),0)

        vision_feature, depth_feature = env.get_state_feature()
        vision_feature = [vision_feature for _ in range(FLAGS.history_steps)]
        depth_feature = [depth_feature for _ in range(FLAGS.history_steps)]
        visible_targets = env.get_visible_objects()
        list_visible_targets = [visible_targets for _ in range(FLAGS.history_steps)]
        valid_targets = get_distinct_list(list_visible_targets, add_on=FLAGS.num_labels-1)


        for _ in range(FLAGS.max_episode_steps):
            states_buffer.append(env.position)

            qvalues, term_prob = sess.run([self.highlevel_network_main.q_values,
                                           self.highlevel_network_main.terminations],
                                          feed_dict={self.highlevel_network_main.visions: [np.vstack(vision_feature)],
                                                     self.highlevel_network_main.depths: [np.vstack(depth_feature)],
                                                     self.highlevel_network_main.targets: [target_input]})
            termination = (np.random.rand() < term_prob[0][subtarget_id] or subtarget_id not in valid_targets) #and False
            # termination = np.random.rand() < term_prob[0][subtarget_id]
            # print (term_prob[0][subtarget_id], termination)

            if highlevel_steps == 0 or sub_done or termination or \
                    lowlevel_steps == FLAGS.max_lowlevel_episode_steps:
                # print '~~~~~~~~~~~~~~~'
                # print (sub_done, termination, lowlevel_steps)

                if highlevel_steps != 0:
                    avg_lowlevel_steps += lowlevel_steps
                    avg_lowlevel_sr += sub_done
                    disc_intrinsic_cumu_rewards += lowlevel_disc_rewards
                    disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                    highlevel_gamma *= FLAGS.gamma


                # print [id2class[str(o)] if o != FLAGS.num_labels-1 else 'background' for o in valid_targets]

                if np.random.rand() < ep_end:
                    subtarget_id = np.random.choice(valid_targets)
                else:
                    # print qvalues[0][valid_targets]
                    # print (max(qvalues[0][valid_targets]), max(qvalues))
                    subtarget_id = valid_targets[np.argmax(qvalues[0][valid_targets])]
                # print (id2class[str(subtarget_id)], target)

                highlevel_steps += 1

                subtarget = id2class[str(subtarget_id)]
                subtarget_input = np.zeros(FLAGS.num_labels)
                subtarget_input[subtarget_id] = 1

                sub_done = state in target_points[subtarget]

                lowlevel_steps = 0
                lowlevel_disc_rewards = 0
                lowlevel_gamma = 1
                step_extrinsic_cumu_rewards = 0
                action = -1

            if not sub_done:
                # if np.random.rand() < ep_end or subtarget_id == FLAGS.num_labels-1:
                #     action = np.random.choice(FLAGS.a_size)
                # else:
                #     if np.random.rand() < alpha:
                #         lowlevel_qvalues_in = sess.run(self.lowlevel_network_main.qvalues,
                #                                        feed_dict={
                #                                            self.lowlevel_network_main.visions: [
                #                                                np.vstack(vision_feature)],
                #                                            self.lowlevel_network_main.depths: [
                #                                                np.vstack(depth_feature)],
                #                                            self.lowlevel_network_main.subtargets: [subtarget_input]})
                #         action = np.argmax(lowlevel_qvalues_in[0])
                #     else:
                #         lowlevel_qvalues_ex = sess.run(self.lowlevel_network_ex.qvalues,
                #                                        feed_dict={
                #                                            self.lowlevel_network_ex.visions: [
                #                                                np.vstack(vision_feature)],
                #                                            self.lowlevel_network_ex.depths: [np.vstack(depth_feature)],
                #                                            self.lowlevel_network_ex.subtargets: [subtarget_input],
                #                                            self.lowlevel_network_ex.targets: [target_input]})
                #         action = np.argmax(lowlevel_qvalues_ex[0])

                if np.random.rand() < alpha:

                    if np.random.rand() < ep_end or subtarget_id == FLAGS.num_labels-1:
                        action = np.random.choice(FLAGS.a_size)
                    else:
                        lowlevel_qvalues_in = sess.run(self.lowlevel_network_main.qvalues,
                                                       feed_dict={
                                                           self.lowlevel_network_main.visions: [np.vstack(vision_feature)],
                                                           self.lowlevel_network_main.depths: [np.vstack(depth_feature)],
                                                           self.lowlevel_network_main.subtargets: [subtarget_input]})
                        action = np.argmax(lowlevel_qvalues_in[0])
                else:

                    if np.random.rand() < epsilon or subtarget_id == FLAGS.num_labels-1:
                        action = np.random.choice(FLAGS.a_size)
                    else:
                        lowlevel_qvalues_ex = sess.run(self.lowlevel_network_ex.qvalues,
                                                       feed_dict={
                                                           self.lowlevel_network_ex.visions: [np.vstack(vision_feature)],
                                                           self.lowlevel_network_ex.depths: [np.vstack(depth_feature)],
                                                           self.lowlevel_network_ex.subtargets: [subtarget_input],
                                                           self.lowlevel_network_ex.targets: [target_input]})
                        action = np.argmax(lowlevel_qvalues_ex[0])

                for _ in range(FLAGS.skip_frames):
                    new_state = env.action_step(action)
                    action_steps += 1
                    sub_done = new_state in target_points[subtarget]
                    done = new_state in target_points[target] #and subtarget == target
                    if sub_done or done:
                        break

            intrinsic_reward = 1 if sub_done else 0
            extrinsic_reward = 1 if done else 0

            disc_cumu_rewards += gamma * extrinsic_reward
            gamma *= FLAGS.gamma

            lowlevel_disc_rewards += lowlevel_gamma * intrinsic_reward
            lowlevel_gamma *= FLAGS.gamma

            step_extrinsic_cumu_rewards += extrinsic_reward
            subtargets_buffer.append(subtarget_id)
            actions_buffer.append(action)

            new_vision_feature, new_depth_feature = env.get_state_feature()
            new_vision_feature = vision_feature[1:] + [new_vision_feature]
            new_depth_feature = depth_feature[1:] + [new_depth_feature]
            visible_targets = env.get_visible_objects()
            list_visible_targets = list_visible_targets[1:] + [visible_targets]
            new_valid_targets = get_distinct_list(list_visible_targets, add_on=FLAGS.num_labels-1)

            if not testing:
                if episode_steps < FLAGS.max_episode_steps - 1:
                    self.replay_buffer.add(np.reshape(np.array([np.vstack(vision_feature),
                                                                np.vstack(depth_feature),
                                                                valid_targets,
                                                                target_input,
                                                                subtarget_id,
                                                                action,
                                                                intrinsic_reward,
                                                                extrinsic_reward,
                                                                sub_done,
                                                                done]), [1, -1]))

                if self.episode_count > FLAGS.replay_start_size and \
                        len(self.replay_buffer.buffer) >= FLAGS.batch_size and \
                        self.frame_count % FLAGS.highlevel_update_freq == 0:
                    qvalue_loss = self._train_highlevel(sess=sess)
                    qvalue_losses.append(qvalue_loss)
                    ex_loss = self._train_ex_lowlevel(sess=sess)
                    ex_losses.append(ex_loss)

                # if self.episode_count > FLAGS.replay_start_size and \
                #         len(self.replay_buffer.buffer) >= FLAGS.batch_size and \
                #         self.frame_count % FLAGS.lowlevel_update_freq == 0:
                #     in_loss = self._train_lowlevel(sess=sess)
                #     in_losses.append(in_loss)

                if self.episode_count > FLAGS.replay_start_size and self.frame_count % FLAGS.target_update_freq == 0:
                    sess.run(self.update_target_ops)


            episode_steps += 1
            lowlevel_steps += 1
            self.frame_count += 1
            if self.name == 0:
                sess.run(self.frame_increment)

            vision_feature = new_vision_feature
            depth_feature = new_depth_feature
            valid_targets = new_valid_targets

            if done:
                states_buffer.append(env.position)
                disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                break

        if testing:
            left_step = min_steps[str(state)][target]

            return disc_cumu_rewards, episode_steps, min_step, done, left_step, states_buffer, subtargets_buffer, actions_buffer


        ql = np.mean(qvalue_losses) if len(qvalue_losses) != 0 else 0
        exl = np.mean(ex_losses) if len(ex_losses) != 0 else 0
        inl = np.mean(in_losses) if len(in_losses) != 0 else 0

        avg_lowlevel_steps /= highlevel_steps
        avg_lowlevel_sr /= highlevel_steps
        disc_intrinsic_cumu_rewards /= highlevel_steps

        return disc_cumu_rewards, episode_steps, min_step, done, ql, exl, inl, \
               disc_extrinsic_cumu_rewards, highlevel_steps, disc_intrinsic_cumu_rewards, avg_lowlevel_steps, avg_lowlevel_sr

    def _get_spl(self,
                 success_records,
                 min_steps,
                 steps):
        spl = 0
        n = 0
        for i in range(len(success_records)):
            if min_steps[i] != 0:
                spl += float(success_records[i] * min_steps[i]) / max(min_steps[i], steps[i])
                n += 1
        spl = spl / n
        return spl

    def _load_map_loc2idx(self,
                          scene):
        loc2idx = {}
        map_path = '%s/Environment/houses/%s/map.txt' % (cfg['codeDir'], str(scene))
        with open(map_path, 'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 3:
                    idx = int(nums[0])
                    loc = (int(nums[1]), int(nums[2]))
                    loc2idx[loc] = idx
        return loc2idx

    def _local2global(self,
                      loc2idx,
                      lid):
        (x, y, orien) = lid
        idx = loc2idx[(x, y)]
        gid = 4 * idx + orien
        return gid

    def _save_trajectory(self,
                         scene,
                         target,
                         states_buffer,
                         options_buffer,
                         actions_buffer):
        print 'len(states_buffer): ' + str(len(states_buffer))

        file_path = 'evaluate_%s.txt' % FLAGS.model_path.split('/')[-2]
        loc2idx = self._load_map_loc2idx(scene)

        n = len(states_buffer)
        with open(file_path, 'a') as f:
            f.write('%s\n' % scene)
            f.write('%s\n' % target)
            for i in range(n - 1):
                lid = states_buffer[i]
                gid = self._local2global(loc2idx, lid)
                oid = options_buffer[i]
                olabel = id2class[str(oid)]
                f.write('%d %s %d %d %s %d\n' % (
                i, str(lid), gid, oid, olabel, actions_buffer[i]))
            lid = states_buffer[n - 1]
            gid = self._local2global(loc2idx, lid)
            f.write('%d %s %d \n' % (n - 1, str(lid), gid))
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def test(self):
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            sess.run(self.update_local_ops)
            self.episode_count = 0
            self.frame_count = 0

            # [t, s, s', o, a, r, done]
            self.lowlevel_replay_buffer = ReplayBuffer()
            self.highlevel_replay_buffer = ReplayBuffer()

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            while self.episode_count < FLAGS.max_episodes:
                sid = np.random.choice(len(self.scenes))
                scene = self.scenes[sid]
                env = self.envs[sid]
                tid = np.random.choice(len(self.targets[sid]))
                target = self.targets[sid][tid]
                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid]
                scene_min_steps = self.min_steps[sid]



                disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                states_buffer, options_buffer, actions_buffer = self._run_training_episode(
                    sess=sess,
                    env=env,
                    scene=scene,
                    target=target,
                    min_steps=scene_min_steps,
                    starting_points=starting_points,
                    target_points=target_points,
                    testing=True)

                rewards.append(disc_cumu_rewards)
                steps.append(episode_steps)
                min_steps.append(min_step)
                is_success.append(done)
                left_steps.append(left_step)

                if done and episode_steps < 100:
                    self._save_trajectory(scene=scene,
                                          target=target,
                                          states_buffer=states_buffer,
                                          options_buffer=options_buffer,
                                          actions_buffer=actions_buffer)

                self.episode_count += 1

            success_steps = np.array(steps)[np.array(is_success) == 1]
            mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

            print "SR:%4f" % np.mean(is_success)
            print "AS:%4f" % mean_success_steps
            print "SPL:%4f" % self._get_spl(success_records=is_success,
                                            min_steps=min_steps,
                                            steps=steps)
            print "AR:%4f" % np.mean(rewards)
            print "LS:%4f" % np.mean(left_steps)

    def work(self,
             sess):
        print 'starting worker %s' % str(self.name)
        np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            self._initialize_network(sess)
            self.episode_count = sess.run(self.global_episodes)
            self.frame_count = sess.run(self.global_frames)

            # [t, s, s', o, a, r, done]
            self.replay_buffer = ReplayBuffer()

            num_record = 100

            rewards = np.zeros(num_record)
            highlevel_rewards = np.zeros(num_record)
            lowlevel_rewards = np.zeros(num_record)
            steps = np.zeros(num_record)
            all_highlevel_steps = np.zeros(num_record)
            all_lowlevel_steps = np.zeros(num_record)
            lowlevel_sr = np.zeros(num_record)
            min_steps = np.zeros(num_record)
            is_success = np.zeros(num_record)

            qvalue_losses = np.zeros(num_record)
            ex_losses = np.zeros(num_record)
            in_losses = np.zeros(num_record)

            if self.name == 0:
                self.summary_writer = tf.summary.FileWriter(
                    os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

            while self.episode_count < FLAGS.max_episodes:
                sess.run(self.update_local_ops)

                sid = np.random.choice(len(self.scenes))
                scene = self.scenes[sid]
                env = self.envs[sid]
                tid = np.random.choice(len(self.targets[sid]))
                target = self.targets[sid][tid]
                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid]
                scene_min_steps = self.min_steps[sid]


                disc_cumu_rewards, action_steps, min_step, done, ql, exl, inl,\
                disc_extrinsic_cumu_rewards, highlevel_steps, disc_intrinsic_cumu_rewards, avg_lowlevel_steps, avg_lowlevel_sr \
                    = self._run_training_episode(sess=sess,
                                                 env=env,
                                                 scene=scene,
                                                 target=target,
                                                 min_steps=scene_min_steps,
                                                 starting_points=starting_points,
                                                 target_points=target_points,
                                                 testing=False)
                if self.name == 0:
                    print 'episode:{:6}, scene:{} target:{:20} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                        self.episode_count, scene, target, round(disc_cumu_rewards, 2), action_steps, min_step, done)
                    rewards[self.episode_count%num_record] = disc_cumu_rewards
                    highlevel_rewards[self.episode_count%num_record] = disc_extrinsic_cumu_rewards
                    lowlevel_rewards[self.episode_count%num_record] = disc_intrinsic_cumu_rewards
                    steps[self.episode_count%num_record] = action_steps
                    all_highlevel_steps[self.episode_count%num_record] = highlevel_steps
                    all_lowlevel_steps[self.episode_count%num_record] = avg_lowlevel_steps
                    lowlevel_sr[self.episode_count%num_record] = avg_lowlevel_sr
                    min_steps[self.episode_count%num_record] = min_step
                    is_success[self.episode_count%num_record] = done

                    qvalue_losses[self.episode_count%num_record] = ql
                    ex_losses[self.episode_count%num_record] = exl
                    in_losses[self.episode_count % num_record] = inl

                    success_steps = steps[is_success == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

                    summary = tf.Summary()
                    summary.value.add(tag='Training/discounted cumulative rewards',
                                      simple_value=np.mean(rewards))
                    summary.value.add(tag='Training/highlevel rewards',
                                      simple_value=np.mean(highlevel_rewards))
                    summary.value.add(tag='Training/lowlevel rewards',
                                      simple_value=np.mean(lowlevel_rewards))
                    summary.value.add(tag='Training/steps', simple_value=mean_success_steps)
                    summary.value.add(tag='Training/highlevel steps', simple_value=np.mean(all_highlevel_steps))
                    summary.value.add(tag='Training/lowlevel steps', simple_value=np.mean(all_lowlevel_steps))
                    summary.value.add(tag='Training/success rate', simple_value=np.mean(is_success))
                    summary.value.add(tag='Training/lowlevel success rate', simple_value=np.mean(lowlevel_sr))
                    summary.value.add(tag='Training/spl', simple_value=self._get_spl(success_records=is_success,
                                                                                     min_steps=min_steps,
                                                                                     steps=steps))
                    summary.value.add(tag='Loss/qvalue_loss', simple_value=np.mean(qvalue_losses))
                    summary.value.add(tag='Loss/ex_loss', simple_value=np.mean(ex_losses))
                    summary.value.add(tag='Loss/in_loss', simple_value=np.mean(in_losses))

                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                    if self.episode_count % 1000 == 0 and self.episode_count != 0:
                        self.saver.save(sess, FLAGS.model_path + '/model' + str(self.episode_count) + '.cptk')


                    sess.run(self.episode_increment)

                self.episode_count += 1


    def evaluate(self,
                 read_file='../random_method/1s1t.txt'):
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            self.episode_count = 0
            self.frame_count = 0

            self.lowlevel_replay_buffer = ReplayBuffer()
            self.highlevel_replay_buffer = ReplayBuffer()

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            with open(read_file, 'r') as f:
                for line in f:
                    nums = line.split()
                    if len(nums) != 5:
                        continue
                    scene = nums[0]
                    target = nums[1]
                    start_state = (int(nums[2]), int(nums[3]), int(nums[4]))
                    # print (scene, target, start_state)

                    sid = self.scenes.index(scene)
                    tid = self.targets[sid].index(target)

                    env = self.envs[sid]

                    starting_points = self.starting_points[sid][tid]
                    target_points = self.target_points[sid]
                    scene_min_steps = self.min_steps[sid]


                    disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                    states_buffer, options_buffer, actions_buffer = self._run_training_episode(sess=sess,
                                                                                               env=env,
                                                                                               scene=scene,
                                                                                               target=target,
                                                                                               min_steps=scene_min_steps,
                                                                                               starting_points=starting_points,
                                                                                               target_points=target_points,
                                                                                               testing=True,
                                                                                               start_state=start_state)
                    # if self.name == 0:# and episode_steps < 100:
                    #     print 'episode:{:6}, scene:{} target:{:20} start position:{:15} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                    #         self.episode_count, scene, target, start_state, round(disc_cumu_rewards, 2), episode_steps, min_step,
                    #         done)

                        # if episode_steps < 100:
                        #     self._save_trajectory(scene=scene,
                        #                           target=target,
                        #                           states_buffer=states_buffer,
                        #                           options_buffer=options_buffer,
                        #                           actions_buffer=actions_buffer)
                    # print "min_step: " + str(min_step)
                    # print "episode_step: " + str(episode_steps)
                    rewards.append(disc_cumu_rewards)
                    steps.append(episode_steps)
                    min_steps.append(min_step)
                    is_success.append(done)
                    left_steps.append(left_step)

            success_steps = np.array(steps)[np.array(is_success) == 1]
            mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

            success_min_steps = np.array(min_steps)[np.array(is_success) == 1]
            mean_success_min_steps = np.mean(success_min_steps) if sum(is_success) != 0 else 0

            print "SR:%4f" % np.mean(is_success)
            print "AS:%4f / %4f" % (mean_success_steps, mean_success_min_steps)
            print "SPL:%4f" % self._get_spl(success_records=is_success,
                                            min_steps=min_steps,
                                            steps=steps)
            print "AR:%4f" % np.mean(rewards)
            # print "LS:%4f" % np.mean(left_steps)

    def _get_valid_options(self,
                           env_dir,
                           positions,
                           target_id):
        mode = 'gt' if FLAGS.use_gt else 'pred'
        semantic_dynamic = json.load(open('%s/%s_dynamic.json' % (env_dir, mode), 'r'))
        valid_options = [target_id]
        for pos in positions:
            transition = semantic_dynamic[str(pos)]
            options_str = transition.keys()
            valid_options += [int(o) for o in options_str if int(o) != FLAGS.num_labels-1]
        return set(valid_options)
        # return valid_options






















