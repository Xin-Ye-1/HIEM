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
                 networks,
                 global_episodes,
                 global_frames):
        self.name = name
        self.envs = envs
        self.scenes = scenes
        self.targets = targets
        self.min_steps = min_steps
        self.starting_points = starting_points
        self.target_points = target_points
        self.network_main, self.network_target = networks

        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_multiple_target_graphs(from_scopes=['global/main'],
                                                              to_scopes=['local_%d' % self.name])
        self.update_target_ops = update_multiple_target_graphs(from_scopes=['global/main'],
                                                               to_scopes=['global/target'])

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
                    # sess.run(tf.global_variables_initializer())
                    # ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    # variable_to_restore = slim.get_variables_to_restore(exclude=['global_episodes', 'global_frames'])
                    # temp_saver = tf.train.Saver(variable_to_restore)
                    # temp_saver.restore(sess, ckpt.model_checkpoint_path)
                    # with open(FLAGS.pretrained_model_path + '/graphcheckpoint.json', 'r') as f:
                    #     graph_path = json.load(f)['graph_checkpoint_path']
                    # graph_params = sio.loadmat(graph_path)['graph']
                    # self.graph.set_params(graph_params)
                    sess.run(tf.global_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    variable_to_restore = slim.get_variables_to_restore(include=['lowlevel/global/in'])
                    # variable_to_restore = [val for val in variable_to_restore if 'termination' not in val.name]
                    var_list = {}
                    for var in variable_to_restore:
                        var_list[var.name.replace('lowlevel/global/in/', 'global/').split(':')[0]] = var
                    temp_saver = tf.train.Saver(var_list)
                    temp_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


    def _get_learning_rate(self,
                           lr):
        if self.episode_count < 500000:
            e = 0
        elif self.episode_count < 500000:
            e = 1
        elif self.episode_count < 500000:
            e = 2
        else:
            e = 3
        return lr / (10 ** e)



    def _train_network(self,
                       sess):
        # replay_buffer:
        # [0:vision_feature, 1:depth_feature, 2:target_input, 3:option, 4:action, 5:reward, 6:done,
        #  7:next_vision_feature, 8:next_depth_feature]
        with sess.as_default():
            batch = self.replay_buffer.get_buffer()
            # batch = self.replay_buffer.sample(FLAGS.batch_size)
            batch_size = batch.shape[0]
            next_option_q_prime, next_termination = sess.run([self.network_target.option_qvalues,
                                                              self.network_target.terminations],
                                                             feed_dict={self.network_target.visions: np.stack(batch[:, 7]),
                                                                        self.network_target.depths: np.stack(batch[:, 8]),
                                                                        self.network_target.targets: np.stack(batch[:, 2])})

            next_termination = next_termination[range(batch_size), batch[:, 3].astype('int32')]
            next_option_q = sess.run(self.network_main.option_qvalues,
                                     feed_dict={self.network_main.visions: np.stack(batch[:, 7]),
                                                self.network_main.depths: np.stack(batch[:, 8]),
                                                self.network_main.targets: np.stack(batch[:, 2])})

            target_option_qvalues = batch[:, 5] + (1 - batch[:, 6]) * FLAGS.gamma * \
                                    (next_termination * next_option_q_prime[range(batch_size), np.argmax(next_option_q, axis=-1)] +
                                     (1 - next_termination) * next_option_q_prime[range(batch_size), batch[:, 3].astype('int32')])

            highlevel_lr = self._get_learning_rate(FLAGS.highlevel_lr)

            option_loss, action_loss, entropy_loss, term_loss, _, _, _ = sess.run([self.network_main.option_qvalue_loss,
                                                                                   self.network_main.action_policy_loss,
                                                                                   self.network_main.entropy_loss,
                                                                                   self.network_main.termination_loss,
                                                                                   self.network_main.option_update,
                                                                                   self.network_main.action_update,
                                                                                   self.network_main.term_update],
                                                                                  feed_dict={
                                                                                      self.network_main.visions: np.stack( batch[:, 0]),
                                                                                      self.network_main.depths: np.stack(batch[:, 1]),
                                                                                      self.network_main.targets: np.stack(batch[:, 2]),
                                                                                      self.network_main.chosen_options: batch[:, 3],
                                                                                      self.network_main.target_option_qvalues: target_option_qvalues,
                                                                                      self.network_main.chosen_actions: batch[:, 4],
                                                                                      self.network_main.lr: highlevel_lr,
                                                                                      self.network_main.termination_reg: FLAGS.termination_reg})

            return option_loss, action_loss, entropy_loss, term_loss





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
                state = env.start((10,10,0))
            else:
                scope = max(int(num_starting_points * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training else num_starting_points
                state = env.start(starting_points[np.random.choice(scope)])

        min_step = min_steps[str(state)][target]


        done = False

        states_buffer = []

        disc_cumu_rewards = 0
        step_extrinsic_cumu_rewards = 0
        disc_extrinsic_cumu_rewards = 0

        episode_steps = 0
        action_steps = 0
        highlevel_steps = 0
        lowlevel_steps = 0
        avg_lowlevel_steps = 0.0
        avg_lowlevel_sr = 0.0

        gamma = 1
        highlevel_gamma = 1

        subtargets_buffer = []
        actions_buffer = []

        option_losses = []
        action_losses = []
        entropy_losses = []
        termination_losses = []


        termination = False
        action = -1
        option = -1

        (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
        if testing:
            epsilon = ep_end
            alpha = 0
        else:
            ratio = max((anneal_steps - max(self.episode_count - FLAGS.replay_start_size, 0)) / float(anneal_steps), 0)
            epsilon = (ep_start - ep_end) * ratio + ep_end
            alpha = max((10000 - max(self.episode_count - 21000, 0)) / float(10000),0)

        vision_feature, depth_feature = env.get_state_feature()
        vision_feature = [vision_feature for _ in range(FLAGS.history_steps)]
        depth_feature = [depth_feature for _ in range(FLAGS.history_steps)]

        for _ in range(FLAGS.max_episode_steps):
            states_buffer.append(env.position)

            option_qvalues, action_policy, term_prob = sess.run([self.network_main.option_qvalues,
                                                                 self.network_main.action_policy,
                                                                 self.network_main.terminations],
                                                                feed_dict={self.network_main.visions: [np.vstack(vision_feature)],
                                                                           self.network_main.depths: [np.vstack(depth_feature)],
                                                                           self.network_main.targets: [target_input]})
            termination = (np.random.rand() < term_prob[0][option]) #or (self.episode_count < 1000)


            if highlevel_steps == 0 or termination:

                if highlevel_steps != 0:
                    avg_lowlevel_steps += lowlevel_steps
                    disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                    highlevel_gamma *= FLAGS.gamma

                if np.random.rand() < epsilon:
                    option = np.random.choice(FLAGS.num_options)
                else:
                    option = np.argmax(option_qvalues[0])

                highlevel_steps += 1

                lowlevel_steps = 0
                step_extrinsic_cumu_rewards = 0

            action = np.random.choice(FLAGS.a_size, p=action_policy[0][option])

            for _ in range(FLAGS.skip_frames):
                new_state = env.action_step(action)
                action_steps += 1
                done = new_state in target_points[target]
                if done:
                    break

            # print '~~~~~~~~~~~~~~~~~~~~'
            # print (term_prob[0][option], termination)
            # print (option_qvalues[0], option)
            # print (action_policy[0][option], action)
            # print '~~~~~~~~~~~~~~~~~~~~'


            extrinsic_reward = 1 if done else 0

            disc_cumu_rewards += gamma * extrinsic_reward
            gamma *= FLAGS.gamma

            step_extrinsic_cumu_rewards += extrinsic_reward
            subtargets_buffer.append(option)
            actions_buffer.append(action)

            new_vision_feature, new_depth_feature = env.get_state_feature()
            new_vision_feature = vision_feature[1:] + [new_vision_feature]
            new_depth_feature = depth_feature[1:] + [new_depth_feature]

            if not testing:
                self.replay_buffer.add(np.reshape(np.array([np.vstack(vision_feature),
                                                            np.vstack(depth_feature),
                                                            target_input,
                                                            option,
                                                            action,
                                                            extrinsic_reward,
                                                            done,
                                                            np.vstack(new_vision_feature),
                                                            np.vstack(new_depth_feature)]), [1, -1]))

                if len(self.replay_buffer.buffer) > 0 and \
                        (done or (episode_steps != 0 and episode_steps % FLAGS.lowlevel_update_freq == 0)
                         or episode_steps == FLAGS.max_episode_steps - 1):
                    option_loss, action_loss, entropy_loss, term_loss = self._train_network(sess)
                    option_losses.append(option_loss)
                    action_losses.append(action_loss)
                    entropy_losses.append(entropy_loss)
                    termination_losses.append(term_loss)
                    self.replay_buffer.clear_buffer()
                    sess.run(self.update_local_ops)


                if self.frame_count % FLAGS.target_update_freq == 0:
                    sess.run(self.update_target_ops)

            episode_steps += 1
            lowlevel_steps += 1
            self.frame_count += 1
            if self.name == 0:
                sess.run(self.frame_increment)

            vision_feature = new_vision_feature
            depth_feature = new_depth_feature

            if done:
                disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                break

        if testing:
            left_step = min_steps[str(state)][target]

            return disc_cumu_rewards, episode_steps, min_step, done, left_step, states_buffer, subtargets_buffer, actions_buffer

        ol = np.mean(option_losses) if len(option_losses) != 0 else 0
        al = np.mean(action_losses) if len(action_losses) != 0 else 0
        el = np.mean(entropy_losses) if len(entropy_losses) != 0 else 0
        tl = np.mean(termination_losses) if len(termination_losses) != 0 else 0

        avg_lowlevel_steps /= highlevel_steps
        avg_lowlevel_sr /= highlevel_steps

        return disc_cumu_rewards, episode_steps, min_step, done, ol, al, el, tl, \
               disc_extrinsic_cumu_rewards, highlevel_steps, avg_lowlevel_steps, avg_lowlevel_sr

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
            steps = np.zeros(num_record)
            all_highlevel_steps = np.zeros(num_record)
            all_lowlevel_steps = np.zeros(num_record)
            lowlevel_sr = np.zeros(num_record)
            min_steps = np.zeros(num_record)
            is_success = np.zeros(num_record)

            option_losses = np.zeros(num_record)
            action_losses = np.zeros(num_record)
            entropy_losses = np.zeros(num_record)
            term_losses = np.zeros(num_record)

            if self.name == 0:
                self.summary_writer = tf.summary.FileWriter(
                    os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

            while self.episode_count < FLAGS.max_episodes:
                # sess.run(self.update_local_ops)

                sid = np.random.choice(len(self.scenes))
                scene = self.scenes[sid]
                env = self.envs[sid]
                tid = np.random.choice(len(self.targets[sid]))
                target = self.targets[sid][tid]
                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid]
                scene_min_steps = self.min_steps[sid]

                disc_cumu_rewards, action_steps, min_step, done, ol, al, el, tl,\
                disc_extrinsic_cumu_rewards, highlevel_steps, avg_lowlevel_steps, avg_lowlevel_sr \
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

                    steps[self.episode_count%num_record] = action_steps
                    all_highlevel_steps[self.episode_count%num_record] = highlevel_steps
                    all_lowlevel_steps[self.episode_count%num_record] = avg_lowlevel_steps
                    lowlevel_sr[self.episode_count%num_record] = avg_lowlevel_sr
                    min_steps[self.episode_count%num_record] = min_step
                    is_success[self.episode_count%num_record] = done

                    option_losses[self.episode_count%num_record] = ol
                    action_losses[self.episode_count%num_record] = al
                    term_losses[self.episode_count%num_record] = tl
                    entropy_losses[self.episode_count%num_record] = el

                    success_steps = steps[is_success == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

                    summary = tf.Summary()
                    summary.value.add(tag='Training/discounted cumulative rewards',
                                      simple_value=np.mean(rewards))
                    summary.value.add(tag='Training/highlevel rewards',
                                      simple_value=np.mean(highlevel_rewards))
                    summary.value.add(tag='Training/steps', simple_value=mean_success_steps)
                    summary.value.add(tag='Training/highlevel steps', simple_value=np.mean(all_highlevel_steps))
                    summary.value.add(tag='Training/lowlevel steps', simple_value=np.mean(all_lowlevel_steps))
                    summary.value.add(tag='Training/success rate', simple_value=np.mean(is_success))
                    summary.value.add(tag='Training/lowlevel success rate', simple_value=np.mean(lowlevel_sr))
                    summary.value.add(tag='Training/spl', simple_value=self._get_spl(success_records=is_success,
                                                                                     min_steps=min_steps,
                                                                                     steps=steps))
                    summary.value.add(tag='Loss/qvalue_loss', simple_value=np.mean(option_losses))
                    summary.value.add(tag='Loss/policy_loss', simple_value=np.mean(action_losses))
                    summary.value.add(tag='Loss/entropy_loss', simple_value=np.mean(entropy_losses))
                    summary.value.add(tag='Loss/termination_loss', simple_value=np.mean(term_losses))

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
                    _, _, _ = self._run_training_episode(sess=sess,
                                                         env=env,
                                                         scene=scene,
                                                         target=target,
                                                         min_steps=scene_min_steps,
                                                         starting_points=starting_points,
                                                         target_points=target_points,
                                                         testing=True,
                                                         start_state=start_state)
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























