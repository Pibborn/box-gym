# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.

import re
import pandas as pd
import numpy as np

from scipy import optimize
from numba import jit, float32

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Lasso

import gym
import gym.spaces as spaces

from utils.grammar_parser import ProbabilisticGrammar
from utils.constraints import Constraints
from utils.process_data import to_categorical


@jit(nopython=True)  # , float32(float32, float32)
def base_metric(y, yhat):
    return 1 / (1 + ((y - yhat) ** 2).mean())


class BatchSymbolicRegressionEnv(gym.Env):
    def __init__(self, grammar_file_path,
                 start_symbol,
                 train_data_path,
                 target,
                 test_data_path=None,
                 eval_params={},
                 metric=base_metric,
                 max_horizon=30,
                 min_horizon=4,
                 hidden_size=8,
                 batch_size=1,
                 normalize=False,
                 normalization_type="",
                 apply_constraints=False,
                 observe_brotherhood=False,
                 observe_parent=False,
                 observe_previous_actions=True,
                 observe_hidden_state=True,
                 observe_symbol=True,
                 observe_mask=True,
                 observe_depth=True,
                 outlier_heuristic=False,
                 constant_optimizer=True
                 ):

        # MDP related parameters
        self.max_horizon = max_horizon  # Maximal complexity of the solution
        self.min_horizon = min_horizon
        self.eval_params = eval_params  # dictionnary containing the parameters for expression evaluation
        self.batch_size = batch_size

        self.apply_constraints = apply_constraints
        self.observe_brotherhood = observe_brotherhood
        self.observe_parent = observe_parent
        self.observe_previous_actions = observe_previous_actions
        self.observe_hidden_state = observe_hidden_state
        self.observe_symbol = observe_symbol
        self.observe_mask = observe_mask
        self.observe_depth = observe_depth
        self.outlier_heuristic = outlier_heuristic

        prepared_all_dataset = False
        target = 0
        self.dataset = {}
        while not prepared_all_dataset:
            columns_train, X_train, X_test, y_train, y_test, scaler = \
                self.prepare_data_set(normalization_type, normalize, target, test_data_path, train_data_path)
            self.dataset[target] = {}
            self.dataset[target]['X_train'] = X_train
            self.dataset[target]['X_test'] = X_test
            self.dataset[target]['y_train'] = y_train
            self.dataset[target]['y_test'] = y_test

            if target == X_train.shape[1]:
                prepared_all_dataset = True
                self.num_features = X_train.shape[1]
            else:
                target += 1

        self.metric = base_metric

        # Load grammar from file
        self.start_symbol = start_symbol
        self.grammar = ProbabilisticGrammar(grammar_file_path, start_symbol=start_symbol,
                                            dataset_n_vars=len(columns_train))

        if apply_constraints:
            self.constraints = Constraints(self.grammar, self.max_horizon, self.min_horizon)

        self.constant_optimizer = constant_optimizer

        self.masks = self.grammar.symbols_to_mask
        self.symbols = list(self.grammar.productions_dict.keys())

        self.n_actions = len(self.grammar.productions_list)
        self.n_symbols = len(self.grammar.symbols) + 1
        self.hidden_size = hidden_size

        # Define gym observation space
        space_dict = self.create_space_dict()

        self.observation_space = spaces.Dict(space_dict)

        # Define gym action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=np.float32)

        self.queue = None
        self.translations = None
        self.past_actions = None
        self.past_actions_with_parent_infos = None
        self.current_parent_infos = None
        self.done = None
        self.i_step = None

        # set a caching function
        self.cache = {}
        for k, v in self.eval_params.items():
            vars()[k] = v

    def create_space_dict(self):
        space_dict = {}
        if self.observe_symbol:
            space_dict['current_symbol'] = spaces.MultiBinary(self.n_symbols)
        if self.observe_mask:
            space_dict['current_mask'] = spaces.MultiBinary(self.n_actions)
        if self.observe_depth:
            space_dict['current_depth'] = spaces.Box(low=0, high=1, shape=(1, 1))
        if self.observe_hidden_state:
            space_dict["h"] = spaces.MultiBinary(self.hidden_size)
            space_dict["c"] = spaces.MultiBinary(self.hidden_size)
        if self.observe_previous_actions:
            space_dict['past_actions'] = spaces.MultiBinary((self.max_horizon, self.n_actions + 1))
        if self.observe_brotherhood:
            space_dict['brother_action'] = spaces.MultiBinary((self.grammar.max_brother_symbols, self.n_actions + 1))
        if self.observe_parent:
            space_dict['parent_action'] = spaces.MultiBinary(self.n_actions + 1)
        space_dict['target_label'] = spaces.MultiBinary(self.num_features)
        return space_dict

    def prepare_data_set(self, normalization_type, normalize, target_label, test_data_path, train_data_path):
        # Load datasets
        X_train, X_test, scaler = None, None, None
        if test_data_path is not None:
            if ".feather" in train_data_path:
                X_train = pd.read_feather(train_data_path).iloc[:20000]
                X_test = pd.read_feather(test_data_path).iloc[:20000]
            elif ".csv" in train_data_path:
                X_train = pd.read_csv(train_data_path).iloc[:20000]
                X_test = pd.read_csv(test_data_path).iloc[:20000]
            if normalize:
                if normalization_type == 'standard_scaler':
                    scaler = StandardScaler()
                    X_train.iloc[:, :] = scaler.fit_transform(X_train)
                    X_test.iloc[:, :] = scaler.transform(X_test)
                else:
                    mini = min(X_train.min().values.min(), X_test.min().values.min())
                    maxi = max(X_train.max().values.max(), X_test.max().values.max())
                    X_train = (X_train - mini) / (maxi - mini)
                    X_test = (X_test - mini) / (maxi - mini)
            self.feature_names = X_train.columns
            target_feature = X_train.columns[target_label]
            y_train = X_train[target_feature]
            X_train.drop(columns=[target_feature], inplace=True)
            y_test = X_test[target_feature]
            X_test.drop(columns=[target_feature], inplace=True)
        else:
            if ".feather" in train_data_path:
                x = pd.read_feather(train_data_path).iloc[:20000]
            elif ".csv" in train_data_path:
                x = pd.read_csv(train_data_path).iloc[:20000]
            if normalize:
                scaler = StandardScaler()
                x.iloc[:, :] = scaler.fit_transform(x)
            y = x[target_label]
            x = x.drop(columns=[target_label], inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(x, y)
        columns_train = X_train.columns

        return columns_train, X_train, X_test, y_train, y_test, scaler

    def reset(self):

        self.done = np.zeros((self.batch_size, 1))
        self.queue = [[]] * self.batch_size
        self.current_parent_infos = [('#', -1)] * self.batch_size
        self.past_actions_with_parent_infos = [[]] * self.batch_size
        self.translations = [self.start_symbol] * self.batch_size
        self.i_step = 0

        self.past_actions = np.full((self.batch_size, self.max_horizon, self.grammar.n_discrete_actions + 1),
                                    self.grammar.action_encoding["#"])

        observation = {}
        if self.observe_symbol:
            observation['current_symbol'] = np.full((self.batch_size, 1, self.grammar.n_symbols + 1),
                                                    self.grammar.symbol_encoding[self.start_symbol])
        if self.observe_mask:
            observation['current_mask'] = np.full((self.batch_size, 1, self.grammar.n_discrete_actions),
                                                  self.grammar.symbols_to_mask[self.grammar.start_symbol])
        if self.observe_depth:
            observation["current_depth"] = np.zeros((self.batch_size, 1, 1))
        if self.observe_previous_actions:
            observation['past_actions'] = self.past_actions
        if self.observe_brotherhood:
            observation['brother_action'] = np.full((self.batch_size, self.grammar.max_brother_symbols,
                                                     self.grammar.n_discrete_actions + 1),
                                                    self.grammar.action_encoding["#"])
        if self.observe_parent:
            observation['parent_action'] = np.full((self.batch_size, 1, self.grammar.n_discrete_actions + 1),
                                                   self.grammar.action_encoding["#"])

        self.target_label = np.random.randint(low=0, high=self.num_features)
        self.target_label_one_hot = np.zeros((self.batch_size, 1, self.num_features))
        self.target_label_one_hot[:, :] = to_categorical(self.target_label, self.num_features)
        observation['target_label'] = self.target_label_one_hot
        return observation

    def step(self, action_ids):
        new_brothers, new_masks, new_parents, new_symbols = self.create_empty_array()

        for i, action_id in enumerate(action_ids):
            self.update_one_equation(action_id, i, new_brothers, new_masks, new_parents, new_symbols)

        self.i_step += 1

        observation = self.fill_observation_dic_with_new_status(new_brothers, new_masks, new_parents, new_symbols)

        return observation, self.done

    def update_one_equation(self, action_id, i, new_brothers, new_masks, new_parents, new_symbols):
        if self.done[i]:
            new_symbols[i] = self.grammar.symbol_encoding["#"]
            new_masks[i] = self.grammar.symbols_to_mask[self.start_symbol]
            new_parents[i] = self.grammar.action_encoding["#"]
            new_brothers[i] = np.array([[self.grammar.action_encoding['#'][0]] * self.grammar.max_brother_symbols])

        action = self.grammar.productions_list[action_id]
        self.translations[i] = re.sub(pattern="<.+?>",
                                      repl=action['raw'],
                                      string=self.translations[i],
                                      count=1)
        # Store action in list
        self.past_actions_with_parent_infos[i] = self.past_actions_with_parent_infos[i] + \
                                                 [(action['raw'], self.current_parent_infos[i])]
        # Append descendant symbols to the queue.
        parent_action_info = (action['raw'], self.i_step)
        self.queue[i] = [(s, parent_action_info) for s in action["descendant_symbols"]] + self.queue[i]
        if not self.queue[i]:
            self.done[i] = 1
            if self.i_step <= self.min_horizon:
                self.translations[i] = ""

            new_symbols[i] = self.grammar.symbol_encoding["#"]
            new_masks[i] = self.grammar.symbols_to_mask[self.start_symbol]
            new_parents[i] = self.grammar.action_encoding["#"]
            new_brothers[i] = np.array([[self.grammar.action_encoding['#'][0]] * self.grammar.max_brother_symbols])
            return
        new_current_symbol, new_parent_infos = self.queue[i].pop(0)
        new_symbols[i] = self.grammar.symbol_encoding[new_current_symbol][0]
        new_parents[i] = self.grammar.action_encoding[new_parent_infos[0]]
        self.current_parent_infos[i] = new_parent_infos
        # get brothers
        if self.observe_brotherhood:
            self.update_brotherhood(i, new_brothers, new_parent_infos)
        m = self.grammar.symbols_to_mask[new_current_symbol]
        if self.apply_constraints:
            m = self.apply_constraints_to_equation(i, m, new_current_symbol)
        new_masks[i] = m
        # Update current symbol previously_selected_actions with the action input
        self.past_actions[i][self.i_step] = self.grammar.action_encoding[str(action['raw'])][0]
        return

    def fill_observation_dic_with_new_status(self, new_brothers, new_masks, new_parents, new_symbols):
        observation = {}
        if self.observe_symbol:
            observation['current_symbol'] = new_symbols
        if self.observe_mask:
            observation['current_mask'] = new_masks
        if self.observe_depth:
            observation['current_depth'] = np.ones((self.batch_size, 1, 1)) / self.i_step
        if self.observe_previous_actions:
            observation['past_actions'] = self.past_actions
        if self.observe_brotherhood:
            assert len(new_brothers) == self.batch_size
            if new_brothers.shape == (
                    self.batch_size, self.grammar.max_brother_symbols, 1, self.grammar.n_discrete_actions + 1):
                new_brothers = np.squeeze(new_brothers, axis=2)
            observation['brother_action'] = new_brothers
        if self.observe_parent:
            assert len(new_parents) == self.batch_size
            observation['parent_action'] = new_parents
        observation['target_label'] = self.target_label_one_hot
        return observation

    def update_brotherhood(self, i, new_brothers, new_parent_infos):
        brothers = [self.grammar.action_encoding[potential_brother_action][0]
                    for potential_brother_action, parent_info in self.past_actions_with_parent_infos[i]
                    if parent_info == new_parent_infos]
        brothers = brothers + [self.grammar.action_encoding['#'][0] for _ in
                               range(self.grammar.max_brother_symbols - len(brothers))]
        new_brothers[i] = np.array(brothers)

    def apply_constraints_to_equation(self, i, m, new_current_symbol):
        c = self.constraints.init_constraint()
        queue_min_size = sum([min([p['distance_to_terminal']
                                   for p in self.grammar.productions_dict[q]]) for q, _ in self.queue[i]])
        c = self.constraints.make_min_max_constraint(c, new_current_symbol, self.i_step + queue_min_size)
        c = self.constraints.make_trig_constraint(c, new_current_symbol, self.translations[i])
        m = np.multiply(c, m)
        if m.sum() == 0:
            m = self.grammar.symbols_to_mask[new_current_symbol]
        return m

    def create_empty_array(self):
        new_symbols = np.zeros((self.batch_size, 1, self.n_symbols))
        new_masks = np.zeros((self.batch_size, 1, self.grammar.n_discrete_actions))
        new_parents = np.zeros((self.batch_size, 1, self.n_actions + 1))
        new_brothers = np.zeros((self.batch_size, self.grammar.max_brother_symbols, self.n_actions + 1))
        return new_brothers, new_masks, new_parents, new_symbols

    def compute_final_reward(self):
        rewards = np.empty((self.batch_size,))
        for i, translation in enumerate(self.translations):
            rewards[i] = self.evaluate_on_data(i, translation)
            rewards[i] += self.complexity_translation(translation)
        return rewards

    def evaluate_on_data(self, i_t, t):
        reward = 0
        X_train =  self.dataset[self.target_label]['X_train']
        if ("<" in t) or (t == ""):
            return reward
        else:
            def lasso_fit(columns_list):
                model = Lasso(alpha=0.01, max_iter=1000)
                model.fit(X_train[columns_list], y_train.values.reshape(-1, 1))
                return model.predict(X_train[columns_list])

            try:
                if self.constant_optimizer & ("const" in t):
                    constants = self.optimize_constants(t)
                    for i_constant, c in enumerate(constants):
                        t = t.replace('const', "{:.2f}".format(c), 1)
                        self.translations[i_t] = t
            except Exception as e:
                print(e)
            try:
                x = X_train
                if isinstance(x, pd.DataFrame):
                    x = x.values
                y_pred = eval(t)
                y_pred[np.isnan(y_pred)] = 0
                if isinstance(y_pred, np.float64) or isinstance(y_pred, int):
                    return reward
                elif np.mean(np.abs(y_pred)) < 1e-10:
                    return reward
                else:
                    reward = self.metric(self.dataset[self.target_label]['y_train'].values, y_pred)

                if np.isnan(reward):
                    reward = 0
            except Exception as e:
                reward = 0
                print(f"{t} Evaluate on data error {e}", flush=True)
        # print(t, reward)
        return reward

    def optimize_constants(self, expression):
        global nb_const
        nb_const = 0

        def my_replacer(match):
            global nb_const
            res = f"{match.group()}{nb_const}"
            nb_const += 1
            return res

        expression = re.sub('(const)', my_replacer, expression)
        string_args = ", ".join(['const' + str(i) for i in range(expression.count("const"))])

        def f(const, expression, str_args, x_train, y_train):
            tmp_f = eval(f"lambda {str_args}, x, y: np.sum(({expression} - y)**2)")
            res = tmp_f(x=x_train, y=y_train, *const)
            return res

        initial_guess = np.random.rand(expression.count('const'))
        constants = optimize.minimize(f, initial_guess, method='BFGS', args=(expression,
                                                                             string_args,
                                                                             X_train,
                                                                             y_train))

        return constants['x']

    def complexity_translation(self, translation):
        length_translation = len(translation)
        if length_translation == 0:
            return -1
        return 0.5 / len(translation)
