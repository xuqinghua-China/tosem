from tkinter import Variable

import torch.nn as nn
import torch

from layer import ATTAINGenerator, ATTAINDiscriminator, GANGenerator, GANDiscriminator
import warnings


from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class LSTMCNNModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.input_transform = nn.Linear(input_size, config.hidden_size)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        self.cnn = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.cnn_out_channels,
                             kernel_size=config.cnn_kernel_size)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=config.window_size - config.cnn_kernel_size + 1)
        self.output_transform = nn.Linear(config.cnn_out_channels, config.classification_size)

    def forward(self, x):
        x = x.unsqueeze(0)
        hidden_states = self.get_hidden(x)
        inputs = self.input_transform(x)  # batch_size*window_size*hidden_size
        lstm_out, _ = self.lstm(inputs, hidden_states)  # batch_size*window_size*hidden_size
        lstm_out = lstm_out.transpose(1, 2)  # batch_size*hidden_size*window_size
        cnn_out = self.cnn(lstm_out)  # batch_size*output_channel*window_size
        outputs = self.relu(cnn_out)  # batch_size*output_channel*window_size
        outputs = outputs.transpose(1, 2)
        outputs = outputs.squeeze(0)
        outputs = self.output_transform(outputs)
        outputs = torch.tanh(outputs)
        return outputs

    def get_hidden(self, x):
        h = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        c = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        return h, c


class AEModel(nn.Module):
    """Autoencoder model"""

    def __init__(self, input_size, config):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_size, out_features=config.hidden_size
        )
        self.encoder_output_layer = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size
        )
        self.decoder_output_layer = nn.Linear(
            in_features=config.hidden_size, out_features=config.classification_size
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


class CUSUMModel(nn.Module):
    def __init__(self, config):
        super(CUSUMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)

    def forward(self, x):
        hidden = self.get_hidden(x)
        outputs, hidden = self.lst(hidden, x)
        return outputs

    def get_hidden(self, x):
        h = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        c = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        return h, c


class GANModel(nn.Module):
    def __init__(self):
        super(GANModel, self).__init__()
        self.generator = GANGenerator()
        self.discriminator = GANDiscriminator()

    def forward(self, x):
        real_samples = x
        latent_x = self.get_latent_x()
        fake_samples = self.generator(latent_x)
        likelihood = self.discriminator(real_samples, fake_samples)
        return likelihood, fake_samples

    def get_latent_x(self):
        return torch.rand(self.config.laten_X_shape)


class ATTAINModel(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.generator = ATTAINGenerator(config, input_size)
        self.discriminator = ATTAINDiscriminator(config, input_size)
        self.config = config

    def forward(self, x):
        real_samples = x
        fake_samples = self.generator()
        likelihood_real = self.discriminator(real_samples)
        likelihood_real = torch.cat([likelihood_real, torch.zeros_like(likelihood_real)], dim=1)

        likelihood_fake = self.discriminator(fake_samples)
        likelihood_fake = torch.cat([torch.zeros_like(likelihood_fake), likelihood_fake], dim=1)
        likelihood = torch.cat([likelihood_real, likelihood_fake], dim=0)
        return likelihood


class LATTICEModel(nn.Module):
    def __init__(self,config,input_size):
        super().__init__()
        self.generator = ATTAINGenerator(config, input_size)
        self.discriminator = ATTAINDiscriminator(config, input_size)
        self.config = config

    def forward(self, x):
        real_samples = x
        fake_samples = self.generator()
        likelihood_real = self.discriminator(real_samples)
        likelihood_real = torch.cat([likelihood_real, torch.zeros_like(likelihood_real)], dim=1)

        likelihood_fake = self.discriminator(fake_samples)
        likelihood_fake = torch.cat([torch.zeros_like(likelihood_fake), likelihood_fake], dim=1)
        likelihood = torch.cat([likelihood_real, likelihood_fake], dim=0)
        return likelihood



#
# class Queue:
#     def __init__(self):
#         self._queue = []
#
#     @property
#     def empty(self):
#         return len(self) == 0
#
#     def __len__(self):
#         return len(self._queue)
#
#     def __next__(self):
#         if self.empty:
#             raise StopIteration("Queue is empty, no more objects to retrieve.")
#         obj = self._queue[0]
#         self._queue = self._queue[1:]
#         return obj
#
#     def next(self):
#         return self.__next__()
#
#     def add(self, obj):
#         """Add object to end of queue."""
#         self._queue.append(obj)
#
#
# class Observable(object):
#
#     def __init__(self, events):
#         # maps event names to subscribers
#         # str -> dict
#         self._events = {event: dict() for event in events}
#
#     def get_subscribers(self, event):
#         return self._events[event]
#
#     def subscribe(self, event, subscriber, callback=None):
#         if callback is None:
#             callback = getattr(subscriber, 'update')
#         self.get_subscribers(event)[subscriber] = callback
#
#     def unsubscribe(self, event, subscriber):
#         del self.get_subscribers(event)[subscriber]
#
#     def dispatch(self, event):
#         for _, callback in self.get_subscribers(event).items():
#             callback(event, self)
#
#
# class BayesianOptimization(Observable):
#     def __init__(self, f, pbounds, random_state=None, verbose=2,
#                  bounds_transformer=None):
#         self._random_state = ensure_rng(random_state)
#
#         # Data structure containing the function to be optimized, the bounds of
#         # its domain, and a record of the evaluations we have done so far
#         self._space = TargetSpace(f, pbounds, random_state)
#
#         self._queue = Queue()
#
#         # Internal GP regressor
#         self._gp = GaussianProcessRegressor(
#             kernel=Matern(nu=2.5),
#             alpha=1e-6,
#             normalize_y=True,
#             n_restarts_optimizer=5,
#             random_state=self._random_state,
#         )
#
#         self._verbose = verbose
#         self._bounds_transformer = bounds_transformer
#         if self._bounds_transformer:
#             try:
#                 self._bounds_transformer.initialize(self._space)
#             except (AttributeError, TypeError):
#                 raise TypeError('The transformer must be an instance of '
#                                 'DomainTransformer')
#
#         super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)
#
#     @property
#     def space(self):
#         return self._space
#
#     @property
#     def max(self):
#         return self._space.max()
#
#     @property
#     def res(self):
#         return self._space.res()
#
#     def register(self, params, target):
#         """Expect observation with known target"""
#         self._space.register(params, target)
#         self.dispatch(Events.OPTIMIZATION_STEP)
#
#     def probe(self, params, lazy=True):
#
#         if lazy:
#             self._queue.add(params)
#         else:
#             self._space.probe(params)
#             self.dispatch(Events.OPTIMIZATION_STEP)
#
#     def suggest(self, utility_function):
#         """Most promising point to probe next"""
#         if len(self._space) == 0:
#             return self._space.array_to_params(self._space.random_sample())
#
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             self._gp.fit(self._space.params, self._space.target)
#
#         suggestion = acq_max(
#             ac=utility_function.utility,
#             gp=self._gp,
#             y_max=self._space.target.max(),
#             bounds=self._space.bounds,
#             random_state=self._random_state
#         )
#
#         return self._space.array_to_params(suggestion)
#
#     def _prime_queue(self, init_points):
#         if self._queue.empty and self._space.empty:
#             init_points = max(init_points, 1)
#
#         for _ in range(init_points):
#             self._queue.add(self._space.random_sample())
#
#     def _prime_subscriptions(self):
#         if not any([len(subs) for subs in self._events.values()]):
#             _logger = _get_default_logger(self._verbose)
#             self.subscribe(Events.OPTIMIZATION_START, _logger)
#             self.subscribe(Events.OPTIMIZATION_STEP, _logger)
#             self.subscribe(Events.OPTIMIZATION_END, _logger)
#
#     def maximize(self,
#                  init_points=5,
#                  n_iter=25,
#                  acq='ucb',
#                  kappa=2.576,
#                  kappa_decay=1,
#                  kappa_decay_delay=0,
#                  xi=0.0,
#                  **gp_params):
#
#         self._prime_subscriptions()
#         self.dispatch(Events.OPTIMIZATION_START)
#         self._prime_queue(init_points)
#         self.set_gp_params(**gp_params)
#
#         util = UtilityFunction(kind=acq,
#                                kappa=kappa,
#                                xi=xi,
#                                kappa_decay=kappa_decay,
#                                kappa_decay_delay=kappa_decay_delay)
#         iteration = 0
#         while not self._queue.empty or iteration < n_iter:
#             try:
#                 x_probe = next(self._queue)
#             except StopIteration:
#                 util.update_params()
#                 x_probe = self.suggest(util)
#                 iteration += 1
#
#             self.probe(x_probe, lazy=False)
#
#             if self._bounds_transformer:
#                 self.set_bounds(
#                     self._bounds_transformer.transform(self._space))
#
#         self.dispatch(Events.OPTIMIZATION_END)
#
#     def set_bounds(self, new_bounds):
#
#         self._space.set_bounds(new_bounds)
#
#     def set_gp_params(self, **params):
#         self._gp.set_params(**params)