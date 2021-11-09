import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv

from utils import EdgeMatrix


class ATTAINGenerator(nn.Module):
    def __init__(self, config, input_size, graph=None):
        super(ATTAINGenerator, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=self.config.pool_kernel_size, stride=self.config.pool_kernel_size)
        self.out = nn.Linear(self.config.hidden_size, input_size)

    def forward(self):
        latent_X = torch.rand(size=self.config.latent_X_shape)  # 1*1*128*128
        cnn_out = self.conv(latent_X)  # 1*1*128*128
        output = torch.relu(cnn_out)  # 1*1*128*128
        output = torch.dropout(output, p=self.config.dropout, train=self.training)  # 1*1*128*128
        fake_sample = self.pool(output)  # 1*1*32*64
        fake_sample = fake_sample.squeeze(0)
        fake_sample = self.out(fake_sample)  # 1*32*51
        fake_sample = fake_sample.squeeze(0)  # 32*51
        return fake_sample


class ATTAINDiscriminator(nn.Module):
    def __init__(self, config, input_size):
        super(ATTAINDiscriminator, self).__init__()
        self.config = config
        self.edge_matrix = EdgeMatrix(input_size)
        self.edge_indices = self.edge_matrix.get_all_connected_indices()
        self.gcn_conv = GCNConv(config.batch_size, config.batch_size)
        self.input2hidden = nn.Linear(input_size, config.hidden_size)
        self.out = nn.Linear(input_size, 2)

    def forward(self, data):
        data = data.transpose(0, 1)  # 51* 32
        edges = self.edge_indices
        output = self.gcn_conv(data, edges)  # 51* 32
        output = torch.relu(output)  # 51* 32
        output = output.transpose(0, 1)  # 32*51
        output = self.out(output)
        return output

    def get_hidden(self, x):
        h = torch.zeros(1, x.shape[0], self.config.hidden_size)
        c = torch.zeros(1, x.shape[0], self.config.hidden_size)
        return h, c


class GANGenerator(nn.Module):
    def __init__(self, config, input_size):
        super(GANGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        self.linear2d = nn.Linear(config.hidden_size, config.gan_generator_out_size)
        self.config = config
        self.input_size = input_size

    def forward(self, x):
        lstm_out, hidden = self.lstm(self.get_hidden(x), x)
        lstm_out = lstm_out.reshape(-1, self.config.hidden_size)
        logits_2d = self.linear2d(lstm_out)
        output_2d = torch.tanh(logits_2d)
        output_3d = output_2d.reshape(-1, x.shape[0], self.input_size)
        return output_3d

    def get_hidden(self, x):
        h = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        c = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        return h, c


class GANDiscriminator(nn.Module):
    def __init__(self, config):
        super(GANDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        self.out = nn.Linear(config.hidden_size, config.classification_size)
        self.config = config

    def forward(self, x):
        mean_over_batch = torch.cat([torch.mean(x).unsqueeze(0)] * self.config.batch_size)
        inputs = torch.cat([x, mean_over_batch], dim=2)
        lstm_outputs, lstm_hidden = self.lstm(inputs, self.get_hidden(x))
        logits = torch.einsum(lstm_outputs)
        output = self.out(logits)
        output = torch.sigmoid(output)
        return output, logits

    def get_hidden(self, x):
        h = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        c = torch.zeros(x.shape[0], 1, self.config.hidden_size)
        return h, c


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propagator(nn.Module):
    def __init__(self, state_dim, n_nodes, n_edge_types):
        super(Propagator, self).__init__()

        self.n_nodes = n_nodes
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )
        self.state_dim = state_dim

    def forward(self, state_in, state_out, state_cur, A):  # A = [A_in, A_out]
        A_in = A[:, :, :self.n_nodes * self.n_edge_types]
        A_out = A[:, :, self.n_nodes * self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)  # batch size x |V| x state dim
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)  # batch size x |V| x 3*state dim

        r = self.reset_gate(a.view(-1, self.state_dim * 3))  # batch size*|V| x state_dim
        z = self.update_gate(a.view(-1, self.state_dim * 3))
        r = r.view(-1, self.n_nodes, self.state_dim)
        z = z.view(-1, self.n_nodes, self.state_dim)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.transform(joined_input.view(-1, self.state_dim * 3))
        h_hat = h_hat.view(-1, self.n_nodes, self.state_dim)
        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    def __init__(self, state_dim, annotation_dim, n_edge_types, n_nodes, n_steps):
        super(GGNN, self).__init__()
        assert (state_dim >= annotation_dim, 'state_dim must be no less than annotation_dim')
        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_nodes = n_nodes
        self.n_steps = n_steps
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propagation Model
        self.propagator = Propagator(self.state_dim, self.n_nodes, self.n_edge_types)

        # Output Model
        self.graph_rep = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),  # self.state_dim + self.annotation_dim
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )
        self.score = nn.Linear(self.n_nodes, 1)
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            # print ("PROP STATE SIZE:", prop_state.size()) #batch size x |V| x state dim
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state.view(-1, self.state_dim)))
                out_states.append(self.out_fcs[i](prop_state.view(-1, self.state_dim)))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_nodes * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_nodes * self.n_edge_types,
                                         self.state_dim)  # batch size x |V||E| x state dim

            prop_state = self.propagator(in_states, out_states, prop_state, A)
        join_state = torch.cat((prop_state, annotation), 2)  # batch size x |V| x 2*state dim
        output = self.graph_rep(join_state.view(-1, self.state_dim + self.annotation_dim))
        out = self.score(output.view(-1, self.n_nodes))

        # output = output.sum(1)
        return out


class LATTICEGenerator(nn.Module):
    def __init__(self, config, input_size, graph=None):
        super(LATTICEGenerator, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=self.config.pool_kernel_size, stride=self.config.pool_kernel_size)
        self.out = nn.Linear(self.config.hidden_size, input_size)

    def forward(self):
        latent_X = torch.rand(size=self.config.latent_X_shape)  # 1*1*128*128
        cnn_out = self.conv(latent_X)  # 1*1*128*128
        output = torch.relu(cnn_out)  # 1*1*128*128
        output = torch.dropout(output, p=self.config.dropout, train=self.training)  # 1*1*128*128
        fake_sample = self.pool(output)  # 1*1*32*64
        fake_sample = fake_sample.squeeze(0)
        fake_sample = self.out(fake_sample)  # 1*32*51
        fake_sample = fake_sample.squeeze(0)  # 32*51
        return fake_sample


class LATTICEDiscriminator(nn.Module):
    def __init__(self, config, input_size):
        super(ATTAINDiscriminator, self).__init__()
        self.config = config
        self.edge_matrix = EdgeMatrix(input_size)
        self.edge_indices = self.edge_matrix.get_all_connected_indices()
        self.gcn_conv = GGNN(config.batch_size, config.batch_size)
        self.input2hidden = nn.Linear(input_size, config.hidden_size)
        self.out = nn.Linear(input_size, 2)

    def forward(self, data):
        data = data.transpose(0, 1)  # 51* 32
        edges = self.edge_indices
        output = self.gcn_conv(data, edges)  # 51* 32
        output = torch.relu(output)  # 51* 32
        output = output.transpose(0, 1)  # 32*51
        output = self.out(output)
        return output

    def get_hidden(self, x):
        h = torch.zeros(1, x.shape[0], self.config.hidden_size)
        c = torch.zeros(1, x.shape[0], self.config.hidden_size)
        return h, c
