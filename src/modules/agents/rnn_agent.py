import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        if args.self_loc:
            if args.env == "sc2":
                if args.env_args["map_name"] == "6h_vs_8z":
                    loc_shape = 5
                elif args.env_args["map_name"] == "3s5z_vs_3s6z":
                    loc_shape = 8
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.fc_sl1 = nn.Linear(loc_shape, args.rnn_hidden_dim)
            self.fc_sl2 = nn.Linear(args.rnn_hidden_dim*2, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bs, ts, na, os = inputs.shape

        x = F.relu(self.fc1(inputs))
        if self.args.self_loc:
            if self.args.env_args["map_name"] == "6h_vs_8z":
                self_f = th.cat([inputs[:,:,:,:4], inputs[:,:,:,77:78]], dim=-1)
            elif self.args.env_args["map_name"] == "3s5z_vs_3s6z":
                self_f = th.cat([inputs[:,:,:,:4], inputs[:,:,:,132:136]], dim=-1)
            x_loc = F.relu(self.fc_sl1(self_f))
            x = F.relu(self.fc_sl2(th.cat([x, x_loc], dim=-1)))
        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x = x[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x, h)
            hs.append(h.view(bs, na, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time

        q = self.fc2(hs)
        return q, hs
