import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .attention import EntityAttentionLayer,EntityPoolingLayer

class AverageMessageEncoder(nn.Module):
    def __init__(self):
        super(AverageMessageEncoder, self).__init__()
    def forward(self, msg, msg_matrix):
        sum_msg = th.sum(msg, dim=2) #bs*n*msg_d
        num_msg = th.sum(msg_matrix, dim=2) #bs*n
        is_header = num_msg.bool()
        ave_msg = sum_msg/num_msg.unsqueeze(2)
        ave_msg.masked_fill_(th.logical_not(is_header).unsqueeze(2),0.0)
        return ave_msg

class LeaderMessageEncoder(nn.Module):
    def __init__(self):
        super(LeaderMessageEncoder, self).__init__()
    def forward(self, msg, msg_matrix):
        bs, n, _, msg_d = msg.shape
        msg_ind = th.arange(n, device=msg.device).reshape(1,n,1,1).repeat(bs,1,1,msg_d) #bs, n, 1, msg_d
        header_msg = th.gather(msg, 2, msg_ind).squeeze(2)#bs*n*msg_d
        num_msg = th.sum(msg_matrix, dim=2) #bs*n
        is_header = num_msg.bool()
        header_msg.masked_fill_(th.logical_not(is_header).unsqueeze(2),0.0)
        return header_msg

class AttentionMessageEncoder(nn.Module):
    def __init__(self, input_shape, args):
        super(AttentionMessageEncoder, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.msg_attn_embed_dim,
                                             args.msg_attn_embed_dim,
                                             args.msg_attn_embed_dim, args)
        else:
            self.attn = EntityPoolingLayer(args.msg_attn_embed_dim,
                                           args.msg_attn_embed_dim,
                                           args.msg_attn_embed_dim,
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(args.msg_attn_embed_dim, args.msg_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        if len(inputs) == 3:
            entities, obs_mask, entity_mask = inputs
        else:
            entities, obs_mask, entity_mask, gt_mask = inputs
        if self.args.gt_obs_mask:
            obs_mask = gt_mask
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = F.relu(self.fc1(entities))
        attn_outs = self.attn(x1, pre_mask=obs_mask,
                              post_mask=agent_mask,
                              ret_attn_logits=ret_attn_logits)
        attn_outs = F.relu(attn_outs)
        x2 = attn_outs #[bs,n_agent,msg_attn_embed_dim]
        msg = self.fc2(x2)
        # zero out output for inactive agents
        msg = msg.reshape(bs, ts, self.args.n_agents, -1)
        msg = msg.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)
        # q = q.reshape(bs * self.args.n_agents, -1)
        
        return msg