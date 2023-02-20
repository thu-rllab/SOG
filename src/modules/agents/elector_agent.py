import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from modules.layers import EntityAttentionLayer, EntityPoolingLayer
from modules.mixers.flex_qmix import AttentionHyperNet

class ElectorAgent(nn.Module):
    '''
    reserver global information for leader election
    '''
    def __init__(self, input_shape, args) -> None:
        super(ElectorAgent, self).__init__()
        self.args = args
        # pre fc layer
        self.entity_fc = nn.Linear(input_shape, self.args.attn_embed_dim)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.elect_attn_embed_dim,
                                             args.elect_attn_embed_dim,
                                             args.elect_attn_embed_dim, args)
        else:
            self.attn = EntityPoolingLayer(args.elect_attn_embed_dim,
                                           args.elect_attn_embed_dim,
                                           args.elect_attn_embed_dim,
                                           args.pooling_type,
                                           args)
        # post attetion layer
        self.fc1 = nn.Linear(args.elect_attn_embed_dim, args.elect_hidden_dim)
        self.fc2 = nn.Linear(args.elect_hidden_dim, 1)
    def forward(self, entities, entity_mask, **kwargs):
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        entity_mask = entity_mask.view(bs*ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = F.relu(self.entity_fc(entities))
        attn_mask = 1 - torch.bmm((1 - agent_mask.to(torch.float)).unsqueeze(2),
                                (1 - entity_mask.to(torch.float)).unsqueeze(1))
        # TODO: same outputs seems a bug
        x_attn = self.attn(x1, pre_mask=attn_mask, post_mask = agent_mask)
        
        hidden_feat = F.relu(self.fc1(x_attn))
        output = torch.sigmoid(self.fc2(hidden_feat)).view(bs, ts, -1) # [b,]
        filter_out = output.masked_fill(agent_mask.view(bs, ts, self.args.n_agents).bool(),
            0.0) # mask dead agents
        
        dist = Bernoulli(probs=filter_out)

        return dist

class ElectorAgentPPO(ElectorAgent):
    '''
    reserver global information for leader election
    '''
    def __init__(self, input_shape, args) -> None:
        super(ElectorAgentPPO, self).__init__(input_shape, args)
        # net for value function [?? really need here]
        self.V = AttentionHyperNet(args, mode = 'scalar')
    def v(self, inputs):
        '''
        get V(s)
        '''
        entities, entity_mask = inputs
        bs, ts, ne, ed = entities.shape
        return self.V(entities, entity_mask) # B,v