import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer
import torch.distributions as D


class EntityAttentionCOPAAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionCOPAAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        self.fc1_coach = nn.Linear(input_shape, args.attn_embed_dim)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)
            self.coach_attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)
            
        else:
            self.attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.pooling_type,
                                           args)
            self.coach_attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.pooling_type,
                                           args)
        # self.fc2_coach = nn.Linear(args.attn_embed_dim, args.mixing_embed_dim)    
        self.fc2 = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)
        self.fc_msg = nn.Linear(args.attn_embed_dim, args.msg_dim*2)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim+args.msg_dim, args.n_actions)
        self.fc_q = nn.Linear(args.rnn_hidden_dim, args.msg_dim*2)
        self.zt = None
        self.max_logvar = nn.Parameter((th.ones((1, args.msg_dim)).float() / 2).to(args.device), requires_grad=False)
        self.min_logvar = nn.Parameter((-th.ones((1, args.msg_dim)).float() * 10).to(args.device), requires_grad=False)
        self.attn_weights=None
        

    def init_hidden(self):
        # make hidden states on same device as model
        self.attn_weights=None
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ret_attn_logits=None, new_msg=False, ret_attn_weights=False):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = F.relu(self.fc1(entities))
        x1_coach = F.relu(self.fc1_coach(entities))
        attn_outs = self.attn(x1, pre_mask=obs_mask,
                              post_mask=agent_mask,
                              ret_attn_logits=ret_attn_logits,
                              ret_attn_weights=ret_attn_weights)
        agent_mask = entity_mask[:, :self.args.n_agents]
        attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2_coach = self.coach_attn(x1_coach, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask)
        # h_team = self.fc2_coach(x2_coach)
        # h_team = h_team.masked_fill(agent_mask.unsqueeze(2).bool(), 0)
        if ret_attn_logits is not None:
            x2, attn_logits = attn_outs
        elif ret_attn_weights:
            x2, attn_weights = attn_outs
            if self.attn_weights is None:
                self.attn_weights = attn_weights
            else:
                self.attn_weights=th.cat([self.attn_weights, attn_weights], dim=0)
        else:
            x2 = attn_outs
        zt_logits = F.relu(self.fc_msg(x2_coach)) # should be [bs, ts, n, msg_d]
        zt_logits = zt_logits.reshape(bs,ts,self.args.n_agents, self.args.msg_dim*2)

        mean=zt_logits[:,:,:,:self.args.msg_dim]
        logstd = self.max_logvar - F.softplus(self.max_logvar - zt_logits[:,:,:,self.args.msg_dim:])
        logstd = self.min_logvar + F.softplus(logstd - self.min_logvar)
        zt_dis = th.distributions.Normal(mean, logstd.exp())
        zt = zt_dis.rsample()
        if ts==1: #means inference, use new_msg to check
            self.zt = zt if new_msg else self.zt.detach()
        else: #means training, 
            self.zt = th.zeros_like(zt)
            Ti=0
            for t in range(ts):
                if t % self.args.msg_T==0:
                    Ti=t
                self.zt[:,t] = zt[:, Ti]
        x3 = F.relu(self.fc2(x2))
        x3 = x3.reshape(bs, ts, self.args.n_agents, -1)

        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x3 = x3[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x3, h)
            hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time
        msg_q_logits = self.fc_q(hs)
        q = self.fc3(th.cat([hs, self.zt], dim=-1))
        # zero out output for inactive agents
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)
        # q = q.reshape(bs * self.args.n_agents, -1)
        if ret_attn_logits is not None:
            return q, h, zt_logits, msg_q_logits, attn_logits.reshape(bs, ts, self.args.n_agents, ne)
        return q, hs, zt_logits, msg_q_logits


class ImagineEntityAttentionCOPAAgent(EntityAttentionCOPAAgent):
    def __init__(self, *args, **kwargs):
        super(ImagineEntityAttentionCOPAAgent, self).__init__(*args, **kwargs)

    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def entitymask2attnmask(self, entity_mask):
        bs, ts, ne = entity_mask.shape
        # agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - entity_mask.to(th.float)).reshape(bs * ts, ne, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)

    def forward(self, inputs, hidden_state, imagine=False, new_msg=False, **kwargs):
        if not imagine:
            return super(ImagineEntityAttentionCOPAAgent, self).forward(inputs, hidden_state, new_msg=new_msg, **kwargs)
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)
        groupB = self.logical_not(groupA)
        # mask out entities not present in env
        groupA = self.logical_or(groupA, entity_mask[:, [0]])
        groupB = self.logical_or(groupB, entity_mask[:, [0]])

        # convert entity mask to attention mask
        groupAattnmask = self.entitymask2attnmask(groupA)
        groupBattnmask = self.entitymask2attnmask(groupB)
        # create attention mask for interactions between groups
        interactattnmask = self.logical_or(self.logical_not(groupAattnmask),
                                           self.logical_not(groupBattnmask))
        # get within group attention mask
        withinattnmask = self.logical_not(interactattnmask)

        activeattnmask = self.entitymask2attnmask(entity_mask[:, [0]])
        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self.logical_or(withinattnmask, activeattnmask)
        Iattnmask_noobs = self.logical_or(interactattnmask, activeattnmask)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self.logical_or(withinattnmask, obs_mask)
        interactattnmask = self.logical_or(interactattnmask, obs_mask)

        entities = entities.repeat(3, 1, 1, 1)
        obs_mask = th.cat([obs_mask, withinattnmask, interactattnmask], dim=0)
        entity_mask = entity_mask.repeat(3, 1, 1)

        inputs = (entities, obs_mask, entity_mask)
        hidden_state = hidden_state.repeat(3, 1, 1)
        q, h, zt_logits, msg_q_logits = super(ImagineEntityAttentionCOPAAgent, self).forward(inputs, hidden_state, new_msg=new_msg)
        return q, h, zt_logits, msg_q_logits, (Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1))
