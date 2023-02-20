import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer


class MessageEntityAttentionRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MessageEntityAttentionRNNAgent, self).__init__()
        self.args = args
        assert self.args.use_msg
        if self.args.no_summary:
            self.args.msg_dim = self.args.attn_embed_dim
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)
        else:
            self.attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim+args.msg_dim , args.n_actions)
        self.fc_msg = nn.Linear(args.attn_embed_dim, args.msg_dim*2)
        self.inference_fc_msg = nn.Linear(args.attn_embed_dim+args.rnn_hidden_dim, args.msg_dim*2)
        self.attn_outs = None
        self.have_attn = False
        self.max_logvar = nn.Parameter((th.ones((1, args.msg_dim)).float() / 2).to(args.device), requires_grad=False)
        self.min_logvar = nn.Parameter((-th.ones((1, args.msg_dim)).float() * 10).to(args.device), requires_grad=False)
        self.use_attn_num = 0 #when set to 2, means q net and msg net used by once separately, then reset.

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ret_attn_logits=None, msg=None, ret_inf_msg=False, imagine=False,
                ret_attn_input=False, **kwargs):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        if not self.have_attn:
            entities = entities.reshape(bs * ts, ne, ed)
            obs_mask = obs_mask.reshape(bs * ts, ne, ne)
            x1 = F.relu(self.fc1(entities))
            self.attn_outs = self.attn(x1, pre_mask=obs_mask,
                                post_mask=agent_mask,
                                ret_attn_logits=ret_attn_logits)
            self.use_attn_num = 0
            self.have_attn = True
        if msg is None:
            #means I need to generate message and save attn_outs
            self.use_attn_num += 1
            if self.use_attn_num == 2:
                self.have_attn = False
            if ret_attn_logits is not None:
                x2, attn_logits = self.attn_outs
            else:
                x2 = self.attn_outs
            if imagine:
                hidden_state = hidden_state.chunk(3, dim=0)[0]
                x2 = x2.chunk(3, dim=0)[0]
            if self.args.no_summary:
                if ret_inf_msg:
                    return x2, None, None
                else:
                    return x2, None
            x3 = F.relu(self.fc_msg(x2))
            x3 = x3.reshape(bs, ts, self.args.n_agents, -1)
            mean=x3[:,:,:,:self.args.msg_dim]
            logstd = self.max_logvar - F.softplus(self.max_logvar - x3[:,:,:,self.args.msg_dim:])
            logstd = self.min_logvar + F.softplus(logstd - self.min_logvar)
            dis = th.distributions.Normal(mean, logstd.exp())
            if ret_inf_msg:
                #need the inference net for training
                h = hidden_state.detach().reshape(bs*ts, self.args.n_agents, self.args.rnn_hidden_dim)
                x4 = F.relu(self.inference_fc_msg(th.cat([x2, h], dim=2)))
                x4 = x4.reshape(bs, ts, self.args.n_agents, -1)
                inference_mean=x4[:,:,:,:self.args.msg_dim]
                inference_logstd = self.max_logvar - F.softplus(self.max_logvar - x4[:,:,:,self.args.msg_dim:])
                inference_logstd = self.min_logvar + F.softplus(inference_logstd - self.min_logvar)
                inference_dis = th.distributions.Normal(inference_mean, inference_logstd.exp())
                if ret_attn_input:
                    return  dis.rsample(), dis, inference_dis, x2
                return dis.rsample(), dis, inference_dis
            else:
                if ret_attn_input:
                    return  dis.rsample(), dis, x2
                return dis.rsample(), dis

        else:
            # means I need to use attn_outs and given msg to generate Q.
            self.use_attn_num += 1
            if self.use_attn_num == 2:
                self.have_attn = False
            if ret_attn_logits is not None:
                x2, attn_logits = self.attn_outs
            else:
                x2 = self.attn_outs
            x3 = F.relu(self.fc2(x2))
            x3 = x3.reshape(bs, ts, self.args.n_agents, -1)

            h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hs = []
            for t in range(ts):
                curr_x3 = x3[:, t].reshape(-1, self.args.rnn_hidden_dim)
                h = self.rnn(curr_x3, h)
                hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
            hs = th.stack(hs, dim=1)  # Concat over time
            if msg is not None:
                mhs = th.cat([hs, msg.reshape(bs,ts,self.args.n_agents,self.args.msg_dim)], dim=3)
            else:
                mhs = hs
            q = self.fc3(mhs)
            # zero out output for inactive agents
            q = q.reshape(bs, ts, self.args.n_agents, -1)
            q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)
            # q = q.reshape(bs * self.args.n_agents, -1)
            if ret_attn_logits is not None:
                return q, h, attn_logits.reshape(bs, ts, self.args.n_agents, ne)
            return q, hs


class MessageImagineEntityAttentionRNNAgent(MessageEntityAttentionRNNAgent):
    def __init__(self, *args, **kwargs):
        super(MessageImagineEntityAttentionRNNAgent, self).__init__(*args, **kwargs)

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

    def forward(self, inputs, hidden_state, imagine=False, ret_inf_msg=False, msg=None, **kwargs):
        if not imagine or ret_inf_msg ==True:
            return super(MessageImagineEntityAttentionRNNAgent, self).forward(inputs, hidden_state, ret_inf_msg=ret_inf_msg, msg=msg, imagine=imagine, **kwargs)
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
        if msg is not None:
            msg = msg.repeat(3,1,1,1)

        inputs = (entities, obs_mask, entity_mask)
        hidden_state = hidden_state.repeat(3, 1, 1)
        outs = super(MessageImagineEntityAttentionRNNAgent, self).forward(inputs, hidden_state, ret_inf_msg=ret_inf_msg, msg=msg, imagine=imagine, **kwargs)
        return outs+((Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1)), )
