import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .flex_qmix import AttentionHyperNet


class WVDNMixer(nn.Module):
    def __init__(self, args):
        super(WVDNMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hyper_w = AttentionHyperNet(args, mode='alt_vector')
        if self.args.add_V:
            self.V = AttentionHyperNet(args, mode='scalar')

    def forward(self, agent_qs, inputs, imagine_groups=None):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, 1, self.n_agents * 2) #[4800,1,16]
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          ne, ne)) #[4800,8]
            w1_I = self.hyper_w(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          ne, ne)) #[4800,8]
            w1 = th.cat([w1_W, w1_I], dim=1) #[4800,16]
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)#[4800,1,8]
            w1 = self.hyper_w(entities, entity_mask) #[4800,8]
        w1=F.softmax(w1, dim=-1).unsqueeze(-1) #[4800,8,1]
        q=th.bmm(agent_qs,w1)
        if self.args.add_V:
            v = self.V(entities, entity_mask).view(-1, 1, 1)
            q += v
        q_tot = q.view(bs, -1, 1)
        return q_tot



