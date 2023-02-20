from .entity_controller import EntityMAC
import torch as th


# This multi-agent controller shares parameters between agents and takes
# entities + observation masks as input
class GatMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(GatMAC, self).__init__(scheme, groups, args)

    def forward(self, ep_batch, t, test_mode=False, train_mode=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if kwargs.get('imagine', False):
            agent_outs, self.hidden_states, x2_gate, groups = self.agent(agent_inputs, self.hidden_states, **kwargs)
        else:
            agent_outs, self.hidden_states, x2_gate = self.agent(agent_inputs, self.hidden_states, **kwargs)
        if int_t:
            return agent_outs.squeeze(1)
        if kwargs.get('imagine', False):
            if train_mode:
                return agent_outs, x2_gate, groups
            else:
                return agent_outs, groups
        if train_mode:
            return agent_outs, x2_gate
        return agent_outs