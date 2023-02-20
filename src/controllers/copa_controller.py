from .entity_controller import EntityMAC
import torch as th


# This multi-agent controller shares parameters between agents and takes
# entities + observation masks as input
class COPAMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(EntityMAC, self).__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ret_agent_outs=False, ret_attn_weights=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if ret_attn_weights:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, ret_attn_weights=True)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs]
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, need_msg=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True
        if t.stop-t.start == 1 and t.start % self.args.msg_T==0:
            new_msg = True
        else:
            new_msg = False
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if kwargs.get('imagine', False):
            agent_outs, self.hidden_states, zt_logits, msg_q_logits, groups = self.agent(agent_inputs, self.hidden_states, new_msg=new_msg, **kwargs)
        elif kwargs.get('ret_attn_weights', False):
            agent_outs, self.hidden_states, zt_logits, msg_q_logits = self.agent(agent_inputs, self.hidden_states, new_msg=new_msg, ret_attn_weights=True)
        else:
            agent_outs, self.hidden_states, zt_logits, msg_q_logits = self.agent(agent_inputs, self.hidden_states, new_msg=new_msg)
        outs = (agent_outs.squeeze(1),) if int_t else (agent_outs,)
        if kwargs.get('imagine', False):
            outs += (groups,)
        if need_msg:
            outs += (zt_logits, msg_q_logits,)
        if len(outs)==1:
            outs=outs[0]
        return outs
        