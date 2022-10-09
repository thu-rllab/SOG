from .q_learner import QLearner
from .msg_q_learner import MsgQLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["msg_q_learner"] = MsgQLearner

