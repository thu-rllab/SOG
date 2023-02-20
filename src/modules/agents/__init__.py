REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .entity_rnn_agent import ImagineEntityAttentionRNNAgent, EntityAttentionRNNAgent
from .entity_ff_agent import EntityAttentionFFAgent, ImagineEntityAttentionFFAgent
from .msg_entity_rnn_agent import MessageEntityAttentionRNNAgent, MessageImagineEntityAttentionRNNAgent
from .entity_copa_agent import EntityAttentionCOPAAgent, ImagineEntityAttentionCOPAAgent
from .entity_rnn_gat_agent import EntityAttentionRNNGATAgent, ImagineEntityAttentionRNNGATAgent
from .entity_rnn_msg_agent import EntityAttentionRNNMsgAgent, ImagineEntityAttentionRNNMsgAgent
from .elector_agent import ElectorAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["entity_attend_ff"] = EntityAttentionFFAgent
REGISTRY["imagine_entity_attend_ff"] = ImagineEntityAttentionFFAgent
REGISTRY["entity_attend_rnn"] = EntityAttentionRNNAgent
REGISTRY["imagine_entity_attend_rnn"] = ImagineEntityAttentionRNNAgent
REGISTRY["comm_entity_attend_rnn"] = MessageEntityAttentionRNNAgent
REGISTRY["comm_imagine_entity_attend_rnn"] = MessageImagineEntityAttentionRNNAgent
REGISTRY["entity_attend_copa"] = EntityAttentionCOPAAgent
REGISTRY["imagine_entity_attend_copa"] = ImagineEntityAttentionCOPAAgent
REGISTRY["imagine_entity_attend_rnn_gat"] = ImagineEntityAttentionRNNGATAgent
REGISTRY["entity_attend_rnn_gat"] = EntityAttentionRNNGATAgent
REGISTRY["entity_attend_rnn_msg"] = EntityAttentionRNNMsgAgent
REGISTRY["imagine_entity_attend_rnn_msg"] = ImagineEntityAttentionRNNMsgAgent
REGISTRY["election_agent"] = ElectorAgent

