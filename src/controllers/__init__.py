REGISTRY = {}


from .rlcomm_controller import RlCommMAC
from .basic_controller import BasicMAC
from .entity_controller import EntityMAC
from .comm_controller import CommMAC
from .copa_controller import COPAMAC
from .gat_controller import GatMAC
from .dppcomm_contorller import DppCommMac

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["entity_mac"] = EntityMAC
REGISTRY["comm_mac"] = CommMAC
REGISTRY["copa_mac"] = COPAMAC
REGISTRY["gat_mac"] = GatMAC

REGISTRY["rlcomm_mac"] = RlCommMAC
REGISTRY["dppcomm_mac"] = DppCommMac


