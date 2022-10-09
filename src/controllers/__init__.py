REGISTRY = {}

from .basic_controller import BasicMAC
from .entity_controller import EntityMAC
from .comm_controller import CommMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["entity_mac"] = EntityMAC
REGISTRY["comm_mac"] = CommMAC