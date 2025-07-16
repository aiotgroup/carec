from models.finetune import Finetune
from models.lwf import LwF
from models.icarl import iCaRL
from models.bic import BiC
from models.ewc import EWC
from models.ucir import UCIR
from models.wa import WA
from models.der import DER

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetune":
        return Finetune(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "icarl":
        return iCaRL(args)
    elif name == "bic":
        return BiC(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "ucir":
        return UCIR(args)
    elif name == "wa":
        return WA(args)
    elif name == "der":
        return DER(args)
    else:
        assert 0
