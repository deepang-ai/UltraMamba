
from src.MusoMamba.MusoMamba import MusoMamba

from src.AttMamba.AttMamba import AttMamba

from src.mmFormer.mmFormer import mmFormer
from src.NestedFormer.nested_former import NestedFormer
from src.RFNet.models import RFNet
from src.MAML.generic_MAML import Generic_MAML
from src.RoMseg.model import RobustMseg
from src.MoSID.step3.Model.networks import MoSID

from src.MMCANET.MMCANET import CaUnet

from src.H2Aseg.H2ASeg import H2ASeg

from src.ASANet.ASANet import ASANet

from src.MMFFNet.MMFFNet import MMFFNet


def give_model(config):

    if config.finetune.model_choose == 'MusoMamba':
        model = MusoMamba(**config.models.MusoMamba)

    elif config.finetune.model_choose == 'AttMamba':
        model = AttMamba(**config.models.AttMamba)

    elif config.finetune.model_choose == 'mmFormer':
        model = mmFormer(**config.models.mmFormer)

    elif config.finetune.model_choose == 'NestedFormer':
        model = NestedFormer(**config.models.NestedFormer)

    elif config.finetune.model_choose == 'RFNet':
        model = RFNet(**config.models.RFNet)

    elif config.finetune.model_choose == 'MAML':
        model = Generic_MAML(**config.models.MAML)

    elif config.finetune.model_choose == 'RobustMseg':
        model = RobustMseg(**config.models.RobustMseg)

    elif config.finetune.model_choose == 'MoSID':
        model = MoSID(**config.models.MoSID)

    elif config.finetune.model_choose == 'MMCANET':
        model = CaUnet(**config.models.MMCANET)

    elif config.finetune.model_choose == 'H2Aseg':
        model = H2ASeg(**config.models.H2Aseg)

    elif config.finetune.model_choose == 'ASANet':
        model = ASANet(**config.models.ASANet)

    elif config.finetune.model_choose == 'MMFFNet':
        model = MMFFNet(**config.models.MMFFNet)

    else:
        assert 0, "Choose a model!"

    return model