from .dip_acoustic_model import DIP_AcousticModel
from .dip_acoustic_fwi import DIP_AcousticFWI
from .dip_elastic_model import DIP_ElasticModel
from .dip_vti_model import DIP_VTIModel
from .dip_elastic_fwi import DIP_ElasticFWI

from .model.Unet import UNet as DIP_Unet
from .model.CNN import  CNN as DIP_CNN
from .model.ResNet import ResNet as DIP_ResNet
from .model.MLP import MLP as DIP_MLP
from .model.MobileNetV2 import MobileNetV2FWI as DIP_MobilNetV2