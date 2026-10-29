from .lenet import LeNet5
from .googlenet import InceptionBlock, GoogLeNet
from .resnet import BasicBlock, ResNet18
from .pretrained_resnet import PretrainedResnet


def build_model(cfg,num_classes,pretrained=True):
    name=cfg['model']
    if name=='lenet':
        return LeNet5(in_ch=3,img=cfg['img_size'],num_classes=num_classes,pool=cfg.get('pool','avg'))
    if name=='googlenet':
        return GoogLeNet(num_classes=num_classes)
    if name=='resnet18':
        return ResNet18(num_classes=num_classes)
    if name=='resnet50':
        mode=cfg.get('mode','finetune')
        if mode not in ('finetune','feature_extraction'):
            raise ValueError('Unknown ResNet50 mode')
        return PretrainedResnet(num_classes,mode=='feature_extraction',pretrained,
                               cfg.get('model_id','microsoft/resnet-50'))
    raise ValueError(f'Unknown model: {name}')
