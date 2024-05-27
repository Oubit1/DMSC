from networks.VNet import VNet, CCNet3d_V1,MCNet3d_v2,ACNet3d_V1,ACNet3d_V2
from networks.vnet_sdf import DTCNet

def net_factory(net_type="vnet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "UAMT" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dtc" and mode == "train":
        net = DTCNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dtc" and mode == "test":
        net = DTCNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "acnet3d_v1" and mode == "train":
        net = ACNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "acnet3d_v1" and mode == "test":
        net = ACNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "acnet3d_v2" and mode == "train":
        net = ACNet3d_V2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "acnet3d_v2" and mode == "test":
        net = ACNet3d_V2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "ccnet3d_v1" and mode == "train":
        net = CCNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "ccnet3d_v1" and mode == "test":
        net = CCNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
