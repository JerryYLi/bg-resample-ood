def resnet18(num_classes):
    from .resnet import ResNet, BasicBlock
    model = ResNet(BasicBlock, [2,2,2,2], num_classes)
    return model

def densenet100(num_classes):
    from .densenet import DenseNet
    model = DenseNet(growthRate=12, depth=100, reduction=0.5, nClasses=num_classes, bottleneck=True)
    return model

def wrn40(num_classes):
    from .wide_resnet import Wide_ResNet
    model = Wide_ResNet(depth=40, widen_factor=4, dropout_rate=0.3, num_classes=num_classes)
    return model

def wrn28(num_classes):
    from .wide_resnet import Wide_ResNet
    model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)
    return model