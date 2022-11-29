from layer import ConvolutionLayer, ReluLayer, MaxPoolLayer, FullyConnectLayer, FlattenLayer


class VGG16(object):
    def __init__(self, num_classes=4):
        # TODO 根据网络图搭建VGG16模型
        self.layer1_conv1 = ConvolutionLayer(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.layer1_relu1 = ReluLayer()
        self.layer1_conv2 = ConvolutionLayer(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.layer1_relu2 = ReluLayer()
        self.layer1_maxpool = MaxPoolLayer(kernel_size=2, stride=2)

        self.layer2_conv1 = ConvolutionLayer(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer2_relu1 = ReluLayer()
        self.layer2_conv2 = ConvolutionLayer(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.layer2_relu2 = ReluLayer()
        self.layer2_maxpool = MaxPoolLayer(kernel_size=2, stride=2)

        self.layer3_conv1 = ConvolutionLayer(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.layer3_relu1 = ReluLayer()
        self.layer3_conv2 = ConvolutionLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.layer3_relu2 = ReluLayer()
        self.layer3_conv3 = ConvolutionLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.layer3_relu3 = ReluLayer()
        self.layer3_maxpool = MaxPoolLayer(kernel_size=2, stride=2)

        self.layer4_conv1 = ConvolutionLayer(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.layer4_relu1 = ReluLayer()
        self.layer4_conv2 = ConvolutionLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer4_relu2 = ReluLayer()
        self.layer4_conv3 = ConvolutionLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer4_relu3 = ReluLayer()
        self.layer4_maxpool = MaxPoolLayer(kernel_size=2, stride=2)

        self.layer5_conv1 = ConvolutionLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer5_relu1 = ReluLayer()
        self.layer5_conv2 = ConvolutionLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer5_relu2 = ReluLayer()
        self.layer5_conv3 = ConvolutionLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer5_relu3 = ReluLayer()
        self.layer5_maxpool = MaxPoolLayer(kernel_size=2, stride=2)

        self.flatten = FlattenLayer()
        self.fullyconnect1 = FullyConnectLayer(in_features=512 * 7 * 7, out_features=4096)
        self.relu_1 = ReluLayer()
        self.fullyconnect2 = FullyConnectLayer(in_features=4096, out_features=4096)
        self.relu_2 = ReluLayer()
        self.fullyconnect3 = FullyConnectLayer(in_features=4096, out_features=num_classes)

        self.graph_layers = None
        self.create_graph()

    def create_graph(self):
        self.graph_layers = {
            'layer1_conv1': self.layer1_conv1, 'layer1_relu1': self.layer1_relu1,
            'layer1_conv2': self.layer1_conv2, 'layer1_relu2': self.layer1_relu2,
            'layer1_maxpool': self.layer1_maxpool,

            'layer2_conv1': self.layer2_conv1, 'layer2_relu1': self.layer2_relu1,
            'layer2_conv2': self.layer2_conv2, 'layer2_relu2': self.layer2_relu2,
            'layer2_maxpool': self.layer2_maxpool,

            'layer3_conv1': self.layer3_conv1, 'layer3_relu1': self.layer3_relu1,
            'layer3_conv2': self.layer3_conv2, 'layer3_relu2': self.layer3_relu2,
            'layer3_conv3': self.layer3_conv3, 'layer3_relu3': self.layer3_relu3,
            'layer3_maxpool': self.layer3_maxpool,

            'layer4_conv1': self.layer4_conv1, 'layer4_relu1': self.layer4_relu1,
            'layer4_conv2': self.layer4_conv2, 'layer4_relu2': self.layer4_relu2,
            'layer4_conv3': self.layer4_conv3, 'layer4_relu3': self.layer4_relu3,
            'layer4_maxpool': self.layer4_maxpool,

            'layer5_conv1': self.layer5_conv1, 'layer5_relu1': self.layer5_relu1,
            'layer5_conv2': self.layer5_conv2, 'layer5_relu2': self.layer5_relu2,
            'layer5_conv3': self.layer5_conv3, 'layer5_relu3': self.layer5_relu3,
            'layer5_maxpool': self.layer5_maxpool,

            'flatten': self.flatten,
            'fullyconnect1': self.fullyconnect1, 'relu1': self.relu_1,
            'fullyconnect2': self.fullyconnect2, 'relu2': self.relu_2,
            'fullyconnect3': self.fullyconnect3,
        }

    def forward(self, x):
        for name in self.graph_layers.keys():
            print(f'forward: {name}: {x.mean()} {x.sum()}')
            x = self.graph_layers[name].forward(x)
        return x

    def backward(self, grad):
        for name in reversed(list(self.graph_layers.keys())):
            #print(f'backward: {name}: {grad.mean()} {grad.sum()}')
            grad = self.graph_layers[name].backward(grad)
        return grad

    def resume_weights(self, ckpt):
        for name, params in ckpt.items():
            self.graph_layers[name].load_params(params['weight'], params['bias'])
        print('reloaded success')