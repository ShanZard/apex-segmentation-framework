from apex import amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tensorboardX import SummaryWriter
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class upsample(nn.Module):
    def __init__(self,incha,outcha,scale) :
        super().__init__()
        self.convvv=nn.Conv2d(incha,outcha,1,1,0,bias=False)
        self.scale=scale
    def forward(self,x):
        if self.scale==2:
            x=F.interpolate(x,scale_factor=2,mode='bilinear')
        elif self.scale==4:
            x=F.interpolate(x,scale_factor=4,mode='bilinear')
        x=self.convvv(x)
        return x   
class downsample(nn.Module):
    def __init__(self,incha,outcha,scale) :
        super().__init__()
        self.scale=scale
        self.conv2d1=nn.Conv2d(incha, outcha, kernel_size=2, stride=2)
        self.conv2d2=nn.Conv2d(outcha, outcha, kernel_size=2, stride=2)    
    def forward(self,x):
        if self.scale==2:
            x=self.conv2d1(x)
        if self.scale==4:
            x=self.conv2d1(x)
            x=self.conv2d2(x)
        if self.scale==6:
            x=self.conv2d1(x)
            x=self.conv2d2(x)
            x=self.conv2d2(x)
        return x
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,using_amp=False
                 ):
        super().__init__()
        if using_amp:
            amp.register_float_function(torch, 'sigmoid')
            amp.register_float_function(torch, 'softmax')
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )##4倍下采样
        self.relu = nn.GELU()
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.downsam1=downsample(96,192,2)
        self.downsam2=downsample(96,384,4)
        self.downsam3=downsample(96,768,6)
        self.downsam4=downsample(192,384,2)
        self.downsam5=downsample(192,768,4)
        self.upsam1=upsample(192,96,2)
        self.upsam2=upsample(384,96,4)
        self.upsam3=upsample(384,192,2)
        self.apply(self._init_weights)
        self.last_layer=nn.Sequential(
            nn.Conv2d(1440,512,kernel_size=1,stride=1,padding=0),
            LayerNorm(512, eps=1e-6,data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0),
            LayerNorm(256, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(256,2,kernel_size=1,stride=1,padding=0),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        
        x = self.downsample_layers[0](x)
        x1 = self.stages[0](x)

        x2 =self.downsample_layers[1](x1)
        x2=self.stages[1](x2)
        x1=self.stages[0](x1)

        x_12=self.downsam1(x1)
        x_13=self.downsam2(x1)
        x_21=self.upsam1(x2)
        x3=self.downsample_layers[2](x2)
        x3=self.relu(x_13+x3)
        x3=self.stages[2](x3)
        x2=self.relu(x_12+x2)
        x2=self.stages[1](x2)
        x1=self.relu(x_21+x1)
        x1=self.stages[0](x1)

        x4=self.downsample_layers[3](x3)
        x_12=self.downsam1(x1)#
        x_13=self.downsam2(x1)
        x_14=self.downsam3(x1)
        x_21=self.upsam1(x2)
        x_23=self.downsam4(x2)#192,384
        x_24=self.downsam5(x2)#192,768
        x_31=self.upsam2(x3)
        x_32=self.upsam3(x3)   
        x1=self.relu(x1+x_21+x_31) 
        x2=self.relu(x_12+x_32+x2)
        x3=self.relu(x_13+x_23+x3)
        x4=self.relu(x_14+x_24+x4)

        x4=self.stages[3](x4)#758,16,16
        x3=self.stages[2](x3)#384,32,32
        x2=self.stages[1](x2)#192,64,64
        x1=self.stages[0](x1)#92,168,168




        x1=F.interpolate(x1,size=(512,512),mode='bilinear')
        x2=F.interpolate(x2,size=(512,512),mode='bilinear')
        x3=F.interpolate(x3,size=(512,512),mode='bilinear')
        x4=F.interpolate(x4,size=(512,512),mode='bilinear')
        feats = torch.cat([x1, x2, x3,x4], 1)
        out=self.last_layer(feats)
        return out # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model



if __name__=='__main__':
    a=torch.rand(1,3,512,512)
    model=convnext_tiny()
    b=model(a)
    print(b)
    # dummy_input = torch.rand(1, 3, 512 ,512) #假设输入13张1*28*28的图片
    # model = convnext_tiny()
    # with SummaryWriter(comment='LeNet') as w:
    #     w.add_graph(model, (dummy_input, ))