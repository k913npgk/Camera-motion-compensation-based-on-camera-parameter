import os
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

if __name__ == '__main__':
    
    transform = transforms.Compose(
        [transforms.Resize([224, 80]), transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.ImageFolder(root=os.path.join("trans_img/datasets", "train"), transform=transform)
    training_set_2 = torchvision.datasets.ImageFolder(root=os.path.join("trans_img/datasets2", "train"), transform=transform)
    validation_set = torchvision.datasets.ImageFolder(root=os.path.join("trans_img/datasets", "valid"), transform=transform)
    validation_set_2 = torchvision.datasets.ImageFolder(root=os.path.join("trans_img/datasets2", "valid"), transform=transform)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)
    training_loader_2 = torch.utils.data.DataLoader(training_set_2, batch_size=4, shuffle=True, num_workers=2)
    validation_loader_2 = torch.utils.data.DataLoader(validation_set_2, batch_size=4, shuffle=False, num_workers=2)

    # Class labels
    classes = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    #up--dataload
    import torch.nn as nn
    from resnet import *
    import torch.nn.functional as F
    
    class BasicBlock(nn.Module):
        def __init__(self, inplanes, planes, stride=1, dilation=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                                stride=2, padding=1,
                                bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=2, padding=1,
                                bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.stride = stride
            
            self.con1x1 = nn.Conv2d(planes, planes*2, kernel_size=1,
                                stride=1)
            self.bn3 = nn.BatchNorm2d(planes*2)

        def forward(self, x, residual=None):
    #         if residual is None:
    #             residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

    #         out += residual
            out = self.relu(out)
            
            #out = self.con1x1(out)

            return out

    class patchLinearAttention(nn.Module):
        def __init__(self, chan = 128, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 32, value_dim = 32, heads = 4, norm_queries = True):
            super().__init__()
            self.chan = chan
            chan_out = chan if chan_out is None else chan_out

            self.key_dim = key_dim
            self.value_dim = value_dim
            self.heads = heads

            self.norm_queries = norm_queries

            conv_kwargs = {'padding': padding, 'stride': stride}
            self.to_q = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
            self.to_k = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
            self.to_v = nn.Conv2d(chan, value_dim * heads, kernel_size, **conv_kwargs)

            out_conv_kwargs = {'padding': padding}
            self.to_out = nn.Conv2d(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)
            
        def forward(self, x, y, context = None):
            b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

            q, k, v = (self.to_q(x), self.to_k(y), self.to_v(y))
            
            q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))

            q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

            if context is not None:
                context = context.reshape(b, c, 1, -1)
                ck, cv = self.to_k(context), self.to_v(context)
                ck, cv = map(lambda t: t.reshape(b, heads, k_dim, -1), (ck, cv))
                k = torch.cat((k, ck), dim=3)
                v = torch.cat((v, cv), dim=3)

            k = k.softmax(dim=-1)

            if self.norm_queries:
                q = q.softmax(dim=-2)

            context = torch.einsum('bhdn,bhen->bhde', k, v)
            out = torch.einsum('bhdn,bhde->bhen', q, context)
            out = out.reshape(b, -1, h, w)
            out = self.to_out(out)
            return out
        
    class DLASeg(nn.Module):
        def __init__(self):
            super(DLASeg, self).__init__()
            
            self.conv1 = BasicBlock(3,64)

            self.patch_attention = patchLinearAttention(chan = 32)

    #         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            #self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d((1,1))
    #         self.oneone = nn.Conv2d(512, 128, kernel_size=1)
    #         self.SELayer = SELayer(channel = 128)
            self.fc = self._construct_fc_layer(
                128, 128, dropout_p=None
            )
            self.resnet = resnet18(pretrained=True,
                                            replace_stride_with_dilation=[False,True,True])
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            
            self.resnet_stages_num = 4
            expand = 1 
            
            if self.resnet_stages_num == 5:
                layers = 512 * expand
            elif self.resnet_stages_num == 4:
                layers = 256 * expand
            elif self.resnet_stages_num == 3:
                layers = 128 * expand
            else:
                raise NotImplementedError
                
            self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
            if fc_dims is None or fc_dims < 0:
                self.feature_dim = input_dim
                return None

            if isinstance(fc_dims, int):    
                fc_dims = [fc_dims]

            layers = []
            for dim in fc_dims:
                layers.append(nn.Linear(input_dim, dim))
                layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout_p is not None:
                    layers.append(nn.Dropout(p=dropout_p))
                input_dim = dim

            self.feature_dim = fc_dims[-1]

            return nn.Sequential(*layers)




        def inference_forward_fast(self, x1):
            
            x1 = x1.unsqueeze(-1)
        
            x1 = x1.permute(3,2,0,1)
        
            x1 = F.interpolate(x1, (224,80), mode='bilinear')
    
            x1 = self.forward_single(x1) # shape: (B,32,28,10)
        
            width = x1.shape[-1]     
            height = x1.shape[-2]   
            width = int(width)      
            height = int(height)     

            # 整張
            temp_all = x1
            # 左上
            temp_lup = x1[:,:,0:(height//2),0:(width//2)]
            # 右上
            temp_rup = x1[:,:,0:(height//2),(width//2):width]
            # 左下
            temp_ldown = x1[:,:,(height//2):height,0:(width//2)]
            # 右下
            temp_rdown = x1[:,:,(height//2):height,(width//2):width]
        
        
            #round1
            A = self.patch_attention(temp_lup, temp_lup)
            B = self.patch_attention(temp_lup, temp_rup)
            C = self.patch_attention(temp_lup, temp_ldown)
            D = self.patch_attention(temp_lup, temp_rdown)
            final1 = A + B + C + D 

            #round2
            A = self.patch_attention(temp_rup, temp_rup)
            B = self.patch_attention(temp_rup, temp_lup)
            C = self.patch_attention(temp_rup, temp_ldown)
            D = self.patch_attention(temp_rup, temp_rdown)
            final2 = A + B + C + D 

            #round3
            A = self.patch_attention(temp_ldown, temp_ldown)
            B = self.patch_attention(temp_ldown, temp_rup)
            C = self.patch_attention(temp_ldown, temp_lup)
            D = self.patch_attention(temp_ldown, temp_rdown)
            final3 = A + B + C + D 

            #round4
            A = self.patch_attention(temp_rdown, temp_rdown)
            B = self.patch_attention(temp_rdown, temp_lup)
            C = self.patch_attention(temp_rdown, temp_rup)
            D = self.patch_attention(temp_rdown, temp_ldown)
            final4 = A + B + C + D 
    
            v1 = torch.cat((final1,final2,final3,final4),1) 
        
            v1 = self.maxpool(v1)
        
            v1 = v1.squeeze(-1)
            v1 = v1.squeeze(-1)
        
            v1 = self.fc(v1)
            
            
            
            return v1
        
        
        def forward(self, x1, x2): 
            
            x1 = x1.permute(0,1,2,3)   # shape: (B,64,224,80)
            x2 = x2.permute(0,1,2,3)
            x1 = x1.float() 
            x2 = x2.float() 

            # x1 = self.conv1(x1)  # shape: (B,64,56,20)
            # x2 = self.conv1(x2)
            x1 = self.forward_single(x1) # shape: (B,32,28,10)
            x2 = self.forward_single(x2)
        
            width = x1.shape[-1]     
            height = x1.shape[-2]   
            width = int(width)      
            height = int(height)     

            # 整張
            temp_all = x1
            # 左上
            temp_lup = x1[:,:,0:(height//2),0:(width//2)]
            # 右上
            temp_rup = x1[:,:,0:(height//2),(width//2):width]
            # 左下
            temp_ldown = x1[:,:,(height//2):height,0:(width//2)]
            # 右下
            temp_rdown = x1[:,:,(height//2):height,(width//2):width]
        
        
            #round1
            A = self.patch_attention(temp_lup, temp_lup)
            B = self.patch_attention(temp_lup, temp_rup)
            C = self.patch_attention(temp_lup, temp_ldown)
            D = self.patch_attention(temp_lup, temp_rdown)
            final1 = A + B + C + D 

            #round2
            A = self.patch_attention(temp_rup, temp_rup)
            B = self.patch_attention(temp_rup, temp_lup)
            C = self.patch_attention(temp_rup, temp_ldown)
            D = self.patch_attention(temp_rup, temp_rdown)
            final2 = A + B + C + D 

            #round3
            A = self.patch_attention(temp_ldown, temp_ldown)
            B = self.patch_attention(temp_ldown, temp_rup)
            C = self.patch_attention(temp_ldown, temp_lup)
            D = self.patch_attention(temp_ldown, temp_rdown)
            final3 = A + B + C + D 

            #round4
            A = self.patch_attention(temp_rdown, temp_rdown)
            B = self.patch_attention(temp_rdown, temp_lup)
            C = self.patch_attention(temp_rdown, temp_rup)
            D = self.patch_attention(temp_rdown, temp_ldown)
            final4 = A + B + C + D 
    
            v1 = torch.cat((final1,final2,final3,final4),1) 
        
            v1 = self.maxpool(v1)
        
            v1 = v1.squeeze(-1)
            v1 = v1.squeeze(-1)
        
            v1 = self.fc(v1)
            
            
            # 整張
            temp_all = x2
            # 左上
            temp_lup = x2[:,:,0:(height//2),0:(width//2)]
            # 右上
            temp_rup = x2[:,:,0:(height//2),(width//2):width]
            # 左下
            temp_ldown = x2[:,:,(height//2):height,0:(width//2)]
            # 右下
            temp_rdown = x2[:,:,(height//2):height,(width//2):width]
        
        
            #round1
            A = self.patch_attention(temp_lup, temp_lup)
            B = self.patch_attention(temp_lup, temp_rup)
            C = self.patch_attention(temp_lup, temp_ldown)
            D = self.patch_attention(temp_lup, temp_rdown)
            final1 = A + B + C + D 

            #round2
            A = self.patch_attention(temp_rup, temp_rup)
            B = self.patch_attention(temp_rup, temp_lup)
            C = self.patch_attention(temp_rup, temp_ldown)
            D = self.patch_attention(temp_rup, temp_rdown)
            final2 = A + B + C + D 

            #round3
            A = self.patch_attention(temp_ldown, temp_ldown)
            B = self.patch_attention(temp_ldown, temp_rup)
            C = self.patch_attention(temp_ldown, temp_lup)
            D = self.patch_attention(temp_ldown, temp_rdown)
            final3 = A + B + C + D 

            #round4
            A = self.patch_attention(temp_rdown, temp_rdown)
            B = self.patch_attention(temp_rdown, temp_lup)
            C = self.patch_attention(temp_rdown, temp_rup)
            D = self.patch_attention(temp_rdown, temp_ldown)
            final4 = A + B + C + D 
    
            v2 = torch.cat((final1,final2,final3,final4),1) 
        
            v2 = self.maxpool(v2)
        
            v2 = v2.squeeze(-1)
            v2 = v2.squeeze(-1)
        
            v2 = self.fc(v2)
            
            
            sim = self.cos(v1,v2)
            
            return sim
        def forward_single(self, x):
                
            # resnet layers
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
            x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

            if self.resnet_stages_num > 3:
                x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

            if self.resnet_stages_num == 5:
                x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            elif self.resnet_stages_num > 5:
                raise NotImplementedError

            x = x_8
            # output layers
            x = self.conv_pred(x)
            return x 

    def load_model(model_path, optimizer=None, resume=False, 
                lr=None, lr_step=None):
        
        model = DLASeg()
        start_epoch = 0
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch_id']))
        state_dict_ = checkpoint['model_G_state_dict']
        state_dict = {}
        
        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        
        # check loaded parameters and created model parameters
        msg = 'If you see this, your model does not fully load the ' + \
                'pre-trained weight. Please make sure ' + \
                'you have correctly specified --arch xxx ' + \
                'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, '\
                        'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        
        # resume optimizer parameters
        if optimizer is not None and resume:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                start_lr = lr
                for step in lr_step:
                    if start_epoch >= step:
                        start_lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = start_lr
                print('Resumed optimizer with start lr', start_lr)
            else:
                print('No optimizer parameters in checkpoint.')
        if optimizer is not None:
            return model, optimizer, start_epoch
        else:
            return model
    

    model = load_model("ver12.pt")
    #up--model

    loss_fn = torch.nn.CrossEntropyLoss()

    # NB: Loss functions expect data in batches, so we're creating batches of 4
    # Represents the model's confidence in each of the 10 classes for a given input
    dummy_outputs = torch.rand(4, 10)
    # Represents the correct class among the 10 being tested
    dummy_labels = torch.tensor([1, 5, 3, 7])

    print(dummy_outputs)
    print(dummy_labels)

    loss = loss_fn(dummy_outputs, dummy_labels)
    print('Total loss for this batch: {}'.format(loss.item()))

    #up--loss

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        data_2 = list(enumerate(training_loader_2))
        label_data = open("labels.txt", mode="r")
        label = label_data.read().split("\n")
        label = list(map(int, label))
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs2, labels2 = data_2[i][1]
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            labels = label[i]
            # Make predictions for this batch
            outputs = model(inputs, inputs2)
            
            labels=torch.tensor(labels, dtype=torch.long)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    #up--train-epoch

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 50

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.cuda()
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        # model.train(False)

        running_vloss = 0.0
        vdata_2 = list(enumerate(validation_loader_2))
        vlabel_data = open("vlabels.txt", mode="r")
        vlabel = vlabel_data.read().split("\n")
        vlabel = list(map(int, vlabel))
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs2, vlabels2 = vdata_2[i][1]
            voutputs = model(vinputs, vinputs2)
            vlabels = vlabel[i]
            vlabels=torch.tensor(vlabels, dtype=torch.long)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save({
            'epoch': epoch,
            'model_G_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss}, model_path)

        epoch_number += 1

    #up--run model


