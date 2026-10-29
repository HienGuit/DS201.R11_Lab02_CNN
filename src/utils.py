import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def n_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


def build_model_tables(model, img_size, device):
    rows=[];handles=[]
    def hook(name):
        def record(layer,args,output):
            if isinstance(output,torch.Tensor):
                rows.append({'layer':name,'type':type(layer).__name__,
                             'output_shape':str(tuple(output.shape)),
                             'parameters':sum(p.numel() for p in layer.parameters(recurse=False))})
        return record
    for name,layer in model.named_modules():
        if name and not list(layer.children()):
            handles.append(layer.register_forward_hook(hook(name)))
    training=model.training
    model.eval()
    try:
        with torch.no_grad():
            output=model(torch.zeros(2,3,img_size,img_size,device=device))
        if output.ndim!=2 or output.shape[0]!=2:
            raise ValueError(f'Expected logits [N,classes], got {tuple(output.shape)}')
    finally:
        for handle in handles:handle.remove()
        model.train(training)
    tables={'layers':pd.DataFrame(rows)}
    if hasattr(model,'specs'):
        block_rows=[]
        def capture(name):
            def record(_,args,output):
                captured[name]=tuple(output.shape)
            return record
        captured={};handles=[]
        for i in range(len(model.specs)):
            handles.append(getattr(model,f'block{i+1}').register_forward_hook(capture(i)))
        model.eval()
        try:
            with torch.no_grad():model(torch.zeros(2,3,img_size,img_size,device=device))
        finally:
            for handle in handles:handle.remove()
            model.train(training)
        for i,spec in enumerate(model.specs):
            in_ch,f1,f3r,f3,f5r,f5,fp=spec
            theoretical=f1+f3+f5+fp
            actual=captured[i][1]
            if theoretical!=actual:raise ValueError('Inception channel mismatch')
            block_rows.append({'block':i+1,'input_channels':in_ch,
                               'branch_1x1':f1,'branch_3x3':f3,'branch_5x5':f5,'branch_pool':fp,
                               'computed_channels':theoretical,'actual_channels':actual,
                               'output_shape':str(captured[i]),
                               'parameters':n_params(getattr(model,f'block{i+1}'))})
        frame=pd.DataFrame(block_rows)
        tables['inception']=frame
    return tables
