import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import make_loaders, test_loader
from .engine import run_epoch
from .metrics import build_evaluation
from .visualize import plot_history
from .models import build_model
from .utils import seed_everything, build_model_tables
from .artifacts import export_result


def load_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ['model','img_size','batch_size','epochs','lr','seed']:
        if key not in cfg:raise ValueError(f'Missing configuration: {key}')
    if cfg['epochs']<1 or cfg['batch_size']<1 or cfg['lr']<=0:
        raise ValueError('epochs, batch_size and lr must be positive')
    return cfg


def run_experiment(config_path,data_root,device='auto',evaluate_test=True,output_root=None):
    cfg=load_config(config_path)
    seed_everything(cfg['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else torch.device(device)
    train_dl,val_dl,classes,manifest,eval_tf=make_loaders(data_root,cfg)
    print(f'Train: {len(train_dl.dataset)} | Validation: {len(val_dl.dataset)} | Classes: {len(classes)}')
    print(f'Seed: {cfg["seed"]} | Validation split: {manifest["source"]}')
    model=build_model(cfg,len(classes)).to(device)
    counts={'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    model_tables=build_model_tables(model,cfg['img_size'],device)
    optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=cfg['lr'])
    best=-1.0;best_epoch=0;best_state=None;history=[]
    for ep in range(1,cfg['epochs']+1):
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        tr,_,_=run_epoch(model,train_dl,device,len(classes),optimizer)
        va,_,_=run_epoch(model,val_dl,device,len(classes))
        if device.type=='cuda':torch.cuda.synchronize(device)
        seconds=time.perf_counter()-start
        history.append({'epoch':ep,'seconds':seconds,
                        **{'train_'+k:v for k,v in tr.items()},
                        **{'val_'+k:v for k,v in va.items()}})
        if va['f1_macro']>best:
            best=va['f1_macro'];best_epoch=ep
            # Keep the best validation weights in memory for final evaluation.
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'Epoch {ep:02d}/{cfg["epochs"]} | train {tr["loss"]:.4f}/{tr["accuracy"]:.4f}'
              f' | val {va["loss"]:.4f}/{va["accuracy"]:.4f} | macro-F1 {va["f1_macro"]:.4f}',flush=True)
    model.load_state_dict(best_state);model.eval()
    summary={'experiment':Path(config_path).stem,'model':cfg['model'],'img_size':cfg['img_size'],
             **counts,'best_epoch':best_epoch,'best_val_f1_macro':best,
             'mean_seconds_per_epoch':float(np.mean([x['seconds'] for x in history])),
             'split_sha256':manifest['split_sha256']}
    loss_image=plot_history(history)
    evaluation=None
    if evaluate_test:
        loader,_=test_loader(data_root,cfg,classes,eval_tf)
        metrics,y,pred=run_epoch(model,loader,device,len(classes))
        evaluation=build_evaluation(y,pred,classes)
        summary.update({'test_'+k:v for k,v in metrics.items()})
    print(f'Best epoch: {best_epoch} | Validation macro-F1: {best:.4f}')
    result={'config':cfg,'classes':classes,'split':manifest,
            'summary':summary,'history':pd.DataFrame(history),'evaluation':evaluation,
            'model_tables':model_tables,'loss_image':loss_image}
    result['directory']=export_result(result,output_root)
    print(f'Outputs: {result["directory"]}')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--output-root',default=None,
                   help='Default: /kaggle/working/outputs on Kaggle; workspace outputs locally')
    p.add_argument('--validation-only',action='store_true')
    a=p.parse_args()
    result=run_experiment(a.config,a.data_root,device=a.device,
                          evaluate_test=not a.validation_only,output_root=a.output_root)
    print(pd.DataFrame([result['summary']]).to_string(index=False))
