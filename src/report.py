import pandas as pd
from .artifacts import export_comparison

LABELS={
 'lenet_avg':'LeNet · AvgPool', 'lenet_max':'LeNet · MaxPool',
 'googlenet_compact':'GoogLeNet rút gọn',
 'resnet18_aug':'ResNet-18 · có augmentation',
 'resnet18_noaug':'ResNet-18 · không augmentation',
 'resnet50_features':'ResNet50 · feature extraction',
 'resnet50_finetune':'ResNet50 · fine-tuning'}
SCORE_COLUMNS={'test_accuracy':'Accuracy','test_precision_macro':'Precision macro',
               'test_recall_macro':'Recall macro','test_f1_macro':'F1 macro'}


def results_frame(results):
    return pd.DataFrame([result['summary'] for result in results.values()])


def notebook_summary(results):
    frame=results_frame(results)
    if not frame.empty and frame.split_sha256.nunique()>1:
        raise ValueError('Runs use different train/validation splits')
    table=format_comparison(frame)
    return table


def format_comparison(frame):
    if frame.empty:return frame
    df=frame.copy()
    df['Mô hình']=df.experiment.map(LABELS).fillna(df.experiment)
    df['Resize']=df.img_size.map(lambda x:f'{int(x)}×{int(x)}')
    mapping={'parameters':'Tham số','mean_seconds_per_epoch':'Giây/epoch',
             'test_accuracy':'Accuracy','test_f1_macro':'macro-F1'}
    df=df.rename(columns=mapping)
    return df.reindex(columns=['Mô hình','Resize','Tham số','Giây/epoch','Accuracy','macro-F1'])


def compare(frame,names):
    if frame.empty:return pd.DataFrame()
    selected=frame[frame.experiment.isin(names)].copy()
    if selected.split_sha256.nunique()>1:
        raise ValueError('The selected runs use different train/validation splits')
    columns=['experiment','parameters','mean_seconds_per_epoch',*SCORE_COLUMNS]
    selected=selected.reindex(columns=columns)
    selected['experiment']=selected.experiment.map(LABELS).fillna(selected.experiment)
    return selected.rename(columns={'experiment':'Cấu hình','parameters':'Tham số',
                                    'mean_seconds_per_epoch':'Giây/epoch',**SCORE_COLUMNS})


def show_result(result):
    from IPython.display import display,Image,Markdown
    summary=result['summary'];cfg=result['config']
    display(Markdown('### '+LABELS.get(summary['experiment'],summary['experiment'])))
    display(pd.DataFrame([{'Resize':f'{cfg["img_size"]}×{cfg["img_size"]}',
                           'Số tham số':summary['parameters'],
                           'Tham số được học':summary['trainable_parameters'],
                           'Best epoch':summary['best_epoch'],
                           'Giây/epoch':summary['mean_seconds_per_epoch']}]))
    if cfg['model']=='lenet':
        display(Markdown('**Bảng layer và tham số**'))
        display(result['model_tables']['layers'])
    if cfg['model']=='googlenet':
        display(Markdown('**Channel từng Inception block: tính từ các nhánh và đối chiếu output**'))
        display(result['model_tables']['inception'])
    if result['evaluation'] is not None:
        display(Markdown('**Bốn độ đo trên tập test**'))
        display(pd.DataFrame([{label:summary[key] for key,label in SCORE_COLUMNS.items()}]))
        display(Markdown('**Kết quả từng lớp (bao gồm F1)**'))
        display(result['evaluation']['per_class'])
        display(Markdown('**Confusion matrix 21×21**'))
        display(Image(data=result['evaluation']['confusion_image']))
    else:
        display(Markdown('Chỉ đánh giá validation; chưa tính chỉ số test.'))
    display(Markdown('**Loss train và validation theo epoch**'))
    display(Image(data=result['loss_image']))


def show_pair(frame,names,title,output_path=None):
    from IPython.display import display,Markdown
    display(Markdown('**'+title+'**'))
    table=compare(frame,names)
    if table.empty:
        print('Chưa có kết quả cho nhóm này.');return
    display(table)
    if output_path is not None:
        print(f'Comparison: {export_comparison(table,output_path)}')
    if len(names)==2 and len(table)==2 and 'test_f1_macro' in frame:
        selected=frame.set_index('experiment')
        if all(name in selected.index for name in names):
            a,b=(selected.loc[name,'test_f1_macro'] for name in names)
            if pd.notna(a) and pd.notna(b):
                print(f'Chênh lệch macro-F1 ({LABELS[names[1]]} − {LABELS[names[0]]}): {b-a:+.4f}')
