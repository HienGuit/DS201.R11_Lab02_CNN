from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,
                              classification_report,confusion_matrix)


def scores(y,pred,num_classes):
    p,r,f,_=precision_recall_fscore_support(
        y,pred,labels=list(range(num_classes)),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)), 'precision_macro':float(p),
            'recall_macro':float(r),'f1_macro':float(f)}


def build_evaluation(y,pred,classes):
    report=classification_report(y,pred,labels=list(range(len(classes))),
                                 target_names=classes,output_dict=True,zero_division=0)
    per_class=pd.DataFrame({name:report[name] for name in classes}).T
    per_class.index.name='class'
    cm=confusion_matrix(y,pred,labels=list(range(len(classes))))
    fig,ax=plt.subplots(figsize=(13,11))
    im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set(xticks=range(len(classes)),yticks=range(len(classes)),
           xticklabels=classes,yticklabels=classes,xlabel='Predicted',ylabel='True')
    plt.setp(ax.get_xticklabels(),rotation=90)
    fig.tight_layout()
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=160)
    image=buffer.getvalue();plt.close(fig)
    return {'per_class':per_class,'confusion_matrix':cm,'confusion_image':image}
