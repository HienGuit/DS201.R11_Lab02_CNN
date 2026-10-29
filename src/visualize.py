from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history):
    df=pd.DataFrame(history)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(df.epoch,df.train_loss,label='Train loss')
    ax.plot(df.epoch,df.val_loss,label='Validation loss')
    ax.set(xlabel='Epoch',ylabel='Loss',title='Train / validation loss')
    ax.legend();fig.tight_layout()
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=160)
    image=buffer.getvalue();plt.close(fig)
    return image
