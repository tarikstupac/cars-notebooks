import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

def get_path():
    return Path('C:\\Users\\Morningstar\\Documents\\CARSv2')
    
#funkcije za ucitavanje slike
def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

# White text on black outline
def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], b[3], b[2], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
    
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)
# funkcija za rezanje slike na osnovu anotacije tj cetverougla oko auta
def crop(df, path, i):
    image = plt.imread(path/df['filename'][i])
    x1 = df['bbox_x1'][i]
    y1 = df['bbox_y1'][i]
    h = df['bbox_h'][i]
    w = df['bbox_w'][i]
    
    if len(image.shape) == 3:
        return image[y1:y1+h , x1:x1+w, :]
    else:
        # ako je slika grayscale, tj ima dva channela
        return image[y1:y1+h , x1:x1+w]

def get_x(df):
    path = Path('C:\\Users\\Morningstar\\Documents\\CARSv2')
    return path/'merged'/df['filename']

def get_y(df):
    path = Path('C:\\Users\\Morningstar\\Documents\\CARSv2')
    return df['class_name']

def compare_top_losses(k, interp, labels_df, num_imgs, path):
    path = path
    tl_val,tl_idx = interp.top_losses(k)
    classes = interp.vocab
    probs = interp.preds
    columns = 2
    rows = 2
    
    topl_idx = 0   
    for i,idx in enumerate(tl_idx):
        fig=plt.figure(figsize=(10, 8))
        columns = 2
        rows = 1
        
        # Actual Image
        act_im,cl = interp.dl.dataset[int(idx)]
        cl = int(cl)        
        act_cl = classes[cl]
        act_fn = labels_df.loc[labels_df['class_name'] == act_cl]['filename'].values[0]
        
        # Predicted Image
        pred_cl = int(np.argmax(interp.preds[int(idx)]))
        pred_cl = classes[pred_cl]
        pred_fn = labels_df.loc[labels_df['class_name'] == pred_cl]['filename'].values[0]
        
        print(f'PREDICTION:{pred_cl}, ACTUAL:{act_cl}')
        print(f'Loss: {tl_val[i]:.2f}, Probability: {probs[i][cl]:.4f}')
              
        # Add image to the left column
        img_path = 'train/' + pred_fn
        im = plt.imread(path/img_path)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(im)
        
        # Add image to the right column, need to change the tensor shape (permute) for matplotlib
        fig.add_subplot(rows, columns, 2)
        plt.imshow(act_im)

        plt.show()

def compare_most_confused(most_confused, labels_df, num_imgs, rank, path):
    path = path
    c1 = most_confused[:][rank][0]
    c2 = most_confused[:][rank][1]
    n_confused = most_confused[:3][0][1]
    print(most_confused[:][rank])
      
    # set the list of 
    f_1 = labels_df.loc[labels_df['class_name'] == c1]['filename'].values
    f_2 = labels_df.loc[labels_df['class_name'] == c2]['filename'].values

    fig=plt.figure(figsize=(10, 8))
    columns = 2
    rows = num_imgs
    for i in range(1, columns*rows +1, 2):
        # Add image to the left column
        img_path = 'train/' + f_1[i]
        im = plt.imread(path/img_path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)
        
        # Add image to the right column
        img_path = 'train/' + f_2[i]
        im = plt.imread(path/img_path)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(im)

    plt.show()
    

