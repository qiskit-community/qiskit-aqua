import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import os

def normalize(v):
    v = np.array(v)
    return v/np.sqrt(v.dot(v.conj()))

def get_ratio_from_image(path, manipulate=True):
    im = mpimg.imread(path)
    if manipulate:
        im = manipulate_image(im)
    im = sparse.csr_matrix(im)
    nnz = im.nnz
    hu = im[:int(im.shape[0]/2), :].nnz
    hl = im[int(im.shape[0]/2):, :].nnz
    vl = im[:, :int(im.shape[1]/2)].nnz
    vr = im[:, int(im.shape[1]/2):].nnz
    vec = (hu/hl*1.3-0.62, vl/vr*0.95-0.42)
    return normalize(vec)

def get_z_rows(im, direction=1):
    i = int(direction/2-0.5)
    count = 0
    while sum(im[i]) == 0:
        i += direction
        count += 1
    return count

def get_z_cols(im, direction=1):
    i = int(direction/2-0.5)
    count = 0
    while sum(im[:, i]) == 0:
        i += direction
        count += 1
    return count

def manipulate_image(im):
    y_move = int((get_z_rows(im, direction=1)-get_z_rows(im, direction=-1))/2)
    im2 = np.zeros(im.shape)
    if y_move > 0:
        im2[:-y_move] = im[y_move:]
    elif y_move < 0:
        im2[-y_move:] = im[:y_move]
    else:
        im2 = im
    x_move = int((get_z_cols(im, direction=1)-get_z_cols(im, direction=-1))/2)
    im3 = np.zeros(im2.shape)
    if x_move > 0:
        im3[:, :-x_move] = im2[:, x_move:]
    elif x_move < 0:
        pass
        im3[:, -x_move:] = im2[:, :x_move]
    else:
        im3 = im2
    return im3

def preprocess_images(paths, labels=None, manipulate=False):
    if labels:
        return [(get_ratio_from_image(path, manipulate=manipulate), label) for path, label in
                zip(paths, labels)]
    else:
        return [get_ratio_from_image(path, manipulate=manipulate) for path in paths]

def show(path):
    im = mpimg.imread(path)
    plt.imshow(im)
    plt.show()

def get_random_set(n=8, path="img/data/", typ="easy"):
    n6 = round(min(max(0.2*np.random.randn()+0.5, 0), 1)*n)
    l6 = np.array([path+typ+"/6/"+name for name in os.listdir(path+typ+"/6/")])
    l9 = np.array([path+typ+"/9/"+name for name in os.listdir(path+typ+"/9/")])
    ret = np.concatenate([np.random.choice(l6, n6),
        np.random.choice(l9, n-n6)])
    np.random.shuffle(ret)
    return ret

def display_images(paths, size=2, title=True):
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(size*n, size*2))
    if n == 1:
        axes = [axes]
    c = 1
    for ax, path in zip(axes, paths):
        im = mpimg.imread(path)
        ax.imshow(im, cmap="binary")
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if title: ax.set_title(str(c), fontsize=20)
        ax.set_xlabel(path.split("/")[-1])
        c+=1
    plt.show()
    return fig, axes

def bulk_ratio(x="6", path = "img/data/easy/"): 
    idx = 0 if x == "9" else 1

    #r = get_ratio_from_image("train6.png")
    #print(r)
    #r = get_ratio_from_image("train9.png")
    #print(r)

    for name in os.listdir(path):
        #show("mnist_png/training/6/%s" % path)
        r = get_ratio_from_image(path + name)
        print(r)

def create_data(x="6", in_path="mnist_png/", out_path="img/data/",
        typ="easy", subs=["training", "testing"]):
    idx = 0 if x == "9" else 1
    typedef = {"easy": 0.96, "medium": 0.9, "hard": 0.8, "bad": 0}
    def typemax(typ):
        k = [1]+sorted(list(typedef.values()), reverse=True)[:-1]
        d = {key: val for key, val in zip(list(typedef.keys()), k)}
        return d[typ]
    if not os.path.exists(out_path+typ):
        os.mkdir(out_path+typ)
    if not os.path.exists(out_path+typ+"/"+x):
        os.mkdir(out_path+typ+"/"+x)
    else:
        os.system("rm -rf " + out_path+typ+"/"+x+"/*")
    c = 0
    for sub in subs:
        for name in os.listdir(in_path+sub+"/"+x):
            r = get_ratio_from_image(in_path+sub+"/"+x+"/" + name)
            if r[idx] > typedef[typ] and r[idx] < typemax(typ):
                os.system("cp " + in_path+sub+"/"+x+"/"+name
                        + " " + out_path+typ+"/"+x+"/")
                c+=1
    print("created data", x, typ, c)
            
if __name__ == "__main__":
    print(get_random_set())
