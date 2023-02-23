import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime


def plot_1d(f_name,df,mean1,sd1,mean2=None,sd2=None):
    idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    count, bins, ignored = plt.hist(df, 30, density=True)
    plt.plot(bins, 1/(sd1 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mean1)**2 / (2 * sd1**2) ),
         linewidth=2, color='r')
    if sd2!=None and mean2!=None:
        plt.plot(bins, 1/(sd2 * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mean2)**2 / (2 * sd2**2) ),
            linewidth=2, color='g')
    f_name=f_name+dt_string+'.png'
    plt.savefig(f_name)
    plt.show()
    plt.close()

def main():
    mean1=0
    sd1=1
    mean2=5
    sd2=2
    #1d 1class data
    df=np.random.normal(mean1,sd1,100)*np.random.normal(mean2,sd2,100)
    
    #plot pdf
    plot_1d('1d_1c_2g_diffMean&Sd',df,mean1,sd1,mean2,sd2)

    #PCA
    pca = PCA(n_components=1)
    df=df.reshape(-1,1)
    pca.fit(df)
    new_df=pca.transform(df)
    #plot pdf of transform
    plot_1d('1d_1c_2g_diffMean&Sd',df,mean1,sd1,mean2,sd2)


if __name__=='__main__':
    main()