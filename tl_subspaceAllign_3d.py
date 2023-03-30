import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits import mplot3d

def inv_sigmoid(value):
    return np.log(value/(1-value))

def gaussian_mixture_model(mus,covs,n):
    # mus = [np.array([-2, 2]), np.array([2, -2])]
    # covs = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]

    pis = np.array([0.5, 0.5])
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
    assert np.isclose(acc_pis[-1], 1)

    # n = 1000
    samples = []

    for i in range(n):
        # sample uniform
        r = np.random.uniform(0, 1)
        # select gaussian
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break

        selected_mu = mus[k]
        selected_cov = covs[k]

        # sample from selected gaussian
        lambda_, gamma_ = np.linalg.eig(selected_cov)

        dimensions = len(lambda_)
        # sampling from normal distribution
        y_s = np.random.uniform(0, 1, size=(dimensions * 1, 3))
        x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
        # transforming into multivariate distribution
        x_multi = (x_normal * lambda_) @ gamma_ + selected_mu
        samples.append(x_multi.tolist()[0])

    samples = np.array(samples)
    return samples

def rot_3d(deg,df):
    rad_rot=np.deg2rad(deg)
    rot_mat=np.array([[1,0,0],[0,np.cos(rad_rot),np.sin(rad_rot)],[0,-np.sin(rad_rot),np.cos(rad_rot)]]) #rot around x
    df_rot=np.dot(df,rot_mat)
    return df_rot

def pca(samp):
    pca = PCA(n_components=2)
    pca.fit(samp)
    new_df=pca.transform(samp)
    # new_df = np.array(new_df)
    return new_df,pca.components_.T

def space_alignment(X_s,X_t):
    m = np.matmul(X_s.T,X_t)
    return m

def plot(df_c0,df_c1,name):
    fig = plt.figure()
    plt.scatter(df_c0[:, 0], df_c0[:, 1],color='r',label='class_0')
    plt.scatter(df_c1[:, 0], df_c1[:, 1],color='g',label='class_1')
    plt.title('Gaussian Mixture Model Samples:'+name)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=name+dt_string
    # plt.savefig(f_name+'.png')
    plt.show()


if __name__=='__main__':
    n=1000
    # source domain: GMM - class 0
    src_mus_c0 = [np.array([-6, 12, -6]), np.array([-6,6,-6])]
    src_covs_c0 = [np.array([[1, 0.8, 0.2], [0.8, 1, 0.5], [0.2, 0.5, 1]]), np.array([[0.5, 0, 0], [0, 0.5, 0], [0,0,0.5]])]
    src_c0 = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)

    # source domain: GMM - class 1
    src_mus_c1 = [np.array([-6, -12, -6]), np.array([-6,-6, -6])]
    src_covs_c1 = [np.array([[1, 0.8, 0.2], [0.8, 1, 0.5], [0.2,0.5,1]]),np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])]
    src_c1 = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
    
    
    # plot
    # plot(src_c0,src_c1,'Source')

    rotation=180 #degrees +ve: counter clockwise

    # target domain: GMM - class 1 (rotated source class 1)
    target_c1 = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
    target_c1 = rot_3d(rotation,target_c1)

    # target domain: GMM - class 0 (rotated target class 0)
    target_c0 = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)
    target_c0 = rot_3d(rotation,target_c0)
    # plot
    # plot(target_c0,target_c1,'Target')

    #combined plot of source domain and target domain
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(src_c0[:, 0], src_c0[:, 1], src_c0[:, 2], color = "r",label='src_class_0',marker='o')
    ax.scatter3D(src_c1[:, 0], src_c1[:, 1], src_c1[:, 2], color = "r",label='src_class_1',marker='x')
    ax.scatter3D(target_c0[:, 0], target_c0[:, 1], target_c0[:, 2], color = "g",label='target_class_0',marker='o')
    ax.scatter3D(target_c1[:, 0], target_c1[:, 1], target_c1[:, 2], color = "g",label='target_class_1',marker='x')
    plt.title('Gaussian Mixture Model Samples: Source|Target')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.legend()
    idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'_src_target_samp'
    plt.savefig(f_name+'.png')
    plt.show()

    #combined dataset - source
    df_src = np.concatenate((src_c0,src_c1),axis=0)
    #combined dataset - target
    df_target = np.concatenate((target_c0,target_c1),axis=0)

    #scaled

    # #no scaling
    # df_src_scaled = df_src
    # df_target_scaled = df_target

    # #Standardized
    # scaler_src = preprocessing.StandardScaler().fit(df_src)
    # df_src_scaled = scaler_src.transform(df_src)

    # scaler_target = preprocessing.StandardScaler().fit(df_target)
    # df_target_scaled = scaler_target.transform(df_target)

    #Normalized
    df_src_scaled = preprocessing.normalize(df_src,axis=0)

    df_target_scaled = preprocessing.normalize(df_target,axis=0)


    #PCA - source domain 
    df_src_pca,X_s = pca(df_src_scaled) #X_s: Dxd (column vector) 
    # plot
    # plot(df_src_pca[:n],df_src_pca[n:],'pca_Source')


    #PCA - target domain 
    df_target_pca,X_t = pca(df_target_scaled)
    # plot
    # plot(df_target_pca[:n],df_target_pca[n:],'pca_Target')

    #combined plot - after scaling
    #combined plot of source domain and target domain
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(df_src_scaled[:n, 0], df_src_scaled[:n, 1], df_src_scaled[:n, 2], color = "r",label='src_class_0',marker='o')
    ax.scatter3D(df_src_scaled[n:, 0], df_src_scaled[n:, 1], df_src_scaled[n:, 2], color = "r",label='src_class_1',marker='x')
    ax.scatter3D(df_target_scaled[:n, 0], df_target_scaled[:n, 1], df_target_scaled[:n, 2], color = "g",label='target_class_0',marker='o')
    ax.scatter3D(df_target_scaled[n:, 0], df_target_scaled[n:, 1], df_target_scaled[n:, 2], color = "g",label='target_class_1',marker='x')
    plt.title('Gaussian Mixture Model Samples: Source|Target-Scaled')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.legend()
    # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'_src_target_scaled'
    plt.savefig(f_name+'.png')
    plt.show()


    #combined plot - after scaling and pca (3d to 2d)
    #combined plot of source domain and target domain
    fig = plt.figure()
    plt.scatter(df_src_pca[:n, 0], df_src_pca[:n, 1],color='r',label='src_class_0',marker='o')
    plt.scatter(df_src_pca[n:, 0], df_src_pca[n:, 1],color='r',label='src_class_1',marker='x')
    plt.scatter(df_target_pca[:n, 0], df_target_pca[:n, 1],color='g',label='target_class_0',marker='o')
    plt.scatter(df_target_pca[n:, 0], df_target_pca[n:, 1],color='g',label='target_class_1',marker='x')
    plt.title('Gaussian Mixture Model Samples: Source|Target - Scaled_PCA')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    idx=datetime.now()
    # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'_src_target_scaled_pca'
    plt.savefig(f_name+'.png')
    plt.show()




    
    #align the source space with the target space
    df_src_aligned = df_src_scaled@(X_s@X_s.T@X_t) #according to paper
    # df_src_aligned = df_src_scaled@(X_s.T@X_t@X_s) 
    
    df_target_projected = df_target_scaled@X_t

    #combined plot - after sub-space alignment
    #combined plot of source domain and target domain
    fig = plt.figure()
    plt.scatter(df_src_aligned[:n, 0], df_src_aligned[:n, 1],color='r',label='src_class_0',marker='o')
    plt.scatter(df_src_aligned[n:, 0], df_src_aligned[n:, 1],color='r',label='src_class_1',marker='x')
    plt.scatter(df_target_projected[:n, 0], df_target_projected[:n, 1],color='g',label='target_class_0',marker='o')
    plt.scatter(df_target_projected[n:, 0], df_target_projected[n:, 1],color='g',label='target_class_1',marker='x')
    plt.title('Gaussian Mixture Model Samples: Source|Target - Aligned')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    idx=datetime.now()
    # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'_src_target_aligned'
    plt.savefig(f_name+'.png')
    plt.show()

