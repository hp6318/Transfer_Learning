# take 2d data - src|tar
# PCA - src|tar
# project points on eigen vector - 2 eigen vectors +/-ve for each SRC | TARGET  
# MSE : +/-ve compare each SRC | TARGET
# Change sign accordingly


import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.datasets import make_spd_matrix

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

def affine_transformation(rot_mat,df):
    df_rot=np.dot(df,rot_mat)
    return df_rot

def pca(samp):
    pca = PCA(n_components=2)
    pca.fit(samp)
    new_df=pca.transform(samp)
    # new_df = np.array(new_df)
    return new_df,pca.components_.T,pca.explained_variance_

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


def mse(data,proj_points):
    error = ((data - proj_points)**2).mean(axis=None)
    return error


def sign_flip_check2(ori_data,pca_data,ev_mat):
    #center the original data
    #run for loop on ev_mat, and proj centered data onto 1 ev at a time +/-
    #proj pca_data onto only +ve inside the ev. 
    #compute mse
    #make a decision based on mse and invert/retatin the sign of each ev. 

    ev=ev_mat.T #make it row vector

    #center the ori_data
    mean_vec=np.mean(ori_data,axis=0)
    centered_data=ori_data-mean_vec

    #project red centered data on +ve & -ve ev (one at a time)
    for i in range(len(ev)):
        #positive case
        proj_x = np.dot(centered_data,ev[i])
        proj_x_points_pos = np.expand_dims(proj_x,axis=-1)*np.expand_dims(ev[i],axis=0)
        
        #negative case
        proj_x = np.dot(centered_data,-ev[i])
        proj_x_points_neg = np.expand_dims(proj_x,axis=-1)*np.expand_dims(ev[i],axis=0)
        
        #project pca trasnformed data only on +ve (one at a time)
        proj_x_pca = np.dot(pca_data,ev[i])
        proj_x_points_pca_pos = np.expand_dims(proj_x_pca,axis=-1)*np.expand_dims(ev[i],axis=0)

        #compute mse
        #mse-1
        mse_pos=mse(proj_x_points_pos,proj_x_points_pca_pos)
        #mse-2
        mse_neg=mse(proj_x_points_neg,proj_x_points_pca_pos)

        #make informed decision based on mse
        if mse_pos>mse_neg:
            #make correction
            ev[i]=-ev[i]
    return ev.T #make it again column vector


if __name__=='__main__':
    n=1000
    # source domain: GMM - class 0
    src_mus_c0 = [np.array([-2, 2]), np.array([-2,-2])]
    src_covs_c0 = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
    src_c0 = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)

    # source domain: GMM - class 1
    src_mus_c1 = [np.array([2, 2]), np.array([2, -2])]
    src_covs_c1 = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
    src_c1 = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
    
    # plot
    # plot(src_c0,src_c1,'Source')

    homography_mat=np.round(np.random.uniform(-1,1,(2,2)),decimals=5) #2x2 #src x T = target
    # homography_mat=np.round(make_spd_matrix(2),decimals=5)
    
    # check pd or nd?
    if np.all(homography_mat-homography_mat.T==0) and np.all(np.linalg.eigvals(homography_mat) > 0):
        print("Positive definite transformation matrix")
    elif np.all(homography_mat-homography_mat.T==0) and np.all(np.linalg.eigvals(homography_mat) < 0):
        print("Negative definite transformation matrix")
    else:
        print("Neither Positive nor Negative definite transformation matrix")

    # target domain: GMM - class 1 (rotated source class 1)
    target_c1_before = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
    target_c1 = affine_transformation(homography_mat,target_c1_before)

    # target domain: GMM - class 0 (rotated target class 0)
    target_c0_before = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)
    target_c0 = affine_transformation(homography_mat,target_c0_before)

    # combined dataset - source
    df_src = np.concatenate((src_c0,src_c1),axis=0)
    # combined dataset - target
    df_target = np.concatenate((target_c0,target_c1),axis=0)

    # save the dataset
    np.save("Src.npy",df_src)
    np.save("Target.npy",df_target)
 
    # pca on sampled data
    df_src_pca,X_s_ori,_ = pca(df_src) #X_s: Dxd (column vector)
    df_target_pca,X_t_ori,_ = pca(df_target) #X_t: Dxd (column vector)

    #combined plot of target before and after trasnformation
    fig = plt.figure()
    plt.scatter(target_c0_before[:, 0], target_c0_before[:, 1],color='r',label='Before_class_0',marker='o')
    plt.scatter(target_c1_before[:, 0], target_c1_before[:, 1],color='r',label='Before_class_1',marker='x')
    plt.scatter(target_c0[:, 0], target_c0[:, 1],color='g',label='After_class_0',marker='o')
    plt.scatter(target_c1[:, 0], target_c1[:, 1],color='g',label='After_class_1',marker='x')
    plt.title('Target: Before & after affine transformation')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'_target_samp'
    # plt.savefig(f_name+'.png')
    plt.axis('scaled')
    plt.show()



    #combined plot of source domain and target domain after trasnformation
    fig = plt.figure()
    plt.scatter(src_c0[:, 0], src_c0[:, 1],color='r',label='src_class_0',marker='o')
    plt.scatter(src_c1[:, 0], src_c1[:, 1],color='r',label='src_class_1',marker='x')
    plt.scatter(target_c0[:, 0], target_c0[:, 1],color='g',label='target_class_0',marker='o')
    plt.scatter(target_c1[:, 0], target_c1[:, 1],color='g',label='target_class_1',marker='x')
    plt.arrow(0, 0, X_s_ori[0][0], X_s_ori[0][1],color='black',label='EV_src_1',head_width=0.1)
    plt.arrow(0, 0, X_s_ori[1][0], X_s_ori[1][1],color='brown',label='EV_src_2',head_width=0.1)
    plt.arrow(0, 0, X_t_ori[0][0], X_t_ori[0][1],color='blue',label='EV_target_1',head_width=0.1)
    plt.arrow(0, 0, X_t_ori[1][0], X_t_ori[1][1],color='olive',label='EV_target_2',head_width=0.1)
    plt.title('Source|Target : Samples')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    # idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'_src_target_samp'
    # plt.savefig(f_name+'.png')
    plt.axis('scaled')
    plt.show()

    for i in range(3):
        if i==0:
            # no scaling
            df_src_scaled = df_src
            df_target_scaled = df_target
            title_var='no_scaling'
        elif i==1:
            #Standardized
            scaler_src = preprocessing.StandardScaler().fit(df_src)
            df_src_scaled = scaler_src.transform(df_src)

            scaler_target = preprocessing.StandardScaler().fit(df_target)
            df_target_scaled = scaler_target.transform(df_target)
            title_var='Standardized'
        else:
            # #Normalized
            df_src_scaled = preprocessing.normalize(df_src,axis=0)

            df_target_scaled = preprocessing.normalize(df_target,axis=0)
            title_var='Normalized'


        #PCA - source domain 
        df_src_pca,X_s,eval_src = pca(df_src_scaled) #X_s: Dxd (column vector) 
        # plot
        # plot(df_src_pca[:n],df_src_pca[n:],'pca_Source')


        #PCA - target domain 
        df_target_pca,X_t,eval_target = pca(df_target_scaled)
        # plot
        # plot(df_target_pca[:n],df_target_pca[n:],'pca_Target')

        print("Source eigen values sign_{}: {}".format(title_var,np.sign(eval_src)))
        print("Target eigen values sign_{}: {}".format(title_var,np.sign(eval_target)))

        #combined plot - after scaling
        #combined plot of source domain and target domain
        fig = plt.figure()
        plt.scatter(df_src_scaled[:n, 0], df_src_scaled[:n, 1],color='r',label='src_class_0',marker='o')
        plt.scatter(df_src_scaled[n:, 0], df_src_scaled[n:, 1],color='r',label='src_class_1',marker='x')
        plt.scatter(df_target_scaled[:n, 0], df_target_scaled[:n, 1],color='g',label='target_class_0',marker='o')
        plt.scatter(df_target_scaled[n:, 0], df_target_scaled[n:, 1],color='g',label='target_class_1',marker='x')
        plt.arrow(0, 0, X_s[0][0], X_s[0][1],color='black',label='EV_src_1',head_width=0.1)
        plt.arrow(0, 0, X_s[1][0], X_s[1][1],color='brown',label='EV_src_2',head_width=0.1)
        plt.arrow(0, 0, X_t[0][0], X_t[0][1],color='blue',label='EV_target_1',head_width=0.1)
        plt.arrow(0, 0, X_t[1][0], X_t[1][1],color='olive',label='EV_target_2',head_width=0.1)
        plt.title('Source|Target - {}'.format(title_var))
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.legend()
        idx=datetime.now()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+title_var+'_src_target_scaled'
        # plt.savefig(f_name+'.png')
        plt.axis('scaled')
        plt.show()


        #combined plot - after scaling and pca
        #combined plot of source domain and target domain
        fig = plt.figure()
        plt.scatter(df_src_pca[:n, 0], df_src_pca[:n, 1],color='r',label='src_class_0',marker='o')
        plt.scatter(df_src_pca[n:, 0], df_src_pca[n:, 1],color='r',label='src_class_1',marker='x')
        plt.scatter(df_target_pca[:n, 0], df_target_pca[:n, 1],color='g',label='target_class_0',marker='o')
        plt.scatter(df_target_pca[n:, 0], df_target_pca[n:, 1],color='g',label='target_class_1',marker='x')
        plt.arrow(0, 0, X_s[0][0], X_s[0][1],color='black',label='EV_src_1',head_width=0.1)
        plt.arrow(0, 0, X_s[1][0], X_s[1][1],color='brown',label='EV_src_2',head_width=0.1)
        plt.arrow(0, 0, X_t[0][0], X_t[0][1],color='blue',label='EV_target_1',head_width=0.1)
        plt.arrow(0, 0, X_t[1][0], X_t[1][1],color='olive',label='EV_target_2',head_width=0.1)
        plt.title('Source|Target - {}+PCA'.format(title_var))
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.legend()
        idx=datetime.now()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+title_var+'_src_target_scaled_pca'
        # plt.savefig(f_name+'.png')
        plt.axis('scaled')
        plt.show()

        # without sign flip correction
        df_src_pca = df_src_scaled@X_s
        df_target_pca = df_target_scaled@X_t

        #align the source space with the target space
        df_src_aligned = df_src_scaled@(X_s@X_s.T@X_t) #according to paper #src x T = Target  #X_s@X_s.T@X_t=T?
        # df_src_aligned = df_src_scaled@(X_s.T@X_t@X_s) 
        
        df_target_projected = df_target_scaled@X_t

        #combined plot - after sub-space alignment
        #combined plot of source domain and target domain
        fig = plt.figure()
        plt.scatter(df_src_aligned[:n, 0], df_src_aligned[:n, 1],color='r',label='src_class_0',marker='o')
        plt.scatter(df_src_aligned[n:, 0], df_src_aligned[n:, 1],color='r',label='src_class_1',marker='x')
        plt.scatter(df_target_projected[:n, 0], df_target_projected[:n, 1],color='g',label='target_class_0',marker='o')
        plt.scatter(df_target_projected[n:, 0], df_target_projected[n:, 1],color='g',label='target_class_1',marker='x')
        plt.arrow(0, 0, X_s[0][0], X_s[0][1],color='black',label='EV_src_1',head_width=0.1)
        plt.arrow(0, 0, X_s[1][0], X_s[1][1],color='brown',label='EV_src_2',head_width=0.1)
        plt.arrow(0, 0, X_t[0][0], X_t[0][1],color='blue',label='EV_target_1',head_width=0.1)
        plt.arrow(0, 0, X_t[1][0], X_t[1][1],color='olive',label='EV_target_2',head_width=0.1)
        plt.title('Source|Target - Aligned(w/o sign correction)')
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.legend()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+title_var+'_src_target_aligned_without_sign_flip'
        # plt.savefig(f_name+'.png')
        plt.axis('scaled')
        plt.show()




        # sign correction of eigen vectors
        X_s = sign_flip_check2(df_src_scaled,df_src_pca,X_s) #src
        X_t = sign_flip_check2(df_target_scaled,df_target_pca,X_t) #target

        df_src_pca = df_src_scaled@X_s
        df_target_pca = df_target_scaled@X_t

        #combined plot - after scaling and pca and sign correction
        #combined plot of source domain and target domain
        fig = plt.figure()
        plt.scatter(df_src_pca[:n, 0], df_src_pca[:n, 1],color='r',label='src_class_0',marker='o')
        plt.scatter(df_src_pca[n:, 0], df_src_pca[n:, 1],color='r',label='src_class_1',marker='x')
        plt.scatter(df_target_pca[:n, 0], df_target_pca[:n, 1],color='g',label='target_class_0',marker='o')
        plt.scatter(df_target_pca[n:, 0], df_target_pca[n:, 1],color='g',label='target_class_1',marker='x')
        plt.arrow(0, 0, X_s[0][0], X_s[0][1],color='black',label='EV_src_1',head_width=0.1)
        plt.arrow(0, 0, X_s[1][0], X_s[1][1],color='brown',label='EV_src_2',head_width=0.1)
        plt.arrow(0, 0, X_t[0][0], X_t[0][1],color='blue',label='EV_target_1',head_width=0.1)
        plt.arrow(0, 0, X_t[1][0], X_t[1][1],color='olive',label='EV_target_2',head_width=0.1)
        plt.title('Source|Target - Scaled_PCA+sign_correction')
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.legend()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+title_var+'_src_target_scaled_pca_signCorr'
        # plt.savefig(f_name+'.png')
        plt.axis('scaled')
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
        plt.arrow(0, 0, X_s[0][0], X_s[0][1],color='black',label='EV_src_1',head_width=0.1)
        plt.arrow(0, 0, X_s[1][0], X_s[1][1],color='brown',label='EV_src_2',head_width=0.1)
        plt.arrow(0, 0, X_t[0][0], X_t[0][1],color='blue',label='EV_target_1',head_width=0.1)
        plt.arrow(0, 0, X_t[1][0], X_t[1][1],color='olive',label='EV_target_2',head_width=0.1)
        plt.title('Source|Target - Aligned(with sign_correction)')
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.legend()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+title_var+'_src_target_aligned_sign_corr'
        # plt.savefig(f_name+'.png')
        plt.axis('scaled')
        plt.show()

