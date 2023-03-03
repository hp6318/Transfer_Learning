import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime

def normal(value):
    return 1/np.sqrt(2*np.pi)* np.exp(-value**2/2)

def gaussian(value, mean, variance):
    return 1/(np.sqrt(2*np.pi)*variance) * np.exp(-1/(2*variance**2)*(value-mean)**2)

def inv_sigmoid(value):
    return np.log(value/(1-value))

def plot_norm_plus_sample():
    x = np.linspace(-5,5,201)
    y = normal(x)

    y_s = np.random.uniform(-1000, 1000, size=(201, 100000))
    x_s = np.mean(y_s, axis=1)


    n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
                                alpha=0.7, rwidth=0.4, label='sigmoid sampled')

    y_s = np.random.uniform(0, 1, size=201)
    x_s = inv_sigmoid(y_s)

    n, bins, patches = plt.hist(x=x_s, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.4, label='uniform sample mean')

    y_s_e = np.random.uniform(0, 1, size=(201, 5))
    x_s_e = np.mean(inv_sigmoid(y_s_e), axis=1)

    n, bins, patches = plt.hist(x=x_s_e, bins='auto', color='g',
                                alpha=1, rwidth=0.6, label='uniform sigmoid sample mean')

    scale = np.max(n)*np.sqrt(2*np.pi)
    plt.plot(x, y* scale, label='normal distribution')
    plt.title('sampling normal distribution')
    plt.legend()
    plt.ylabel('p(x)')
    plt.xlabel('x')
    plt.show()

def multivariate_gaussian(values, mean, covariance):
    fac = 1/np.sqrt((2*np.pi)**len(covariance)*np.linalg.det(covariance))
    return np.diagonal(fac*np.exp((values-mean) @ np.linalg.inv(covariance) @ (values-mean).T))

def plot_multivariate_gaussian():
    mean = np.array([0,0])
    # covariance = np.array([[1, 2], [2, 1]])
    covariance = np.array([[1, 0.8], [0.8,1]])

    lambda_, gamma_ = np.linalg.eig(covariance)

    fig = plt.figure()
    # x1 = np.linspace(-2.5, 5, 100).reshape((-1, 1))
    # x2 = np.linspace(-15, 32, 100).reshape((-1, 1))
    # y = multivariate_gaussian(np.concatenate((x1, x2),1), mean, covariance).reshape((-1,1))
    # plt.contourf(x1, x2, y)

    y_s = np.random.uniform(0, 1, size=(1000, 3))
    x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, 2))
    x_s = (x_normal*lambda_) @ gamma_ + mean

    # n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
    #                             alpha=0.7, rwidth=0.85, label='sampling mean', density=True)
    scale = 1  # np.max(n) * np.sqrt(2 * np.pi)
    # plt.plot(x, y * scale, label='gaussian distribution')
    plt.scatter(x_s[:, 0], x_s[:, 1])
    plt.title(f'Multivariate Gaussian')
    # plt.legend()
    plt.ylabel('x2')
    plt.xlabel('x1')

    plt.show()
    # fig.savefig('../../Portfolio/shifted_gaussian.png')

def pca_2(samp,fname):
    pca = PCA(n_components=2)
    pca.fit(samp)
    new_df=pca.transform(samp)
    fig = plt.figure()
    new_df = np.array(new_df)
    plt.scatter(new_df[:, 0], new_df[:, 1])
    plt.title('Gaussian Mixture Model Samples - PCA')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.savefig(fname+'_pca'+'.png') #2dim_1class_time
    plt.show()

def plot_gaussian_mixture_model():
    mus = [np.array([-2, 2]), np.array([2, -2])]
    covs = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]

    pis = np.array([0.5, 0.5])
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
    assert np.isclose(acc_pis[-1], 1)

    n = 1000
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

    fig = plt.figure()
    samples = np.array(samples)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.title('Gaussian Mixture Model Samples')
    plt.ylabel('x2')
    plt.xlabel('x1')
    idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name='2d_1c_'+dt_string
    plt.savefig(f_name+'.png') #2dim_1class_time
    plt.show()

    #pca
    pca_2(samples,f_name)

def gmm(value, pis, means, variances):
    density = np.zeros(value.shape)
    for i, mean in enumerate(means):
        density += pis[i]/np.sqrt(2*np.pi)/variances[i] * np.exp(-1/(2*variances[i]**2)*(value-mean)**2)
    return density

def pca_1(samp,x_cal,y_cal):
    pca = PCA(n_components=1)
    pca.fit(samp)
    new_df=pca.transform(samp)
    fig = plt.figure()

    n, bins, patches = plt.hist(x=np.array(new_df), bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sampled gmm', density=True)
    plt.plot(x_cal, y_cal, label='gmm')
    plt.title('Sampled 1D gaussian mixture model - PCA')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.savefig('1d_1c_4_3_0_1_pca.png') #1dim_1class_mean1_var1_mean2_var2
    plt.show()


def plot_1D_gaussian_mixture_model():
    means = [4, 0]
    variances = [3,1]
    pis = [0.5, 0.5]
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]

    n = 1000
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

        selected_mu = means[k]
        selected_cov = variances[k]

        # sampling from normal distribution
        y_s = np.random.uniform(0, 1, size=(1, 3))
        x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, 1))
        # transforming into multivariate distribution
        x_multi = (x_normal) * selected_cov + selected_mu
        samples.append(x_multi.tolist()[0])

    x_cal = np.linspace(-10, 10, 200)
    y_cal = gmm(x_cal, pis, means, variances)

    fig = plt.figure()

    n, bins, patches = plt.hist(x=np.array(samples), bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sampled gmm', density=True)
    plt.plot(x_cal, y_cal, label='gmm')
    plt.title('Sampled 1D gaussian mixture model')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.savefig('1d_1c_4_3_0_1.png')  #1dim_1class_mean1_var1_mean2_var2
    plt.show()

    #pca
    pca_1(samples,x_cal,y_cal)
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # plot samples from multivariate gaussian
    # plot_multivariate_gaussian()
    # plot samples from gaussian mixture model
    plot_gaussian_mixture_model()
    # plot samples from 1D gaussian mixture mdodel
    # plot_1D_gaussian_mixture_model()