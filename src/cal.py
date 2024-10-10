import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from .util import print_info

# Calculation
def softComplex(x, T):
    # out = (abs(x) > lambda) .* (x .* (abs(x) - lambda) ./ abs(x))
    absx = np.absolute(x)
    arg1 = (absx > T)
    arg2 = np.multiply(x, (absx - T))
    arg22 = np.divide(arg2, absx)
    out = np.multiply(arg1, arg22)

    return(out)

def ifft2c(x):
    out = np.sqrt(len(x.ravel())) * fftpack.ifftshift(fftpack.ifft2(fftpack.fftshift(x)))
    return(out)

def fft2c(x):
    out = (1 / np.sqrt(len(x.ravel()))) * fftpack.fftshift(fftpack.fft2(fftpack.ifftshift(x)))
    return(out)


def ista_CSmri(y, H, Ht, V, Vt, lam, alpha, Nit, fac, plot=False):
    y /= fac
    J = np.zeros(Nit)
    x = y
    T = lam / (2 * alpha)
    xs = []
    for k in range(Nit):
        Hx = H(x)
        Vx = V(x)
        # print(Hx.shape, Vx.shape)
        Fx = fft2c(x)
        Ft = ifft2c(y)
        # print(Fx.shape, Ft.shape)
        s = Fx.ravel() - y.ravel()
        ss = Hx.ravel() + Vx.ravel()
        J[k] = np.sum(np.abs(s ** 2)) + lam * np.sum(np.abs(ss))
        # print_info(J[k])
        x = softComplex((x + (Ft - (ifft2c(Fx) + lam * Ht(Hx) + lam * Vt(Vx))) / alpha), T)
        # print_info(x)
        xs.append(x)
        if plot and (k % 5 == 0 or k == Nit - 1):
            fig, [ax1,ax2] = plt.subplots(1,2, figsize=[16,8])
            ax1.imshow(abs(x), cmap='gray', vmin=0, vmax=1)
            ax2.plot(np.arange(k+1), J[:k+1], color = 'red', marker = 'o', ms=10)
            # ax2.plot(k, J[k], color = 'red', marker = 'o', ms=10)

            ax1.set_title('Reconstruction - Iteration {0}'.format(k))
            ax1.axis('off')

            ax2.set_title('Cost Function - Iteration {0}'.format(k))

            plt.pause(0.01)
            # plt.draw()		# please enjoy the real time plotting update
            # plt.savefig('reconstruction.png')
            plt.show()
            plt.close()

    return(x, J)

def conv2t(h, g):
    mh, nh = h.shape
    mg, ng = g.shape
    hff = np.fliplr(np.flipud(h))
    # print('error lmao sorry fam')
    # hf = np.fliplr(h)
    f = signal.convolve2d(g, hff[::-1])
    # print('error lmao sorry fam')
    f = f[mh:mg+1, nh:ng+1]
    # print(f.shape)

    return(f)