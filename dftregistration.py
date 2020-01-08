import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
import numpy.fft as fft


def argmax_2d(x):
    m_ids = np.argmax(x)
    dH, dW = np.unravel_index(m_ids, x.shape)
    # dH = dH if dH < x.shape[0] / 2 else dH - x.shape[0]
    # dW = dW if dW < x.shape[1] / 2 else dW - x.shape[1]
    return dH, dW


def dftregistration(buf1ft, buf2ft, usfac=1):
    # Compute error for no pixel shift
    if usfac == 0:
        CCmax = np.sum(np.sum(buf1ft * np.conj(buf2ft)))
        rfzero = np.sum(np.abs(np.power(buf1ft, 2)))
        rgzero = np.sum(np.abs(np.power(buf2ft, 2)))
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.atan2(np.imag(CCmax), np.real(CCmax))
        output = [error, diffphase]
    elif usfac == 1:
        [m, n] = buf1ft.shape
        CC = fft.ifft2(buf1ft * np.conj(buf2ft))
        rloc, cloc = argmax_2d(np.abs(CC))
        CCmax = CC[rloc, cloc]
        rfzero = np.sum(np.abs(buf1ft) ** 2) / (m * n)
        rgzero = np.sum(np.abs(buf2ft) ** 2) / (m * n)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        md2 = np.floor(m / 2)
        nd2 = np.floor(n / 2)
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        output = [error, diffphase, row_shift, col_shift]
    else:
        """
        First upsample by a factor of 2 to obtain initial estimate Embed Fourier data in a 2x larger array
        """
        [m, n] = buf1ft.shape
        mlarge = m * 2
        nlarge = n * 2
        CC = np.zeros((mlarge, nlarge), dtype=buf1ft.dtype)
        CC[m - m // 2: m + 1 + (m - 1) // 2, n - n // 2: n + 1 + (n - 1) // 2] = \
            fft.fftshift(buf1ft) * np.conj(fft.fftshift(buf2ft))

        """
        Compute crosscorrelation and locate the peak
        """
        CC = fft.ifft2(fft.ifftshift(CC))
        rloc, cloc = argmax_2d(np.abs(CC))
        CCmax = CC[rloc, cloc]

        """
        Obtain shift in original pixel grid from the position of the crosscorrelation peak
        """
        [m, n] = CC.shape
        md2 = m // 2
        nd2 = n // 2
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc
        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc
        row_shift = row_shift / 2
        col_shift = col_shift / 2
        # output = [row_shift, col_shift]

        if usfac > 2:
            """
            DFT computation Initial shift estimate in upsampled grid
            """
            row_shift = np.round(row_shift * usfac) / usfac
            col_shift = np.round(col_shift * usfac) / usfac
            dftshift = np.floor(np.ceil(usfac * 1.5) / 2)
            CC = np.conj(dftups(
                buf2ft * np.conj(buf1ft),
                np.ceil(usfac * 1.5),
                np.ceil(usfac * 1.5),
                usfac,
                dftshift - row_shift * usfac,
                dftshift - col_shift * usfac)) / (md2 * nd2 * usfac ** 2)
            rloc, cloc = argmax_2d(np.abs(CC))
            CCmax = CC[rloc, cloc]
            rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, usfac) / (md2 * nd2 * usfac ** 2)
            rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, usfac) / (md2 * nd2 * usfac ** 2)
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = row_shift + rloc / usfac
            col_shift = col_shift + cloc / usfac
        else:
            rg00 = np.sum(np.sum(buf1ft * np.conj(buf1ft))) / m / n
            rf00 = np.sum(np.sum(buf2ft * np.conj(buf2ft))) / m / n

        error = 1.0 - CCmax * np.conj(CCmax) / (rg00 * rf00)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        if md2 == 1:
            row_shift = 0
        if nd2 == 1:
            col_shift = 0
        output = [error, diffphase, row_shift, col_shift]

    if usfac > 0:
        [nr, nc] = buf2ft.shape
        # Nr = fft.ifftshift([-fix(nr / 2):ceil(nr / 2) - 1])
        Nr = fft.ifftshift(np.arange(-nr // 2, np.ceil(nr / 2)))
        # Nc = fft.ifftshift([-fix(nc / 2):ceil(nc / 2) - 1])
        Nc = fft.ifftshift(np.arange(-nc // 2, np.ceil(nc / 2)))
        # [Nc, Nr] = np.mgrid[1:Nc:Nc * 1j, 1:Nr:Nr * 1j]
        [Nc, Nr] = np.meshgrid(Nc, Nr)
        Greg = buf2ft * np.exp(1j * 2 * np.pi * (-row_shift * Nr / nr - col_shift * Nc / nc))
        Greg = Greg * np.exp(1j * diffphase);
    elif usfac == 0:
        Greg = buf2ft * np.exp(1j * diffphase)
    return output, Greg


def dftups(xin, nor=None, noc=None, usfac=1, roff=0, coff=0):
    [nr, nc] = xin.shape
    if noc is None:
        noc = nc
    if nor is None:
        nor = nr
    kernc = np.exp((-1j * 2 * np.pi / (nc * usfac)) *
                   ((fft.ifftshift(np.arange(nc)[:, None]) -
                     np.floor(nc / 2)) *
                    (np.arange(0, noc)[None] - coff)))
    kernr = np.exp((-1j * 2 * np.pi / (nr * usfac)) *
                   (np.arange(0, nor)[:, None] - roff) *
                   (fft.ifftshift(np.arange(0, nr)[None]) - np.floor(nr / 2)))
    # plt.figure(5)
    # plt.subplot(2, 1, 1)
    # plt.imshow(np.angle(kernc))
    # plt.subplot(2, 1, 2)
    # plt.imshow(np.angle(kernr))
    # plt.show()
    out = kernr @ xin @ kernc
    return out


if __name__ == '__main1__':
    import cv2

    dH, dW = 23.48574, 18.73837
    # img1 = cv2.imread("hina02.jpg").mean(-1)
    img1 = cv2.imread("pot.jpg").mean(-1)
    img2 = ndi.shift(img1, [dH, dW], prefilter=False, order=5)
    # img1 = ndi.shift(img1, [0, 0], prefilter=False, order=3)
    output, Greg = dftregistration(np.fft.fft2(img1), np.fft.fft2(img2), 100)
    # print(output[-2:])
    print(output)
    vmin, vmax = 0.0, 255.0
    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.imshow(img1, vmin=vmin, vmax=vmax)
    plt.subplot(2, 2, 2)
    plt.imshow(img2, vmin=vmin, vmax=vmax)
    plt.subplot(2, 2, 3)
    g = np.abs(np.fft.ifft2(Greg))
    plt.imshow(g, vmin=vmin, vmax=vmax)
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(g - img1), vmin=vmin, vmax=vmax)
    plt.show()

    plt.show()


if __name__ == '__main__':
