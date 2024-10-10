# reference https://github.com/hansenms/ismrmrd-paper/blob/6ac8bce40786349545e4747118b950c0ec6438b7/code/do_recon_python.py
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import ismrmrd
import ismrmrd.xsd
import h5py

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions

    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

def read_ismrmrd(fname):
    with ismrmrd.Dataset(fname, 'dataset', create_if_needed=False) as dset:
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = header.encoding[0]

        # Matrix size
        eNx = enc.encodedSpace.matrixSize.x
        eNy = enc.encodedSpace.matrixSize.y
        eNz = enc.encodedSpace.matrixSize.z
        rNx = enc.reconSpace.matrixSize.x
        rNy = enc.reconSpace.matrixSize.y
        rNz = enc.reconSpace.matrixSize.z

        # Number of Slices, Reps, Contrasts, etc.
        if enc.encodingLimits.slice != None:
            nslices = enc.encodingLimits.slice.maximum + 1
        else:
            nslices = 1

        if enc.encodingLimits.repetition != None:
            nreps = enc.encodingLimits.repetition.maximum + 1
        else:
            nreps = 1

        if enc.encodingLimits.contrast != None:
            ncontrasts = enc.encodingLimits.contrast.maximum + 1
        else:
            ncontrasts = 1

        # the number of coils is optional, so get it from the first acquisition
        acq = dset.read_acquisition(0)
        ncoils = acq.active_channels

        # Initialiaze a storage array
        all_data = np.zeros((nreps, ncontrasts, nslices, ncoils, eNz, eNy, eNx), dtype=np.complex64)

        # Loop through the rest of the acquisitions and stuff
        for acqnum in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(acqnum)

            # Ignore noise scans
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                continue

            # Stuff into the buffer
            rep = acq.idx.repetition
            contrast = acq.idx.contrast
            slice = acq.idx.slice
            ky = acq.idx.kspace_encode_step_1
            kz = acq.idx.kspace_encode_step_2
            all_data[rep, contrast, slice, :, kz, ky, :] = acq.data

        # Reconstruct images
        images = np.zeros((nreps, ncontrasts, nslices, eNz, eNy, rNx), dtype=np.float32)
        for rep in range(nreps):
            for contrast in range(ncontrasts):
                for slice in range(nslices):
                    # FFT
                    if eNz>1:
                        #3D
                        im = transform_kspace_to_image(all_data[rep,contrast,slice,:,:,:,:], [1,2,3])
                    else:
                        #2D
                        im = transform_kspace_to_image(all_data[rep,contrast,slice,:,0,:,:], [1,2])

                    # Sum of squares
                    im = np.sqrt(np.sum(np.abs(im) ** 2, 0))
                
                    # Remove oversampling if needed
                    if eNx != rNx:
                        x0 = (eNx - rNx) / 2
                        x1 = (eNx - rNx) / 2 + rNx
                        x0 = int(x0)
                        x1 = int(x1)
                        if eNz>1:
                            #3D
                            im = im[:,:,x0:x1]
                        else:
                            #2D
                            im = im[:,x0:x1]
                
                    # Stuff into the output
                    if eNz>1:
                        #3D
                        images[rep,contrast,slice,:,:,:] = im
                    else:
                        #2D
                        images[rep,contrast,slice,0,:,:] = im

        img = np.squeeze(images)
        kspace = np.squeeze(all_data)
        xml = dset.read_xml_header()
        
    return kspace, img, xml
    

def export_fastmri(out_file_path, kspace, img, xml):
    # output to h5 file
    with h5py.File(out_file_path,'w') as outfile:
        outfile.create_dataset('kspace', data=kspace)
        outfile.create_dataset('reconstruction_rss', data=img)
        outfile.create_dataset('ismrmrd_header', data=xml)

if __name__ == "__main__":
    kspace, img, xml = read_ismrmrd("data/file_brain_AXFLAIR_209_6001372_ismrmrd.h5")
    img = np.transpose(img, (*range(img.ndim-2), -1, -2))  # (16, 320, 320) -> (16, 320, 320)
    kspace = np.transpose(kspace, (*range(kspace.ndim-2), -1, -2))  # (16, 12, 320, 640) ->(16, 12, 640, 320)

    export_fastmri('data/fastmri.h5', kspace, img, xml)
    
    # Validate output
    # from util import read_h5
    # read_kspace, read_img = read_h5('out/fastmri.h5')
    