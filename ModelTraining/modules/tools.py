import numpy as np
from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_fill_holes, label, binary_dilation, generate_binary_structure, binary_erosion
from skimage.transform import resize, rescale
from skimage import filters, morphology
from skimage.measure import label as sklabel
import nibabel as nib

# Select the largest connected component in a mask (brain)
def brain_component(vol):
    label_im, nb_labels = label(vol)
    label_count = np.bincount(label_im.ravel().astype(int))
    label_count[0] = 0
    return label_im == label_count.argmax()

# Get the largest connected component in a segmentation mask
def getLargestCC(segmentation):
    labels = sklabel(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC
    
# Compute a brain mask using the Otsu thresholding method
def maskcompute(b0, b1000):
    # Computes a brain mask using otsu method
    b0_mask, mask0 = median_otsu(b0, 1, 1)
    b1000_mask, mask1000 = median_otsu(b1000, 1, 1)
    # Apply morphological operations and combine masks to create a final brain mask
    mask = binary_fill_holes(morphology.binary_dilation(brain_component(mask0 & mask1000)))
    if b0.min() >= 0: mask = mask & (b0 >= 1)
    if b1000.min() >= 0: mask = mask & (b1000 >= 1)
    return mask
    
# Perform quick normalisation of the input volume based on the given mask
def quicknormalisation(volume, mask):
    mean = np.mean(volume[mask > 0])
    sd = np.std(volume[mask > 0])
    return (volume-mean)/sd
    
# Rescale and pad/crop volume to the target dimensions
def autorescale(volume, zeropadding=True, force_scaling=None, targetX=256, targetY=256):
    if force_scaling is not None:
        scaling = force_scaling
    else:
        scaling = 1
        # Calculate scaling factor based on the dimensions
        if (volume.shape[0] < (targetX - 16) or volume.shape[1] < (targetY - 16)):
            scalingX = targetX/volume.shape[0]
            scalingY = targetY/volume.shape[1]
            scaling = min(scalingX,scalingY)
        if (volume.shape[0] > (targetX + 16) or volume.shape[1] > (targetY + 16)):
            scalingX = targetX/volume.shape[0]
            scalingY = targetY/volume.shape[1]
            scaling = max(scalingX,scalingY)
            
    # Rescale volume if scaling factor is not equal to 1
    if scaling != 1:
        volume = np.stack([rescale(volume[...,i], scaling) for i in range(volume.shape[2])], axis=2)
        
    orig_shape = volume.shape
    
    # Apply padding if dimensions are smaller than target dimensions
    padx1 = padx2 = pady1 = pady2 = 0
    if orig_shape[0] < targetX or orig_shape[1] < targetY:
        if orig_shape[0] < targetX:
            padx1 = int((targetX - orig_shape[0])/2)
            padx2 = int(targetX - orig_shape[0] - padx1)
        if orig_shape[1] < targetY:
            pady1 = int((targetY - orig_shape[1])/2)
            pady2 = int(targetY - orig_shape[1] - pady1)
        volume = np.pad(volume, ((padx1, padx2),(pady1,pady2),(0,0)), mode=("constant" if zeropadding else "edge"))
        
    # Apply cropping if dimensions are larger than target dimensions
    cutx1 = cutx2 = cuty1 = cuty2 = 0
    if orig_shape[0] > targetX or orig_shape[1] > targetY:
        if orig_shape[0] > targetX:
            cutx1 = int((orig_shape[0] - targetX)/2)
            cutx2 = int(orig_shape[0] - targetX - cutx1)
            volume = volume[cutx1:-cutx2,:,:]
        if orig_shape[1] > targetY:
            cuty1 = int((orig_shape[1] - targetY)/2)
            cuty2 = int(orig_shape[1] - targetY - cuty1)
            volume = volume[:,cuty1:-cuty2,:]
    return volume, scaling

# Load and rescale NIfTI images (b0, b1000, mask, ROI)
def nib2rescaled(b0path, b1000path, maskpath=None, roi1path=None, roi2path=None):    
    # Load and rescale b0 image
    nib_b0 = nib.load(b0path)
    raw_b0 = nib_b0.get_fdata().squeeze()
    rescaled_b0, _= autorescale(raw_b0)
    
    # Load and rescale b1000 image
    nib_b1000 = nib.load(b1000path)
    raw_b1000 = nib_b1000.get_fdata().squeeze()
    rescaled_b1000, scaling = autorescale(raw_b1000)

    zlen = nib_b1000.shape[2]
    volvox = np.prod(nib_b1000.header.get_zooms())/(1000*scaling*scaling)

    # Load mask if provided, else compute it
    if maskpath is not None:
        nib_mask = nib.load(maskpath)
        raw_mask = nib_mask.get_fdata().squeeze() > 0.5
        rescaled_mask, _ = autorescale(raw_mask)
    else:
        rescaled_mask = maskcompute(rescaled_b0,rescaled_b1000).astype(bool)

    # Perform image normalization
    norm_b0 = quicknormalisation(rescaled_b0,rescaled_mask)
    norm_b1000 = quicknormalisation(rescaled_b1000,rescaled_mask)
    norm_b0[norm_b0<-10] = -10
    norm_b0[norm_b0>10] = 10
    norm_b1000[norm_b1000<-10] = -10
    norm_b1000[norm_b1000>10] = 10

    # Load and rescale ROI if provided
    if roi1path is not None:
        nib_roi1 = nib.load(roi1path)
        raw_roi1 = nib_roi1.get_fdata().squeeze() > 0.5
        rescaled_roi1, _= autorescale(raw_roi1)
        rescaled_roi1[~rescaled_mask] = False
    else:
        rescaled_roi1 = None
        
    if roi2path is not None:
        nib_roi2 = nib.load(roi2path)
        raw_roi2 = nib_roi2.get_fdata().squeeze() > 0.5
        rescaled_roi2, _= autorescale(raw_roi2)
        rescaled_roi2[~rescaled_roi1] = False
    else:
        rescaled_roi2 = None

    return zlen, volvox, scaling, norm_b0, norm_b1000, rescaled_mask, rescaled_roi1, rescaled_roi2

def quick_zstack(b0n,b1000n,zselector=[-3, -2, -1, 0, 1, 2, 3]):
    x = np.stack([b1000n.squeeze().transpose((2, 0, 1))[:,np.newaxis], 
                  b0n.squeeze().transpose((2, 0, 1))[:,np.newaxis]], axis=1)
    paddingval = np.min(x)
    paddingval = np.tile(paddingval, (2, 256, 256))[np.newaxis,:,np.newaxis]
    all_stacked = []
    for index in range(x.shape[0]):
        stacked = []
        for slindx in zselector:
            if index+slindx < 0 or index+slindx >= x.shape[0]:
                stacked.append(paddingval)
            else:
                stacked.append(x[index+slindx][np.newaxis])
        all_stacked.append(np.concatenate(stacked, axis=2))
    return np.concatenate(all_stacked, axis=0).astype(np.float32)

# Compute a color overlay NIfTI image from the given mask and visualization data
def compute_color_nib(b1000nib, maskdata, blobdata, vizdata, maxvizval=1, cut=0.5,
                     maskintensity=0.33, strokeintensity=0.66, vizintensity=0.66):
    b1000data = b1000nib.get_fdata()

    # Normalize b1000 data to [0, 256] range
    dwirange = b1000data.max() - b1000data.min()
    dwidata = np.array(256*(b1000data-b1000data.min())/dwirange).astype(float)
    
    # Create an empty RGB image initialized with the grayscale data
    rgb_data = np.zeros(maskdata.shape+(3,), dtype=float)
    rgb_data[...,0] = dwidata
    rgb_data[...,1] = dwidata
    rgb_data[...,2] = dwidata
    
    # Generate binary structure for outline
    struct = generate_binary_structure(2,2)
    
    # Create eroded versions of mask and blob data
    maskerode = (maskdata>cut)
    bloberode = (blobdata>cut)
    for i in range(maskdata.shape[2]):
        for j in range(2):
            maskerode[:,:,i] = binary_dilation(maskerode[:,:,i], struct)
            bloberode[:,:,i] = binary_dilation(bloberode[:,:,i], struct)
            
    # Generate outlines of mask and blob data
    maskdataoutline = (maskdata>cut) ^ maskerode
    blobdataoutline = (blobdata>cut) ^ bloberode
    
    # Update RGB values based on mask and blob outlines and visualization data
    rgb_data[maskdataoutline,1] = rgb_data[maskdataoutline,1]+(256-rgb_data[maskdataoutline,1])*maskintensity
    rgb_data[blobdataoutline,2] = rgb_data[blobdataoutline,2]+(256-rgb_data[blobdataoutline,2])*strokeintensity
    rgb_data[...,0] = rgb_data[...,0]+(256-rgb_data[...,0])*vizintensity*(vizdata/maxvizval)
    
    # Ensure RGB values are in valid range
    rgb_data[rgb_data<0] = 0
    rgb_data[rgb_data>256] = 256
    rgb_data = rgb_data.astype(np.uint8)
    
    # Create RGB dtype for NIfTI image
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    rgb_data_withtype = rgb_data.copy().view(dtype=rgb_dtype).reshape(rgb_data.shape[0:3])
    
    # Create NIfTI image with RGB data
    rgb_nib = nib.Nifti1Image(rgb_data_withtype, affine=b1000nib.affine)
    return rgb_nib