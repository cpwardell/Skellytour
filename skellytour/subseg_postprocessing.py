
# Import packages
import SimpleITK as sitk
import numpy as np

## Three inputs; 1.) raw subsegmentation 2.) postprocessed bone segmentation 3.) output filename
def subsegpostprocessing(fname, boneseg_fname, outpath):
    # get file id for creating output file
    #fileid = fname.split("/")[-1].split(".nii.gz")[0]

    # read in fitted labels
    fitted_labs = Nifti(fname)
    # read in bones labels
    bone_labs = Nifti(boneseg_fname)
    regions = np.copy(fitted_labs.nparray)
    bones = np.copy(bone_labs.nparray)

    # make sure voxels match
    regions = check_voxel_match(regions, bones)
    # make sure edges are all cortical
    regions = fix_cort_edges(regions, bones)

    # make sure borders aren't spongy
    border_spong = check_border_ones(regions)
    regions[border_spong] = 2

    regionsitk = fitted_labs.create_new_image(regions)
    fitted_labs.write_new_file(regionsitk,outpath)

def check_voxel_match(arr, bonearray):
    assert arr.shape == bonearray.shape
    arr[((bonearray == 0) | (bonearray == 38)) & (arr != 0)] = 0
    arr[((bonearray != 0) & (bonearray != 38)) & (arr == 0)] = 1
    return arr

def fix_cort_edges(regions, bonearray):
    bkgcheck = np.zeros(regions.shape)
    foridx = np.argwhere((bonearray != 0) & (bonearray != 38))
    for i in range(len(foridx)):
        bkgcheck[foridx[i, 0], foridx[i, 1], foridx[i, 2]] = check_background(
            regions, (foridx[i, 0], foridx[i, 1], foridx[i, 2]), sqsize=1)
    regions[bkgcheck == 1] = 2
    return regions

def check_background(arr, coord, sqsize=5):
    x = coord[0]
    y = coord[1]
    z = coord[2]

    xmin = x-sqsize if x-sqsize > 0 else 0
    xmax = x+(sqsize+1) if x+(sqsize+1) < arr.shape[0] else arr.shape[0]-1
    ymin = y-sqsize if y-sqsize > 0 else 0
    ymax = y+(sqsize+1) if y+(sqsize+1) < arr.shape[1] else arr.shape[1]-1
    zmin = z-sqsize if z-sqsize > 0 else 0
    zmax = z+(sqsize+1) if z+(sqsize+1) < arr.shape[2] else arr.shape[2]-1

    xrang = list(range(xmin, xmax, 1))
    yrang = list(range(ymin, ymax, 1))
    zrang = list(range(zmin, zmax, 1))

    coordcombs = [(x, y, z) for x in xrang for y in yrang for z in zrang]
    near_background = any([arr[c] == 0 for c in coordcombs])
    return near_background

def check_border_ones(arr):
    # Get the shape of the array
    shape = arr.shape

    # Create a boolean mask with the same shape as the array
    border_mask = np.zeros(shape, dtype=bool)

    # Set the borders to True in the mask
    border_mask[0, :, :] = True  # First row
    border_mask[-1, :, :] = True  # Last row
    border_mask[:, 0, :] = True  # First column
    border_mask[:, -1, :] = True  # Last column
    border_mask[:, :, 0] = True  # First depth
    border_mask[:, :, -1] = True  # Last depth

    border_mask[arr != 1] = False
    return border_mask

class Nifti(sitk.SimpleITK.Image):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname

        self.read()
        self.orient_RAS()
        self.define_dims()
        self.to_numpy_array()

    def read(self):
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.fname)
        self.image = reader.Execute()

    def orient_RAS(self):
        # Detect orientation of input image
        # This will be a 3 letter string of LR/PA/IS in some order,...
        # ... usually LPS or RAS
        # I think our whole body scans are LPS, but you can check
        self.inputorientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            self.image.GetDirection())
        # Reorient to RAS
        self.image = sitk.DICOMOrient(self.image,
                                      desiredCoordinateOrientation='RAS')

    def define_dims(self):
        # Find image dimensions and real size of volume in mm
        # Volume of a voxel in mm3 is vspacing[0]*vspacing[1]*vspacing[2]
        self.LR, self.PA, self.IS = 0, 1, 2  # RAS format labels if you ...
        # ... want to access objects e.g. vsizes[0]
        self.vsizes = self.image.GetSize()  # num of pixels in each direction
        self.vspacing = self.image.GetSpacing()  # actual size in mm

    def to_numpy_array(self):
        # Convert sitk image to numpy array
        # This is because the original image object is immutable, ...
        # ...so we need a deep copy of it to do anything with it
        # This inverts the axis order: RAS to SAR
        # The order is correct, but I can't say 100% that ...
        # ...I've got the direction correct
        # 1st is IS: values increase inferior to superior
        # 2nd is PA: values increase posterior to anterior
        # 3rd is LR: values increase left to right
        self.nparray = sitk.GetArrayFromImage(self.image)

    def add_labels(self, label_fname):
        self.label = Nifti(label_fname)
        self.nlabels = len(np.unique(self.label.nparray))

    def create_bone_image(self, bone_index, crop=True, save=True, outdir='.'):
        bone_image = np.copy(self.nparray)
        bone_image[self.label.nparray != bone_index] = -1024

        assert bone_image.shape == self.label.nparray.shape

        if crop:
            mins = np.zeros(3)
            maxs = np.zeros(3)
            for i in range(bone_image.ndim):
                mins[i] = min(np.where(self.label.nparray == bone_index)[i])
                maxs[i] = max(np.where(self.label.nparray == bone_index)[i])

            mins = mins.astype(int) - 20
            maxs = maxs.astype(int) + 20  # add a bit of buffer
            mins = np.where(mins < 0, 0, mins)
            for i in range(len(maxs)):
                if maxs[i] >= bone_image.shape[i]:
                    maxs[i] = bone_image.shape[i] - 1

            bone_image = bone_image[mins[0]:maxs[0],
                                    mins[1]:maxs[1], mins[2]:maxs[2]]

        new = self.create_new_image(bone_image)

        if save:
            self.write_new_file(new, outdir)

        return new

    def create_new_image(self, nparray):
        # Create a new image object from the numpy array, ...
        # ...set its metadata and orientation to
        # the same values as the original input data
        mimage = sitk.GetImageFromArray(nparray)
        mimage.SetSpacing(self.image.GetSpacing())
        mimage.SetOrigin(self.image.GetOrigin())
        mimage.SetDirection(self.image.GetDirection())
        mimage = sitk.DICOMOrient(mimage,
                                  desiredCoordinateOrientation=self.inputorientation)
        # logging.debug('New image orientation is {}'.format(mimage.GetOrientation()))
        return mimage

    def write_new_file(self, image, outdir):
        # Write a new NIfTI file
        sitk.WriteImage(image, outdir)