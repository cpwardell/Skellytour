import os
import SimpleITK as sitk
import numpy as np
import copy
import logging
from skimage import measure

def postprocessing(args):

    ## Define some variables for input/output files
    samplename=os.path.basename(args.i)[:-7]
    segmentation_filename=os.path.join(args.o,samplename+"_"+args.m+".nii.gz")
    postprocessed_filename=segmentation_filename[:-7]+"_postprocessed.nii.gz"

    ## Minimum island size to keep is 1000 mm3
    minisland=1000 

    ## Postprocessing depends on the model used for prediction
    ## Define lists of labels and what to do with them
    if args.m=="low":
        largestonly=[1,2,3,4,5,6,7,8,9,10,11,14,15,16]
        ribs=[12,13]
        keepall=[17]
    if args.m=="medium":
        largestonly=[1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
        ribs=[12,13]
        keepall=[38]
    if args.m=="high":
        largestonly=list(range(1,60))
        keepall=[60]

    ## Read in segmentation, reorientate to RAS and convert to np array
    reader=sitk.ImageFileReader()
    reader.SetFileName(segmentation_filename)
    image = reader.Execute()
    inputorientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    image = sitk.DICOMOrient(image, desiredCoordinateOrientation='RAS')
    narr=sitk.GetArrayFromImage(image)

    ## Get size in mm of each voxel dimension and calculate minimum number of voxels to keep an island
    vspacing = image.GetSpacing()
    voxelvolume=vspacing[0]*vspacing[1]*vspacing[2]
    minvoxels=int(np.round(minisland/voxelvolume))

    ## Create a backup copy that will not be affected by further operations
    narrbackup=copy.deepcopy(narr)

    ## Create the final segmentation file and set every value to 0
    finalnarr=copy.deepcopy(narr)
    finalnarr.fill(0)

    ## Iterate through all segmentation labels, excluding 0
    seglabels=np.unique(narr)
    seglabels=seglabels[seglabels!=0]
    #for i in range(np.max(narr)):
    for seglabel in seglabels:
        logging.info("Postprocessing segmentation label "+str(seglabel))

        ## Restore the segmentation file from the backup copy
        narr=copy.deepcopy(narrbackup)

        ## Purge any non-foreground/background values in the segmentation (this is essential for any segmentation with multiple labels)
        ## Background elements are any elements not identical to the foreground label
        belements = narr!=seglabel
        narr[belements]=0

        ## Calculate islands; this produces a numpy array with the same dimensions as the input
        all_labels = measure.label(narr)

        ## This is the number of islands (because island labels are incrementing integers)
        numberofislands=np.max(all_labels)

        ## Loop through island labels and get the size of each
        ## We store in a dict in case we need to do something more in depth later
        islandsizes=dict()
        for i in range(numberofislands):
            islandlabel=i+1
            islandsizes[islandlabel]=np.sum(all_labels==islandlabel)*voxelvolume

        ## Find the order of the islands by size
        sizeorder=sorted(islandsizes,key=islandsizes.get,reverse=True)

        ## Keep the largest island only
        if(seglabel in largestonly):
            finalnarr[all_labels==sizeorder[0]]=seglabel
            continue
        ## Keep all the islands
        if(seglabel in keepall):
            elements = narr==seglabel
            finalnarr[elements]=seglabel
            continue
        ## Keep up to the first 12 islands that are larger than 1000 mm3
        if(seglabel in ribs):
            ## Loop through islands in the correct order
            for i in range(numberofislands):
                islandlabel=i+1
                if (islandlabel<=12) and (islandsizes[sizeorder[i]] >= minisland):
                    finalnarr[all_labels==sizeorder[i]]=seglabel

    ## Write postprocessed segmentation
    finalimage = sitk.GetImageFromArray(finalnarr)
    finalimage.SetSpacing(image.GetSpacing())
    finalimage.SetOrigin(image.GetOrigin())
    finalimage.SetDirection(image.GetDirection())
    finalimage = sitk.DICOMOrient(finalimage, desiredCoordinateOrientation=inputorientation)
    sitk.WriteImage(finalimage,postprocessed_filename)



