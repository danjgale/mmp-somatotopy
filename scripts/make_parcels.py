
import os
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib

pjoin = os.path.join


def _module_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _get_surfaces(hemi=None):
    resources = os.path.join(_module_dir(), '../resources')
    template = 'Q1-Q6_RelatedParcellation210.{}.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'
    if hemi is None:
        return (os.path.join(resources, template.format(h)) for h in ['L', 'R'])
    else:
        return os.path.join(resources, template.format(hemi))


def _add_medwall(data):
    medwall = 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
    medwall = nib.load(pjoin(_module_dir(), '../resources/', medwall))

    out = medwall.get_fdata().ravel()
    out[out == 1] = np.nan
    out[out == 0] = data
    return out


def load_mmp():
    mmp_parc = 'Q1-Q6_RelatedValidation210.' \
               'CorticalAreas_dil_Final_Final_Areas_Group_Colors' \
               '.32k_fs_LR.dlabel.nii'
    mmp_parc = pjoin(_module_dir(), '../resources/', mmp_parc)

    # MMP roi vertices
    img = nib.load(mmp_parc)
    darray = img.get_fdata().ravel()
    darray = _add_medwall(darray)
    
    rois = [8, 9, 51, 52, 53, 188, 189, 231, 232, 233]
    roi_vertices = np.where(np.isin(darray, rois), darray, 0)

    left_mmp = roi_vertices[:32492]
    right_mmp = roi_vertices[32492:]
    return left_mmp, right_mmp


def _get_flattened_labels(img):
    img = nib.load(img)
    x = np.asarray(img.agg_data())
    labels = x * np.arange(1, x.shape[0] + 1).reshape(x.shape[0], -1)
    return np.sum(labels, axis=0)


def border_to_label(hemi):

    if hemi not in ['L', 'R']:
        raise ValueError("hemi must be 'L' or 'R'")
    surf = _get_surfaces(hemi)
    
    border = f'Q1-Q6_RelatedParcellation210.{hemi}.SubAreas.32k_fs_LR.border'
    border = pjoin(_module_dir(), '../resources/', border)

    gii = pjoin(out_dir, f'SomatotopicAreas.{hemi}.32k_fs_LR.label.gii')

    cmd = f'wb_command -border-to-rois {surf} {border} {gii}'
    subprocess.run(cmd.split())
    return gii


def create_label_img(x, label_file):

    darray = nib.gifti.GiftiDataArray(x, intent='NIFTI_INTENT_LABEL',
                                      datatype='NIFTI_TYPE_INT32')

    label_data = pd.read_csv(label_file)
    names = label_data['Name'].values
    index = label_data['Index'].values
    labels = label_data.drop(['Name', 'Index'], axis=1)
    labels[['Red', 'Green', 'Blue']] /= 255
    labels = labels.values

    # add in background labels

    labeltable = nib.gifti.GiftiLabelTable()
    for i, l, n in zip(index.tolist(), labels.tolist(), names.tolist()):
        glabel = nib.gifti.GiftiLabel(key=int(i), red=l[0], green=l[1], blue=l[2])
        glabel.label = n
        labeltable.labels.append(glabel)

    return nib.GiftiImage(darrays=[darray], labeltable=labeltable)
    

def intersect_mmp(mmp_vertices, subareas):

    x = np.asarray(subareas.agg_data()).ravel()

    results = []
    init_ix = 1
    for i in np.unique(x)[1:]:
        mask = (x == i).astype(float)
        masked_rois = mask * mmp_vertices

        roi_ix = np.unique(masked_rois)[1:]
        n_rois = len(roi_ix)
        roi_map = dict(zip(roi_ix, np.arange(n_rois) + init_ix))

        relabeled = masked_rois.copy()
        for k, v in roi_map.items(): relabeled[relabeled == k] = v
        results.append(relabeled)
        init_ix += n_rois
    return np.sum(np.array(results), axis=0)


if __name__ == '__main__':
    

    out_dir = '../parcellations'
    os.makedirs(out_dir, exist_ok=True)

    subarea_labels = '../parcellations/SomatotopicAreas_labels.csv'

    list_ = []
    for mmp, h in zip(load_mmp(), ['L', 'R']):
        
        # create sub area label image
        gii = border_to_label(h)
        sub_areas = create_label_img(_get_flattened_labels(gii), 
                                     subarea_labels)
        sub_areas.to_filename(gii)

        # intersect sub area label image with HCP-MMP
        subarea_parc_labels = f'../parcellations/{h}_Parcellation_Labels.csv'
        sub_area_parc = create_label_img(intersect_mmp(mmp, sub_areas), 
                                         subarea_parc_labels)
        sub_area_parc.to_filename(f'../parcellations/SomatotopicParc.{h}.32k_fs_LR.label.gii')


    # make CIFTI version of sub area label image 

    # make CIFTI version of sub area parcellation