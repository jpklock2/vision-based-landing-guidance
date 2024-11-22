# trackit/datasets/SOT/datasets/MyDataset.py

import os
import numpy as np

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

class MyDataset_Seed(BaseSeed):
    def __init__(self, root_path: str = None, data_split=('train', 'test')):
        if root_path is None:
            # get the path from `consts.yaml` file
            root_path = self.get_path_from_config('MyDataset_PATH')
            root_path = os.path.join(os.getcwd(), root_path)
        super(MyDataset_Seed, self).__init__(
            'MyDataset', # dataset name
            root_path,   # dataset root path
            data_split,
            ('train', 'test'),
            1
        )

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        # Implement the dataset construction logic here
        
        # sequence_names = ['BIRK_01_500', 'BIRK_13_500', 'DAAG_05_500', 'DAAG_23_500']
        # sequence_names = ['DAAG_23_500']
        data_split = self.data_split[0]
        print(data_split)

        # if data_split == "train":
        #     sequence_names = ['BIRK_01_500',
        #                     'BIRK_13_500',
        #                     'DAAG_05_500',
        #                     'DAAG_23_500',
        #                     'DIAP_03_500',
        #                     'DIAP_21_500',
        #                     'KMSY_11_500',
        #                     'KMSY_20_500',
        #                     'KMSY_29_500',
        #                     'KMSY_2_500',
        #                     'LFMP_15_500',
        #                     'LFPO_02_500',
        #                     'LFPO_24_500',
        #                     'LFPO_25_500',
        #                     'LFQQ_08_500',
        #                     'LFQQ_26_500',
        #                     'LFST_05_500',
        #                     'LFST_23_500',
        #                     'LPPT_21_500',
        #                     'SRLI_14_500',
        #                     'SRLI_32_500',
        #                     'VABB_09_500',
        #                     'VABB_14_500',
        #                     'VABB_27_500',
        #                     'VABB_32_500']
            
        # else:
        #     sequence_names = [
        #                     'CYUL_06L_35_test',
        #                     'CYUL_24R_35_test',
        #                     'CYVR_08L_35_test',
        #                     'CYVR_26R_35_test',
        #                     'CYYZ_05_35_test',
        #                     'CYYZ_23_35_test',
        #                     'DAAS_27_35_test',
        #                     'DAAS_9_35_test',
        #                     'EDDV_09L_35_test',
        #                     'EDDV_27R_35_test',
        #                     'EHAM_18R_35_test',
        #                     'EHAM_36L_35_test',
        #                     'FMEP_15_35_test',
        #                     'FMEP_33_35_test',
        #                     'FTTJ_23_35_test',
        #                     'FTTJ_5_35_test',
        #                     'GCRR_03_35_test',
        #                     'GCRR_21_35_test',
        #                     'HTDA_23_35_test',
        #                     'HTDA_5_35_test',
        #                     'KIAH_08L_35_test',
        #                     'KIAH_26R_35_test',
        #                     'KJFK_22L_35_test',
        #                     'KJFK_4R_35_test',
        #                     'KMIA_9_35_test',
        #                     'LCPH_11_35_test',
        #                     'LCPH_29_35_test',
        #                     'LEMD_14L_35_test',
        #                     'LEMD_32R_35_test',
        #                     'LFRN_10_35_test',
        #                     'LFRN_28_35_test',
        #                     'LFRS_03_35_test',
        #                     'LFRS_21_35_test',
        #                     'LFSB_15_35_test',
        #                     'LFSB_33_35_test',
        #                     'LGSM_27_35_test',
        #                     'LGSM_9_35_test',
        #                     'LICJ_02_35_test',
        #                     'LICJ_20_35_test',
        #                     'LICJ_25_35_test',
        #                     'LIRN_24_35_test',
        #                     'LOWL_26_35_test',
        #                     'LOWL_8_35_test',
        #                     'LSZH_34_35_test',
        #                     'LTAI_18L_35_test',
        #                     'LTAI_36R_35_test',
        #                     'LWSK_16_35_test',
        #                     'MDSD_17_35_test',
        #                     'MDSD_35_35_test',
        #                     'OMAD_13_35_test',
        #                     'OMAD_31_35_test',
        #                     'RJAA_16L_35_test',
        #                     'RJAA_34R_35_test',
        #                     'RJTT_04_35_test',
        #                     'RJTT_22_35_test',
        #                     'RPMD_23_35_test',
        #                     'RPMD_5_35_test',
        #                     'SAEZ_17_35_test',
        #                     'SAEZ_35_35_test',
        #                     'SEQM_18_35_test',
        #                     'SEQM_36_35_test',
        #                     'VDPP_23_35_test',
        #                     'VDPP_5_35_test',
        #                     'VHHH_07L_35_test',
        #                     'VHHH_25R_35_test',
        #                     'VOTV_14_35_test',
        #                     'VOTV_32_35_test',
        #                     'VQPR_15_35_test',
        #                     'VQPR_33_35_test',
        #                     'WSSS_02L_35_test',
        #                     'WSSS_20R_35_test',
        #                     'YBBN_01_35_test',
        #                     'YBBN_19_35_test',
        #                     'YMLT_14R_35_test',
        #                     'YMLT_32L_35_test',
        #                     'ZBAA_01_35_test',
        #                     'ZBAA_18L_35_test',
        #                     'ZBAA_19_35_test',
        #                     'ZBAA_36R_35_test'
        #                     ]
        # sequence_names = ['DAAG_23_500']
        # sequence_names = ['CYUL_06L_35_test']
        sequence_names = ['CYUL_06L_35_empty']
        
        # Set the total number of sequences (Optional, for progress bar)
        constructor.set_total_number_of_sequences(len(sequence_names))
        
        # Set the bounding box format (Optional, 'XYXY' or 'XYWH', default for XYWH)
        constructor.set_bounding_box_format('XYXY')
        
        # get root_path
        root_path = self.root_path
        
        for sequence_name in sequence_names:
            '''
            The following is an example of the dataset structure:
            root_path
            ├── seq1
            │   ├── frames
            │   │   ├── 0001.jpg
            │   │   ├── 0002.jpg
            │   │   └── ...
            │   └── groundtruth.txt
            ├── seq2
            ...            
            '''
            with constructor.new_sequence() as sequence_constructor:
                sequence_constructor.set_name(sequence_name)
                
                sequence_path = os.path.join(root_path, sequence_name)
                # groundtruth.txt: the path of the bounding boxes file
                boxes_path = os.path.join(sequence_path, 'groundtruth.txt')
                frames_path = os.path.join(sequence_path, 'frames')
                
                # load bounding boxes using numpy
                boxes = np.loadtxt(boxes_path, delimiter=',')
                
                for frame_id, box in enumerate(boxes):
                    # frame_path: the path of the frame image, assuming the frame image is named as 0001.jpg, 0002.jpg, ...
                    frame_path = os.path.join(frames_path, f'{frame_id + 1:04d}.jpg')
                    
                    with sequence_constructor.new_frame() as frame_constructor:
                        # set the frame path and image size 
                        # image_size is optional (will be read from the image if not provided)
                        frame_constructor.set_path(frame_path)
                        # set the bounding box
                        # validity is optional (False for fully occluded or out-of-view or not annotated)
                        frame_constructor.set_bounding_box(box, validity=True)
