import numpy as np
import cv2
import sys
from facewarp.gen_puppet_utils import *

from face_of_art.menpo_functions import *
from face_of_art.deep_heatmaps_model_fusion_net import DeepHeatmapsModel


# Hide warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LandmarkDetector():

    def __init__(self, imgpath, src_dir="ape_src", out_dir="ape_src", inputImage="ape3.jpg"):
        self.data_dir = src_dir
        self.out_dir = out_dir

        self.test_data = inputImage
        self.CH = self.test_data[:-4]
        self.use_gt_bb = False

        # Create necessary files for holding landmarks
        filename = ["_open_mouth.txt", "_close_mouth.txt",
                    "_open_mouth_norm.txt", "_scale_shift.txt"]

        for eachFile in filename:
            with open(os.path.join("./" + self.data_dir, self.CH + eachFile), 'w+') as temp_file:
                temp_file.write("")

    def detect(self):
        """ Performs a detection and stores landmarks in <image>.pts in ./ape_src directory """

        if(not os.path.exists(os.path.join(self.data_dir, self.CH + '.pts'))):

            model_path = 'models/deep_heatmaps-60000'  # model for estimation stage
            # models for correction stage
            pdm_path = 'models/face_of_art/pdm_clm_models/pdm_models/'
            # model for tuning stage
            clm_path = 'models/face_of_art/pdm_clm_models/clm_models/g_t_all'

            # if true use tuning stage on eyebrows+jaw, else use tuning stage on jaw only
            outline_tune = True
            # if True, landmark predictions will be mapped to match original
            map_landmarks_to_original_image = True
            # input image size. otherwise the predicted landmarks will match the cropped version (256x256) of the images

            # load images
            bb_dir = os.path.join(self.data_dir, 'Bounding_Boxes')
            bb_dictionary = load_bb_dictionary(
                bb_dir, mode='TEST', test_data=self.test_data)
            bb_type = 'init'

            img_list = load_menpo_image_list(
                img_dir=self.data_dir, test_data=self.test_data, train_crop_dir=self.data_dir, img_dir_ns=self.data_dir, bb_type=bb_type,
                bb_dictionary=bb_dictionary, mode='TEST', return_transform=map_landmarks_to_original_image)

            # load model
            heatmap_model = DeepHeatmapsModel(
                mode='TEST', img_path=self.data_dir, test_model_path=model_path, test_data=self.test_data, menpo_verbose=False)

            print("\npredicting landmarks for: " +
                  os.path.join(self.data_dir, self.test_data))
            print("\nsaving landmarks to: " + self.data_dir)

            for i, img in enumerate(img_list):
                if i == 0:
                    reuse = None
                else:
                    reuse = True

                preds = heatmap_model.get_landmark_predictions(
                    img_list=[img], pdm_models_dir=pdm_path, clm_model_path=clm_path, reuse=reuse, map_to_input_size=map_landmarks_to_original_image)

                if map_landmarks_to_original_image:
                    img = img[0]

                if outline_tune:
                    pred_lms = preds['ECpTp_out']
                else:
                    pred_lms = preds['ECpTp_jaw']

                mio.export_landmark_file(PointCloud(pred_lms[0]), os.path.join(self.data_dir, img.path.stem + '.pts'),
                                         overwrite=True)

            print("\nFOA landmark detection DONE!")

    def edit(self):
        """ This function reads the <image>.pts file and opens a window to edit points. Press Q to exit. """
        pts0=[]

        if(os.path.exists(os.path.join(self.data_dir, self.CH + '_face_open_mouth.txt'))):
            pts0 = np.loadtxt(os.path.join(
                self.data_dir, self.CH + '_face_open_mouth.txt'))
            pts0 = pts0[:, 0:2]

        else:
            f = open(os.path.join(self.data_dir,
                     self.test_data[:-4] + '.pts'), 'r')
            lines = f.readlines()
            pts = []
            for i in range(3, 3+68):
                line = lines[i]
                line = line[:-1].split(' ')
                pts += [float(item) for item in line]

            pts0 = np.array(pts).reshape((68, 2))

        pts = np.copy(pts0) 
        img0 = cv2.imread(os.path.join(self.data_dir, self.test_data))
        img = np.copy(img0)
        node = -1

        def click_adjust_wireframe(event, x, y, flags, param):
            global img, pts, node

            def update_img(node, button_up=False):
                global img, pts

                # update carton points object and get fresh pts list
                pts[node, 0], pts[node, 1] = x, y

                img = np.copy(img0)
                draw_landmarks(img, pts)

                # zoom-in feature
                if (not button_up):
                    zoom_in_scale = 2
                    zoom_in_box_size = int(150 / zoom_in_scale)
                    zoom_in_range = int(np.min([zoom_in_box_size, x, y,
                                                (img.shape[0] - y) /
                                                2 / zoom_in_scale,
                                                (img.shape[1] - x) / 2 / zoom_in_scale]))

                    img_zoom_in = img[y - zoom_in_range:y + zoom_in_range,
                                      x - zoom_in_range:x + zoom_in_range].copy()
                    img_zoom_in = cv2.resize(img_zoom_in, (0, 0), fx=zoom_in_scale,
                                             fy=zoom_in_scale)
                    cv2.drawMarker(img_zoom_in, (zoom_in_range * zoom_in_scale,
                                                 zoom_in_range * zoom_in_scale),
                                   (0, 0, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=30,
                                   thickness=2, line_type=cv2.LINE_AA)
                    height, width, depth = np.shape(img_zoom_in)

                    img[y:y + height, x:x + width] = img_zoom_in
                    cv2.rectangle(img, (x, y), (x + height, y + width),
                                  (0, 0, 255), thickness=2)

            if event == cv2.EVENT_LBUTTONDOWN:
                # search for nearest point
                node = closest_node((x, y), pts)
                if(node >= 0):
                    update_img(node)

            if event == cv2.EVENT_LBUTTONUP:
                node = closest_node((x, y), pts)
                if (node >= 0):
                    update_img(node, button_up=True)
                node = -1

            if event == cv2.EVENT_MOUSEMOVE:
                # redraw figure
                if (node != -1):
                    update_img(node)

        draw_landmarks(img, pts)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("img", click_adjust_wireframe)

        while(True):
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            
        cv2.destroyAllWindows()

        print('vis and refine landmark Done!')

        # Grab the readjusted marks
        pts = np.concatenate([pts, np.ones((68, 1))], axis=1)

        # Save them back in original file
        np.savetxt(os.path.join(
            self.data_dir, '{}_face_open_mouth.txt'.format(self.CH)), pts, fmt='%.4f')

    def find_closed_mouth_landmark(self):
        ''' =================================================================
                    find closed mouth landmark and normalize

        Input: param are used to change closed mouth strength
                param[0]: larger -> outer-upper lip higher
                param[1]: larger -> outer-lower lip higher
                param[2]: larger -> inner-upper lip higher
                param[3]: larger -> inner-lower lip higher

        Output: saved as CH_face_open_mouth_norm.txt
                        CH_scale_shift.txt
                        CH_face_close_mouth.txt

        Press Q or close the image window to continue.
        ================================================================= '''

        norm_anno(self.data_dir, self.CH, param=[
                  0.7, 0.4, 0.5, 0.5], show=True)




    def use_inner_lips_only(self, INNER_ONLY=False):
        """ =================================================================
                            delauney tri

        Input: INNER_ONLY indicates whether use the inner lip landmarks only

        Output: saved as CH_delauney_tri.txt

        Press any key to continue.
        ================================================================= """

        delauney_tri(self.data_dir, self.test_data, INNER_ONLY)

 
