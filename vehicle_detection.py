import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import pickle
from collections import deque
import time
from keras.models import load_model
from lane_detection.lane_line_tracker import Lane_Line_Tracker


class Vehicle_Detection():
    def __init__(self, ystart=400, ystop=656, xstart=200, xstop=1280, scale=1.5,scales=[1.5], threshold_factor=5, heat_threshold=1, nb_frame_ave=5, window=64, cnn_predict=False):
        self.dist_pickle = pickle.load( open("model/dist_pickle.p", "rb" ) )
        self.svc = self.dist_pickle["clf"]
        self.X_scaler = self.dist_pickle["scaler"]
        self.orient = self.dist_pickle["orient"]
        self.pix_per_cell = self.dist_pickle["pix_per_cell"]
        self.pix_per_cell2 =  np.int(self.pix_per_cell/2)
        self.cell_per_block = self.dist_pickle["cell_per_block"]
        self.cell_per_block2 = self.cell_per_block*2
        self.spatial_size = self.dist_pickle["spatial_size"]
        self.hist_bins = self.dist_pickle["hist_bins"]
        self.ystart = ystart
        self.ystop = ystop
        self.ystart2 = ystart
        self.ystop2 = np.int(ystart + (ystop - ystart)/2)
        self.scale = scale
        self.threshold_factor = threshold_factor
        self.heatmaps = deque(maxlen=self.threshold_factor)
        self.heatmap = []
        self.nb_frame_ave = nb_frame_ave
        self.frames = deque(maxlen=self.threshold_factor)
        self.heat_threshold = heat_threshold
        self.window=window
        self.scales = scales
        self.cnn_model = load_model('./dl_detect/model.h5')
        self.cnn_predict = cnn_predict
        self.xstart = xstart
        self.xstop = xstop
        self.lane_line_tracker = Lane_Line_Tracker()

    def calibrate(self):
        # Define feature parameters
        self.color_space = 'YCrCb' # #ANY OTHER
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block =2
        self.hog_channel ='ALL'
        self.spatial_size = (16,16)
        self.hist_bins = 16
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True

        t=time.time()
        n_samples = 1000

        dirs = os.listdir("data/vehicles/")
        cars = []
        print(dirs)
        for image_type in dirs:
            cars.extend(glob.glob('data/vehicles/'+ image_type+'/*.jpg'))
            
        print('Number of Vehicles Images found', len(cars))

        with open('data/vehicles/cars.txt', 'w') as f:
            for fn in cars:
                f.write(fn+'\n')


        dirs = os.listdir("data/non-vehicles/")
        notcars = []
        print(dirs)
        for image_type in dirs:
            notcars.extend(glob.glob('data/non-vehicles/'+ image_type+'/*.jpg'))
            
        print('Number of Non-Vehicles Images found', len(notcars))

        with open('data/non-vehicles/notcars.txt', 'w') as f:
            for fn in notcars:
                f.write(fn+'\n')

        # Read in car / not-car image
        test_cars = cars#np.array(cars)[car_indxs]
        test_notcars = notcars#np.array(notcars)[notcar_indxs]

        car_features = self.extract_features(test_cars, color_space=self.color_space,
                                             spatial_size=self.spacial_size,hist_bins=self.hist_bins,
                                             orient=self.orient,pix_per_cell=self.pix_per_call, cell_per_block=self.cell_per_block, hog_channel=self.hog_channel,spatial_feat=self.spatial_feat, hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        notcar_features = self.extract_features(test_notcars, color_space=self.color_space,
                                                spatial_size=self.spacial_size,
                                                hist_bins=self.hist_bins,
                                                orient=self.orient,
                                                pix_per_cell=self.pix_per_call,
                                                cell_per_block=self.cell_per_block,
                                                hog_channel=self.hog_channel,
                                                spatial_feat=self.spatial_feat,
                                                hist_feat=self.hist_feat,
                                                hog_feat=self.hog_feat)
        print(time.time()-t, ' Seconds to compute features...')
        X = np.vstack((car_features, notcar_features)).astype(np.float)
        # Fit a per column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0,100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

        print('Using: ', self.orient, 'orientations,', self.pix_per_cell, 'pixels per cell', self.cell_per_block, 'cells per block,', self.hist_bins,
            'histogram bins, and', self.spacial_size, 'spatial sampling')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the learning time of the SVC
        t=time.time()
        self.svc.fit(X_train,y_train)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        self.dist_pickle = {}
        dist_pickle["clf"] = self.svc
        dist_pickle["scaler"] = self.X_scaler
        dist_pickle["color_space"] = self.color_space
        dist_pickle["orient"] = self.orient
        dist_pickle["pix_per_cell"] = self.pix_per_cell
        dist_pickle["cell_per_block"] = self.cell_per_block
        dist_pickle["spatial_size"] = self.spatial_size
        dist_pickle["hog_channel"] = self.hog_channel
        dist_pickle["hist_bins"] = self.hist_bins
        dist_pickle["spatial_feat"] = self.spatial_feat
        dist_pickle["hist_feat"] = self.hist_feat
        dist_pickle["hog_feat"] = self.hog_feat
        pickle.dump( self.dist_pickle, open( "model/dist_pickle.p", "wb" ) )

        print(round(time.time()-t, 2), ' Seconds to train SVC...')
        # Check the score of the SVC
        print('Test accuracy of svc = ', round(self.svc.score(X_test,y_test),4))

    # Define a function to compute color histogram features
    def color_hist(self, image, nbins=32):
        # Compute the histogram of the RGB channels separately
        # Concatenate the histograms into a single feature vector
        # Return the feature vector
        # Take histograms in R, G, and B
        rhist = np.histogram(image[:, :, 0], bins=32)
        ghist = np.histogram(image[:, :, 1], bins=32)
        bhist = np.histogram(image[:, :, 2], bins=32)
        # Generating bin centers
        bin_edges = rhist[1]
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    def data_look(self, car_list, notcar_list):
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(car_list)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(notcar_list)
        # Read in a test image, either car or notcar
        example_img = mpimg.imread(car_list[0])
        # Define a key "image_shape" and store the test image shape 3-tuple
        data_dict["image_shape"] = example_img.shape
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = example_img.dtype
        # Return data_dict
        return data_dict
    # Define a function to return HOG features and visualization

    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis == True:
            # Use skimage.hog() to get both features and a visualization
            features, hog_image = hog(img, orientations=orient,
                                      pixels_per_cell=(
                                          pix_per_cell, pix_per_cell),
                                      cells_per_block=(
                                          cell_per_block, cell_per_block),
                                      visualise=vis, feature_vector=feature_vec)

            return features, hog_image
        else:
            # Use skimage.hog() to get features only
            features = features, hog_image = hog(img, orientations=orient,
                                                 pixels_per_cell=(
                                                     pix_per_cell, pix_per_cell),
                                                 cells_per_block=(
                                                     cell_per_block, cell_per_block),
                                                 visualise=vis, feature_vector=feature_vec)

            return features
    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()

    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                         hist_bins=32, orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            if spatial_feat == True:
                spatial_features = self.bin_spatial(
                    feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat == True:
                # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                                                  orient, pix_per_cell, cell_per_block,
                                                                  vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                         pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(
                                          cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(
                               cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        if (color_space != 'RGB'):
            if (color_space == 'HSV'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif (color_space == 'LUV'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif (color_space == 'HLS'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif (color_space == 'YUV'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif (color_space == 'YCrCb'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            # feature_image = np.copy(img)
            feature_image = img.copy()
            
        # 3) Compute spatial features if flag is set
        if (spatial_feat == True):
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if (hist_feat == True):
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if (hog_feat == True):
            if (hog_channel == 'ALL'):
                hog_features = []
                hog_image = None
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel],
                                                              orient, pix_per_cell, cell_per_block,
                                                              vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                     pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)

    def single_img_features_train(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        if (color_space != 'RGB'):
            if (color_space == 'HSV'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif (color_space == 'LUV'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif (color_space == 'HLS'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif (color_space == 'YUV'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif (color_space == 'YCrCb'):
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            # feature_image = np.copy(img)
            feature_image = img.copy()
            
        # 3) Compute spatial features if flag is set
        if (spatial_feat == True):
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if (hist_feat == True):
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if (hog_feat == True):
            if (hog_channel == 'ALL'):
                hog_features = []
                hog_image = None
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel],
                                                              orient, pix_per_cell, cell_per_block,
                                                              vis=False, feature_vec=True))
            else:
                hog_features, hog_image = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                     pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features), hog_image

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows, clf, scaler, color_space='RGB',
                       spatial_size=(32, 32), hist_bins=32,
                       hist_range=(0, 256), orient=9,
                       pix_per_cell=8, cell_per_block=2,
                       hog_channel=0, spatial_feat=True,
                       hist_feat=True, hog_feat=True):

        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(
                img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img, color_space=color_space,
                                                spatial_size=spatial_size, hist_bins=hist_bins,
                                                orient=orient, pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                hist_feat=hist_feat, hog_feat=hog_feat)
            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = clf.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def convert_color(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, window = 64, subsample=(64, 64)):
        img_boxes = []
        count = 0

        draw_img = np.copy(img)
        #Make a heatmap of zeros
        heatmap = np.zeros_like(img[:,:,0])
        img2 = img
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        img_tosearch2 = img2[ystart:ystop, :, :]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(
                imshape[1] / scale), np.int(imshape[0] / scale)))
            img_tosearch2 = cv2.resize(img_tosearch2, (np.int(
                imshape[1] / scale), np.int(imshape[0] / scale)))
        
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, orient, pix_per_cell,
                                cell_per_block, feature_vec=False)
        hog2 = self.get_hog_features(ch2, orient, pix_per_cell,
                                cell_per_block, feature_vec=False)
        hog3 = self.get_hog_features(ch3, orient, pix_per_cell,
                                cell_per_block, feature_vec=False)
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], subsample)

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                # Get color features
                spatial_features = self.bin_spatial(subimg, size=spatial_size)
                hist_features = self.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                    (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                    img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart+1, xbox_left:xbox_left+win_draw+1]+=1

        return draw_img, heatmap
    def find_cars_nn(self, img, ystart, ystop, scale, pix_per_cell, cell_per_block, window = 64, subsample=(64, 64)):
        img_boxes = []
        count = 0
        draw_img = np.copy(img)
        #Make a heatmap of zeros
        heatmap = np.zeros_like(img[:,:,0])

        img_tosearch = img[ystart:ystop,self.xstart:self.xstop, :]

        imshape = img_tosearch.shape
        if scale != 1:
            img_tosearch = cv2.resize(img_tosearch, (np.int(
                imshape[1] / scale), np.int(imshape[0] / scale)))
        
        # Define blocks and steps as above
        nxblocks = (img_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (img_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        #window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    img_tosearch[ytop:ytop + window, xleft:xleft + window], subsample)
                count += 1

                test_prediction = self.cnn_model.predict(subimg[None, :, :, :], batch_size=1)
                if( test_prediction[0][0] > 0.5
                ):
                    test_prediction = 1
                else:
                    test_prediction = 0

                if (test_prediction == 1):
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left+self.xstart, ytop_draw + ystart),
                    (xbox_left + win_draw+self.xstop, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart+1, xbox_left+self.xstart:xbox_left+win_draw+self.xstart+1]+=1

        return draw_img, heatmap

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def visualise(self, fig, rows, cols, imgs, titles):
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            plt.title(i + 1)
            img_dims = len(img.shape)
            if(img_dims < 3):
                plt.imshow(img, cmap='hot')
                plt.title(titles[i])
            else:
                plt.imshow(img)
                plt.title(titles[i])

    def visualisex(self, fig, rows, cols, imgs, titles):
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            plt.title(i + 1)
            img_dims = len(img.shape)
            if(img_dims < 3):
                plt.imshow(img, cmap='gray')
                plt.title(titles[i])
            else:
                plt.imshow(img)
                plt.title(titles[i])

    def process_image(self,img):
        for scale in self.scales:
            if(self.cnn_predict):
                out_image, heatmap = self.find_cars_nn(img, self.ystart, self.ystop, scale, self.pix_per_cell, self.cell_per_block,)
            else:
                out_image, heatmap = self.find_cars(img, self.ystart, self.ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)
            self.heatmaps.append(heatmap)
        intergrated_heat_maps = np.sum(self.heatmaps, axis=0)
        threshold_heat_map = self.apply_threshold(intergrated_heat_maps,self.heat_threshold)
        labels = label(threshold_heat_map)
        # Draw bounding boxes on a copy of the image
        draw_image = self.draw_labeled_bboxes(np.copy(img), labels)
        draw_image = self.lane_line_tracker.process_image(draw_image)
        return draw_image