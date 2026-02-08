# from sift_config import *



# # def extract_color_sift_features(images):
# #     sift = cv2.SIFT_create()
# #     all_descriptors = []

# #     for img in images:
# #         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         channels = cv2.split(rgb)
# #         img_descriptors = []

# #         for channel in channels:
# #             _, des = sift.detectAndCompute(channel, None)
# #             if des is not None:
# #                 img_descriptors.append(des)

# #         if len(img_descriptors) > 0:
# #             img_descriptors = np.vstack(img_descriptors)
# #             all_descriptors.append(img_descriptors)
# #         else:
# #             all_descriptors.append(np.array([]))
    
# #     return all_descriptors 

# # def extract_color_sift_features(images, labels=None):
# #     sift = cv2.SIFT_create()
# #     all_descriptors = []
# #     valid_labels = []
# #     for idx, img in enumerate(images):
# #         img_descriptors = []
# #         for channel in cv2.split(img):
# #             _, des = sift.detectAndCompute(channel, None)
# #             if des is not None:
# #                 img_descriptors.append(des)
# #         if len(img_descriptors) > 0:
# #             all_descriptors.append(np.vstack(img_descriptors))
# #             if labels is not None:
# #                 valid_labels.append(labels[idx])
# #     if labels is not None:
# #         return all_descriptors, valid_labels
#     # return all_descriptors
# def extract_color_sift_features(images, labels=None):
#     sift = cv2.SIFT_create()
#     all_descriptors = []
#     valid_labels = []

#     for idx, img in enumerate(images):
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         channels = cv2.split(rgb)
#         img_descriptors = []

#         for channel in channels:
#             _, des = sift.detectAndCompute(channel, None)
#             if des is not None:
#                 img_descriptors.append(des)

#         if len(img_descriptors) > 0:
#             img_descriptors = np.vstack(img_descriptors)
#             all_descriptors.append(img_descriptors)
#             if labels is not None:
#                 valid_labels.append(labels[idx])
#         # 否则：丢弃图像和标签

#     return all_descriptors, valid_labels if labels is not None else all_descriptors



# def create_visual_vocabulary(descriptors_list, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, max_samples=MAX_SAMPLES):
#     all_descriptors = np.vstack([d for d in descriptors_list if d.size > 0])

#     if len(all_descriptors) > max_samples:
#         np.random.seed(RANDOM_STATE) 
#         all_descriptors = all_descriptors[np.random.choice(len(all_descriptors), max_samples, replace=False)]

#     kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
#     kmeans.fit(all_descriptors)
#     return kmeans

# def extract_bow_features(descriptors_list, kmeans):
#     n_clusters = kmeans.n_clusters
#     features = []

#     for descriptors in descriptors_list:
#         hist = np.zeros(n_clusters)
#         if descriptors.size > 0:
#             labels = kmeans.predict(descriptors)
            
#             hist = np.bincount(labels, minlength=n_clusters)

#             hist = hist.astype(np.float32)
#             hist /= hist.sum() + 1e-7
#         features.append(hist)

#     return np.array(features)      

from sift_config import *



def extract_color_sift_features(images):
    sift = cv2.SIFT_create()
    all_descriptors = []

    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        channels = cv2.split(rgb)
        img_descriptors = []

        for channel in channels:
            _, des = sift.detectAndCompute(channel, None)
            if des is not None:
                img_descriptors.append(des)

        if len(img_descriptors) > 0:
            img_descriptors = np.vstack(img_descriptors)
            all_descriptors.append(img_descriptors)
        else:
            all_descriptors.append(np.array([]))
    
    return all_descriptors 

def create_visual_vocabulary(descriptors_list, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, max_samples=MAX_SAMPLES):
    all_descriptors = np.vstack([d for d in descriptors_list if d.size > 0])

    if len(all_descriptors) > max_samples:
        np.random.seed(RANDOM_STATE) 
        all_descriptors = all_descriptors[np.random.choice(len(all_descriptors), max_samples, replace=False)]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    kmeans.fit(all_descriptors)
    return kmeans

def extract_bow_features(descriptors_list, kmeans):
    n_clusters = kmeans.n_clusters
    features = []

    for descriptors in descriptors_list:
        hist = np.zeros(n_clusters)
        if descriptors.size > 0:
            labels = kmeans.predict(descriptors)
            
            hist = np.bincount(labels, minlength=n_clusters)

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7
        features.append(hist)

    return np.array(features)      

