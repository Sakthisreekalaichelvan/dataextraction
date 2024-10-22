from PIL import os
import Image 
import numpy as np 
from sklearn.cluster import KMeans 
from skimage.feature import hog 
import shutil 
 
def extract_features_from_image(image):     
    if image.mode != "L":         
        image = image.convert("L")     
        image = image.resize((256, 256))     
        image_array = np.array(image) 
    hog_features = hog(image_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True) 
    return hog_features 
 
flowchart_dir = "C:/Users/Sakth/OneDrive/Pictures/flowchartImages" 
graphics_dir = "C:/Users/Sakth/OneDrive/Pictures/Engineering drawing" 
tables_dir = "C:/Users/Sakth/OneDrive/Pictures/tables diagram" 
 
def load_images_from_folder(folder):     
    images = []     
    for filename in os.listdir(folder): 
        img = Image.open(os.path.join(folder, filename))         
        if img is not None: 
             images.append(img) 
    return images 
 
def copytofolder(destination_folder):     
    image_path = 'C:/Users/Sakth/OneDrive/Documents/Training Data/Engineering drawing/egimage29.jpeg'     
    new_feature = extract_features([new_image])     
    shutil.copy(image_path, destination_folder) 
 
flowchart_images = load_images_from_folder(flowchart_dir) 
graphics_images = load_images_from_folder(graphics_dir) 
table_images = load_images_from_folder(tables_dir) 
images = flowchart_images + graphics_images + table_images 
 
 
 
def extract_features(images):     
    features = []     
    for image in images: 
        # Extract features from the image 
        feature = extract_features_from_image(image)         
        features.append(feature)     
        return features 

features = extract_features(images) 
num_clusters = 3  # Number of clusters (flowcharts, graphics, tables) 
kmeans = KMeans(n_clusters=num_clusters, random_state=42) 
clusters = kmeans.fit_predict(features) 
 
new_image = Image.open("C:/Users/Sakth/OneDrive/Documents/Training Data/Engineering drawing/egimage29.jpeg") 
new_feature = extract_features([new_image]) 
new_cluster = kmeans.predict(new_feature) 
 
class_labels = ["Flowchart", "Engineering Graphics", "Table"] 
predicted_class = class_labels[new_cluster[0]] 
print("Predicted Class:", predicted_class) 
if predicted_class== "Flowchart": 
    copytofolder("C:/Users/Sakth/OneDrive/Desktop/flowchart") 
elif predicted_class=="Engineering Graphics": 
    copytofolder("C:/Users/Sakth/OneDrive/Desktop/Engineering Graphics") 
else:
    copytofolder("C:/Users/Sakth/OneDrive/Desktop/tables") 









