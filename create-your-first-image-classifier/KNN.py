from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.preprocessing import SimplePreprocessor
from utils.datasets import SimpleDatasetLoader, simpledatasetloader
from imutils import paths
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-d",'--dataset',required=True,help='path to input dataset')
ap.add_argument("-k","--neighbors",type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j","--jobs", type=int,default=1,help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.reshape((data.shape[0],3072))

print(f"[INFO] features matrix: {data.nbytes / (1024*1000.0)} MB")



le = LabelEncoder()

labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier....")

model = KNeighborsClassifier(n_neighbors=args['neighbors'],
    n_jobs=args["jobs"])
model.fit(trainX, trainY)

print(classification_report(testY,model.predict(testX),target_names=le.classes_))
