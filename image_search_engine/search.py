# import the necessary packages
from colordescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-l", "--limit", required = True,
	help = "return our limited results")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
ap.add_argument("-s", "--save-path", required = True,
	help = "Path to the save path")
args = vars(ap.parse_args())
# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)
# perform the search
searcher = Searcher(args["index"],int(args["limit"]))
results = searcher.search(features)
# display the query
cv2.imshow("Query", query)
# loop over the results
for (score, resultID) in results:
	print(args["result_path"] + resultID)
	# load the result image and display it
	result = cv2.imread(args["result_path"] + resultID)
	cv2.imwrite(args["save_path"] + resultID, result)
# 	cv2.imshow("Result", result)
# 	cv2.waitKey(0)
    
# https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
# python search.py --index index.csv --limit 15 --query ./whiteBG/train_5026.jpg.png --result-path D:/A/Python/Datasets/CV/Segmentation/PersonSeg/background/ --save-path D:/A/Python/Datasets/CV/Segmentation/PersonSeg/whiteBG/