Spencer Davis 
sdavis342
Unsupprovised learning and Dimensionality reduction Project 3 Machine learning CS7641

Code can be found here: https://github.com/davisspe33/unsuprovisedLearning

This is a Python3 Project 

Download Python3 and pip3 

To install all dependancies run the following command from the root of this directory
    - 'pip3 install -r requirements.txt'

Part 1: 
To run the algortithms on the unprocessed data run the follwoing command
    -  'python3 UnprocessedData/`selectedAlgoDATASET`.py'
    - where selectedAlgo is the algorithm you want to run and DATASET is the data you want to use either cancer or housing

Part 2: 
To run individual dimentionalirty reduction algos run the follwoing command
    - 'python3 processedData/runProcessed/processing/`dimetionalityAlgo`/`selectedAlgoDATASET`.py'
    - where dimetionalityAlgo is the algorithm you want to run
    - where selectedAlgoDATASET is the algorithm you want to run on the dataset you want to run either housing or cancer
    - uncomment the lines at the bottom of the file if you want to see the algorithms run indivudually and recomment them before runing parts 3-5

Part 3/4/5:
To run the algortithms on the BreastCancer data run the follwoing command
    - 'python3 processedData/runProcessed/ `selectedAlgoDATASET`.py'
    - where selectedAlgo is the algorithm you want to run and DATASET is the data you want to use either cancer or housing

To get the plots used in the project write up please uncomment the approperate parts in the main fuction of each file. 

All of the code either came from the sklearn docs(https://scikit-learn.org/stable/modules/classes.html) or was written by me, Spencer Davis
The datasets used in this project can be found in the 2 csv's. 