import cv2
import os
import pdb



class dataManagementFunctions:
	
	def convertThePathListForRNN(self,sortedList,inputSeriesLength,outputSeriesLength):
		inputDataList = []
		outputDataList = []
		indexInputList = []
		indexOutputList = []
		for i in range(0,len(sortedList)):
			currentInputList = []
			currentOutputList = []
			currentIndexInputList = []
			currentIndexOutputList = []
			try:
				for currentIndex in range(0,inputSeriesLength):
					currentInputList.append(sortedList[i+currentIndex])
					currentIndexInputList.append(i+currentIndex)
				for currentIndex in range(inputSeriesLength,inputSeriesLength+outputSeriesLength):
					currentOutputList.append(sortedList[i+currentIndex])
					currentIndexOutputList.append(i+currentIndex)
				inputDataList.append(currentInputList)
				outputDataList.append(currentOutputList)
				indexInputList.append(currentIndexInputList)
				indexOutputList.append(currentIndexOutputList)
			except:
				print("missing continous data at "+str(i+currentIndex))

		return inputDataList,outputDataList,indexInputList,indexOutputList



class commonFunctions:
	
	def getFrameNumbersFromImageFolder(self,imageFolderPath,imagePrefix):
		frameNumberList = []
		for imageName in os.listdir(imageFolderPath):
			if imageName.endswith(".png"):
				frameNumber = imageName.split(imagePrefix+"_")[-1].split(".")[0]
				frameNumberList.append(int(frameNumber))
		frameNumberList = sorted(frameNumberList)
		return frameNumberList

	def getImagePathListFromFrameNumberList(self,imageFolderPath,imagePrefix):
		imagePathList = []
		frameNumberList = self.getFrameNumbersFromImageFolder(imageFolderPath,imagePrefix)
		for i in range(0,len(frameNumberList)):
			currentFrameNumber = frameNumberList[i]
			currentImagePath = self.getImagePathFromDefnition(imageFolderPath,imagePrefix,currentFrameNumber)
			imagePathList.append(currentImagePath)
		return frameNumberList,imagePathList

	def getImagePathFromDefnition(self,imageFolderPath,imagePrefix,frameNumber):
		return os.path.join(imageFolderPath,imagePrefix+"_"+str(frameNumber).zfill(4)+".png")

	def getFrameHeightWidthFromImage(self,imagePath):
		img = cv2.imread(imagePath)
		#pdb.set_trace()
		height,width = len(img),len(img[0])
		return width,height

class videoFunctions:

	def displayTheVideo(self,videoFilePath,videoDisplayName = "NoName"):
		print("Playing Video in the Window "+str(videoDisplayName))
		print("Press 'q' to quit ")
		videoCap = cv2.VideoCapture(videoFilePath)
		while(videoCap.isOpened()):
			ok,frame = videoCap.read()
			if ok is False:
				break
			cv2.imshow(videoDisplayName,frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				print("Video stopped by user")
				break

	def extractFramesFromVideos(self,videoFilePath,outputImageFolder,imagePrefix,frameSkip = 5):
		cF = commonFunctions()
		if not os.path.exists(outputImageFolder):
			os.makedirs(outputImageFolder)
		print("Extracting frames from video "+str(videoFilePath) +" ...")
		videoCap = cv2.VideoCapture(videoFilePath)
		frameNum = 0
		while(videoCap.isOpened()):
			i = 0
			if(frameNum == 0):
				ok,frame = videoCap.read()
			else:
				while(i < frameSkip):
					ok,frame = videoCap.read()
					i = i + 1
			if ok is False:
				break
			#cv2.imwrite(os.path.join(outputImageFolder,imagePrefix+str(frameNum).zfill(4)+".png"),frame)
			cv2.imwrite(cF.getImagePathFromDefnition(outputImageFolder,imagePrefix,frameNum),frame)
			frameNum = frameNum + 1
		print("done..")
		
	def displayAndSaveImageSeriesAsVideo(self,imageCollectionFolder,imagePrefix,ISDISPLAYVIDEOALLOWED = True,ISSAVEVIDEOALLOWED = False):
		cF = commonFunctions()
		frameNumberList = cF.getFrameNumbersFromImageFolder(imageCollectionFolder,imagePrefix)
		for i in range(0,len(frameNumberList)):
			currentFrameNumer = frameNumberList[i]
			imagePath = cF.getImagePathFromDefnition(imageCollectionFolder,imagePrefix,currentFrameNumer)
			frame = cv2.imread(imagePath)
			if(ISDISPLAYVIDEOALLOWED == True):
				cv2.imshow(imagePrefix,frame)
				cv2.waitKey(1)
			if(ISSAVEVIDEOALLOWED == True):
				if(i == 0):
					videoSaveFolder = os.path.join(imageCollectionFolder,"SavedVideo")
					if not os.path.exists(videoSaveFolder):
						os.makedirs(videoSaveFolder)
					frameWidth,frameHeight = cF.getFrameHeightWidthFromImage(imagePath)
					fourcc = cv2.VideoWriter_fourcc(*'XVID')
					videoRec = cv2.VideoWriter(os.path.join(videoSaveFolder,imagePrefix+".avi"), fourcc, 20.0, (frameWidth,frameHeight))
				videoRec.write(frame)
		if(ISSAVEVIDEOALLOWED == True):
			videoRec.release()
			print("Video successfully saved in "+str(videoSaveFolder))




if __name__ == "__main__":
	

	videoFolderPath = "C:\\Datasets\\Video\\CAVIAR_Data\\"
	outputImageFolder = "C:\\ExtractFrames\\"

	FRAMESKIP = 1
	ISDISPLAYVIDEOALLOWED = False
	ISSAVEVIDEOALLOWED = True
	INPUTSERIESLENGTH = 4
	OUTPUTSERIESLENGTH = 1

	vF = videoFunctions()
	cF = commonFunctions()
	dMF = dataManagementFunctions()

	inputImagePathCollectionList = []
	outputImagePathCollectionList = []


	for currentVideoName in os.listdir(videoFolderPath):

		videoFilePath = os.path.join(videoFolderPath,currentVideoName)
		currentVideoName = currentVideoName.split(".")[0]
		videoFrameExtractFolder = os.path.join(outputImageFolder,currentVideoName)

		vF.displayTheVideo(videoFilePath,currentVideoName)
		vF.extractFramesFromVideos(videoFilePath,videoFrameExtractFolder,currentVideoName,frameSkip = FRAMESKIP)
		vF.displayAndSaveImageSeriesAsVideo(videoFrameExtractFolder,currentVideoName,ISDISPLAYVIDEOALLOWED = ISDISPLAYVIDEOALLOWED,ISSAVEVIDEOALLOWED = ISSAVEVIDEOALLOWED)
		frameNumberList,imagePathList = cF.getImagePathListFromFrameNumberList(videoFrameExtractFolder,currentVideoName)
		inputImagePathDataList,outputImagePathDataList,indexInputList,indexOutputList = dMF.convertThePathListForRNN(imagePathList,INPUTSERIESLENGTH,OUTPUTSERIESLENGTH)
		#inputFrameNumberDataList,outputFrameNumberDataList,_,_ = dMF.convertThePathListForRNN(frameNumberList,INPUTSERIESLENGTH,OUTPUTSERIESLENGTH)
		
		inputImagePathCollectionList = inputImagePathCollectionList + inputImagePathDataList
		outputImagePathCollectionList = outputImagePathCollectionList + outputImagePathDataList
		

		
