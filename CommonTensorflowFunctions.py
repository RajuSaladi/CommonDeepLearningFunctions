import tensorflow as tf
import math
import random


class NeuralNetworkBlocks:

	def convultionBlockTF(self,inputToBlock,filterSize,noOfFilters,strideNumber,paddingType,layerName):
		return tf.layers.conv2d(inputs=inputToBlock,filters=noOfFilters,kernel_size=filterSize,padding=paddingType,activation=tf.nn.relu,name = layerName)

	def maxPoolingLayerTF(self,inputToBlock,poolSize,strideNumber):
		return tf.layers.max_pooling2d(inputs=inputToBlock, pool_size=poolSize, strides=strideNumber,name = layerName)

	def dropoutLayerTF(self,inputToBlock,keepProbability,layerName):
		return tf.nn.dropout(inputs= inputToBlock,keep_prob = keepProbability,name = layerName)

	def batchNormalizationLayerTF(self,inputToBlock,weightSize,epsilon = 1e-3,activation = "None"):
		w_initial = np.random.normal(size=weightSize).astype(np.float32)
		w = tf.Variable(w_initial)
		z = tf.matmul(inputToBlock,w)
		batchMean, batchVariance = tf.nn.moments(z_BN,[0])
		alpha = tf.Variable(tf.ones([weightSize[1]]))
		beta = tf.Variable(tf.zeros([weightSize[1]]))
		normalizedLayer = tf.nn.batch_normalization(z,batchMean,batchVariance,beta,alpha,epsilon,name = "NormalizedLayer")
		if(activation == "None"):
			outNormalizedLayer = normalizedLayer
		elif(activation == "Sigmoid"):
			outNormalizedLayer = tf.nn.sigmoid(normalizedLayer)
		elif(activation == "Relu"):
			outNormalizedLayer = tf.nn.relu(normalizedLayer)
		elif(activation == "LeakyRelu"):
			outNormalizedLayer = tf.nn.leaky_relu(normalizedLayer, name = layerName)
		else:
			outNormalizedLayer = normalizedLayer
		return outNormalizedLayer
	
	def batchNormalizationLayer(self,inputToBlock,weightSize,layerName,epsilon = 1e-3,activation = "None"):
		w_initial = np.random.normal(size=weightSize).astype(np.float32)
		w = tf.Variable(w_initial)
		z = tf.matmul(inputToBlock,w)
		batchMean, batchVariance = tf.nn.moments(z,[0])
		zNorm = (z - batchMean) / tf.sqrt(batchVariance + epsilon)
		alpha = tf.Variable(tf.ones([weightSize[1]]))
		beta = tf.Variable(tf.zeros([weightSize[1]]))
		normalizedLayer = alpha * zNorm + beta
		if(activation == "None"):
			outNormalizedLayer = normalizedLayer
		elif(activation == "Sigmoid"):
			outNormalizedLayer = tf.nn.sigmoid(normalizedLayer,name = layerName)
		elif(activation == "Relu"):
			outNormalizedLayer = tf.nn.relu(normalizedLayer, name = layerName)
		elif(activation == "LeakyRelu"):
			outNormalizedLayer = tf.nn.leaky_relu(normalizedLayer, name = layerName)
		else:
			outNormalizedLayer = normalizedLayer
		return outNormalizedLayer

	def convultionLayer(self,inputToBlock,filterSize,noOfFilters,strideNumber,layerName,paddingType = "SAME"):
		W = tf.get_variable(name='W',shape=[filterSize[0],filterSize[1],inputToBlock.shape[2],noOfFilters],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
		conv = tf.nn.conv2d(inputToBlock, W, strides = [1, strideNumber, strideNumber, 1], padding=paddingType)
		b = tf.get_variable('b', shape=[noOfFilters], tf.constant_initializer(0.0),dtype=tf.float32)
		preActivationLayer = tf.nn.bias_add(conv, b)
		convLayer = tf.nn.relu(preActivationLayer, name=layerName)
		return convLayer

	def convultionBlock1D(self,inputToBlock,filterSize,noOfFilters,strideNumber,layerName,subLayerList = ["Conv":"1D","Pool":"Max","Activation":"None"],paddingType = "SAME",activation="None"):
		filterShape = [filterSize,1,noOfFilters]
		wFilt = tf.Variable(tf.truncated_normal(filterShape, stddev=0.03),name=layerName+'_wFilt')
		bias = tf.Variable(tf.truncated_normal([noOfFilters], stddev=0.03),name=layerName+'_bias')
		convLayer = tf.nn.conv1d(inputToBlock,wFilt,strides = [1, strideNumber, 1],padding=paddingType)
		preActivationLayer = tf.nn.bias_add(conv, b)
		if(activation == "None"):
			postActivationLayer = preActivationLayer
		elif(activation == "Sigmoid"):
			postActivationLayer = tf.nn.sigmoid(preActivationLayer,name = layerName)
		elif(activation == "Relu"):
			postActivationLayer = tf.nn.relu(preActivationLayer, name = layerName)
		elif(activation == "LeakyRelu"):
			postActivationLayer = tf.nn.leaky_relu(preActivationLayer, name = layerName)
		else:
			postActivationLayer = preActivationLayer
		return postActivationLayer

	def convultionBlock(self,inputToBlock,layerConfig,layerName):
		layerConfig = {"Conv":{"ConvType":CONVTYPE,"FilterSize":FILTERSIZE,"NoOfFilters":NOOFFILTERS,"Stride":STRIDENUMBER,"PaddingType":PADDINGTYPE},
						"Pool":{"PoolingType":POOLINGTYPE,"PoolSize":POOLSIZE,"Strides":POOLSTRIDENUMBER},
						"Activation":{"ActivationType":ACTIVATIONTYPE},
					}

		if(layerConfig["Conv"]["ConvType"] == "1D"):
			#Z = self.convultionLayer1D(inputToBlock,filterSize,noOfFilters,strideNumber,paddingType,layerName)
			Z = self.convultionLayer1D(inputToBlock,layerConfig["Conv"]["FilterSize"],layerConfig["Conv"]["NoOfFilters"],layerConfig["Conv"]["Stride"],layerConfig["Conv"]["PaddingType"],layerName)
			if "Pool" in layerConfig.keys():
				if(layerConfig["Pool"]["PoolingType"] is not None):
					Z = self.poolLayer1D(Z,layerConfig["Pool"]["PoolingType"],layerConfig["Pool"]["poolSize"],layerConfig["Pool"]["Strides"],layerName)
					#Z = self.poolLayer1D(Z,poolingType,poolSize,strideNumber,layerName)
			if "Activation" in layerConfig.keys():
				if (layerConfig["Activation"]["ActivationType"] is not None):
					Z = self.activationLayer(Z,layerConfig["Activation"]["ActivationType"],layerName)
					#Z = self.activationLayer(Z,activation,layerName)
		elif(layerConfig["Conv"]["ConvType"] == "2D"):

			#Z = self.convultionLayer2D(inputToBlock,filterSize,noOfFilters,strideNumber,paddingType,layerName)
			Z = self.convultionLayer2D(inputToBlock,layerConfig["Conv"]["FilterSize"],layerConfig["Conv"]["NoOfFilters"],layerConfig["Conv"]["Stride"],layerConfig["Conv"]["PaddingType"],layerName)
			if "Pool" in layerConfig.keys():
				if(layerConfig["Pool"]["PoolingType"] is not None):
					Z = self.poolLayer2D(Z,layerConfig["Pool"]["PoolingType"],layerConfig["Pool"]["poolSize"],layerConfig["Pool"]["Strides"],layerName)
					#Z = self.poolLayer2D(Z,poolingType,poolSize,strideNumber,layerName)
			if "Activation" in layerConfig.keys():
				if (layerConfig["Activation"]["ActivationType"] is not None):
					Z = self.activationLayer(Z,layerConfig["Activation"]["ActivationType"],layerName)
					#Z = self.activationLayer(Z,activation,layerName)
		else:
			print("Warninig: Required Configuration is not available.")
			Z = inputToBlock
		return Z

	def convultionLayer1D(self,inputToBlock,filterSize,noOfFilters,strideNumber,paddingType,layerName):
		filterShape = [filterSize,1,noOfFilters]
		wFilt = tf.Variable(tf.truncated_normal(filterShape, stddev=0.03),name=layerName+'_wFilt1D')
		bias = tf.Variable(tf.truncated_normal([noOfFilters], stddev=0.03),name=layerName+'_bias1D')
		convLayer = tf.nn.conv1d(inputToBlock,wFilt,strides = [1, strideNumber, 1],padding=paddingType,name=layerName+'_Conv1D')		
		preActivationLayer = tf.nn.bias_add(convLayer, bias)
		return preActivationLayer

	def poolLayer1D(self,inputToLayer,poolingType,poolSize,strideNumber,layerName):
		if(poolingType == "MAXPOOL"):
			pooledLayer = tf.nn.max_pool(inputToLayer,ksize = [1,poolSize,1],strides = [1,strideNumber,1],name = layerName+"_MaxPool1D")
			#tf.layers.max_pooling2d(inputs=inputToBlock, pool_size=poolSize, strides=strideNumber,name = layerName)
		else:
			pooledLayer = inputToLayer
		return pooledLayer

	def convultionLayer2D(self,inputToBlock,filterSize,noOfFilters,strideNumber,paddingType,layerName):
		noOfChannels = inputToBlock.shape[3]
		filterShape = [filterSize,filterSize,noOfChannels,noOfFilters]
		wFilt = tf.Variable(tf.truncated_normal(filterShape, stddev=0.03),name=layerName+'_wFilt')
		bias = tf.Variable(tf.truncated_normal([noOfFilters], stddev=0.03),name=layerName+'_bias')
		convLayer = tf.nn.conv2d(inputToBlock,wFilt,strides = [1, strideNumber,strideNumber, 1],padding=paddingType)		
		preActivationLayer = tf.nn.bias_add(convLayer, bias)
		return preActivationLayer

	def poolLayer2D(self,inputToLayer,poolingType,poolSize,strideNumber,layerName):
		if(poolingType == "MAXPOOL"):
			pooledLayer = tf.nn.max_pool(inputToLayer,ksize = [1,poolSize,poolSize,1],strides = [1,strideNumber,strideNumber,1],name = layerName+"_MaxPool2D")
			#tf.layers.max_pooling2d(inputs=inputToBlock, pool_size=poolSize, strides=strideNumber,name = layerName)
		else:
			pooledLayer = inputToLayer
		return pooledLayer

	def activationLayer(self,preActivationLayer,activation,layerName):
		if(activation == "None"):
			postActivationLayer = preActivationLayer
		elif(activation == "Sigmoid"):
			postActivationLayer = tf.nn.sigmoid(preActivationLayer,name = layerName+"_Sigmoid")
		elif(activation == "Relu"):
			postActivationLayer = tf.nn.relu(preActivationLayer, name = layerName+"_Relu")
		elif(activation == "LeakyRelu"):
			postActivationLayer = tf.nn.leaky_relu(preActivationLayer, name = layerName+"_LeaklyRelu")
		else:
			postActivationLayer = preActivationLayer
		return postActivationLayer

class optimizerFunctions:
	
	def learningRateDecay(self,epochNum,decayRate = 0.1,InitialAlpha = 1e-04):
		alpha = (1/(1+(decayRate*epochNum)))*InitialAlpha
		return alpha
	
	def gradeintDescent(self,X,dX,alpha = 1e-04):
		X = X - (alpha*dX)
		return X
	
	def calculateMomentum(self,vdX,dX,beta1):
		vdX = (beta1 * vdX) + ((1 - beta1)*dX)
		return vdX
	
	def gradientDescentWIthMomentum(self,X,dX,vdX,alpha = 1e-04,beta1 = 0.9):
		vdX = self.calculateMomentum(vdX,dX,beta1)
		X = X - (alpha*vdX)
		return X,vdX
	
	def calculateRMSProp(self,sdX,dX,beta2):
		sdX = (beta2*sdX) + ((1 - beta2)*(dX**2))
		return sdX
	
	def gradientDescentWithRMSProp(self,X,dX,sdX,alpha=1e-04,beta2=0.999,epsilon = 1e-08):
		sdX = self.calculateRMSProp(sdX,dX,beta2)
		X = X - (alpha*(sdX/((sdX + epsilon)**(1/2))))
		return X,sdX
	
	def gradientCorrection(self,vdX,beta1,iterationCount):
		return (vdX/(1 - (beta1*iterationCount)))

	def adamOptimizerGradientDescent(self,X,dX,vdX,sdX,iterationCount,alpha=1e-04,beta1 = 0.9,beta2=0.999,epsilon = 1e-08):
		vdX = self.calculateMomentum(X,dX,vdX,alpha,beta1)
		sdX = self.calculateRMSProp(X,dX,sdX,alpha,beta2,epsilon)
		vdXCorr = self.gradientCorrection(vdX,beta1,iterationCount)
		sdXCorr = self.gradientCorrection(sdX,beta2,iterationCount)
		X = X - (alpha*(vdXCorr/((sdXCorr + epsilon)**(1/2))))
		return X,vdXCorr,sdXCorr

class lossFunctions:
	
	def getMeanSquareErrorLoss(self,yActual,yPred):
		
		return tf.reduce_mean(tf.squared_difference(yActual, yPred))
	
	def getL1Loss(self,yActual,yPred):
		return tf.reduce_mean(tf.squared_difference(yActual, yPred))
	
	def getLogLoss(self,yActual,yPred):
		return (yActual*(math.log(yPred))) + ((1-yActual)*(math.log(1-yPred)))

		
		
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

class inputDataProcessing:

	#reference: http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

	def createPipelineImageData(self,imageFilePathCollection,labelCollection,testDataRatio):
		allImagePathTensor = ops.convert_to_tensor(imageFilePathCollection, dtype=dtypes.float32)
		allLabelsTensor = ops.convert_to_tensor(labelCollection, dtype=dtypes.int32)
		# create a partition vector
		partitions = [0] * len(imageFilePathCollection)
		partitions[:int(testDataRatio*len(imageFilePathCollection))] = [1] * int((testDataRatio*len(imageFilePathCollection)))
		random.shuffle(partitions)

		# partition our data into a test and train set according to our partition vector
		trainImagePathTensor, testImagePathTensor = tf.dynamic_partition(allImagePathTensor, partitions, 2)
		trainLabelDataTensor, testLabelDataTensor = tf.dynamic_partition(allLabelsTensor, partitions, 2)
		
		# create input queues
		train_input_queue = tf.train.slice_input_producer([trainImagePathTensor, trainLabelDataTensor],shuffle=False)
		test_input_queue = tf.train.slice_input_producer([testImagePathTensor, testLabelDataTensor],shuffle=False)

		# process path and string tensor into an image and a label
		trainFileContent = tf.read_file(train_input_queue[0])
		trainImageQueue = tf.image.decode_jpeg(trainFileContent, channels=NUM_CHANNELS)
		trainLabelQueue = train_input_queue[1]

		testFileContent = tf.read_file(test_input_queue[0])
		testImageQueue = tf.image.decode_jpeg(testFileContent, channels=NUM_CHANNELS)
		testLabelQueue = test_input_queue[1]
		
		# define tensor shape
		trainImageQueue.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
		testImageQueue.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
		
		return trainImageQueue,testImageQueue,trainLabelQueue,testLabelQueue
	
	def makePipelineBatchWise(self,trainDataQueue,trainLabelQueue,batchSize):
		sampleDataBatch, sampleLabelBatch = tf.train.batch([trainDataQueue, trainLabelQueue],batch_size=batchSize)#,num_threads=1)
		return sampleDataBatch, sampleLabelBatch
	
	def createPipelineForData(self,xInputDataList,yOutputDataList,testDataRatio,inputDataType = "float32",outputDataType = "float32"):
		allInputDataTensor = ops.convert_to_tensor(xInputDataList, dtype=self.getTheDType(inputDataType))
		allLabelsTensor = ops.convert_to_tensor(yOutputDataList, dtype=self.getTheDType(outputDataType))
		# create a partition vector
		partitions = [0] * len(xInputDataList)
		partitions[:int(testDataRatio*len(xInputDataList))] = [1] * int((testDataRatio*len(xInputDataList)))
		random.shuffle(partitions)

		# partition our data into a test and train set according to our partition vector
		trainInputDataTensor, testInputDataTensor = tf.dynamic_partition(allInputDataTensor, partitions, 2)
		trainLabelDataTensor, testLabelDataTensor = tf.dynamic_partition(allLabelsTensor, partitions, 2)
		
		# create input queues
		trainInputQueue = tf.train.slice_input_producer([trainInputDataTensor, trainLabelDataTensor],shuffle=False)
		testInputQueue = tf.train.slice_input_producer([testInputDataTensor, testLabelDataTensor],shuffle=False)
		
		trainInputDataQueue = trainInputQueue[0]
		trainLabelQueue = trainInputQueue[1]

		testInputDataQueue = testInputQueue[0]
		testLabelQueue = testInputQueue[1]

		return trainInputDataQueue,testInputDataQueue,trainLabelQueue,testLabelQueue
	
	def getTheDType(self,varType):
		if(varType == "float32"):
			outputType = dtypes.float32
		elif(varType == "int32"):
			outputType = dtypes.int32
		else:
			outputType = dtypes.string
		return outputType

	def normalizeBatch(self,inputDataBatchTensor):
		batchMean, batchVariance = tf.nn.moments(inputDataBatchTensor,[0])
		normalizedBatch = (inputDataBatchTensor - batchMean) / tf.sqrt(batchVariance + epsilon)
		return normalizedBatch

#class sessionRunners:

"""
	def runSessionWithPipelineExample(self):
		
		with tf.Session() as sess:
		
			if(ISINITIALIZEVARIABLES):
				# initialize the variables
				sess.run(tf.initialize_all_variables())
			  
			# initialize the queue threads to start to shovel data
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			print "from the train set:"
			for i in range(20):
				print sess.run(train_label_batch)

			print "from the test set:"
			for i in range(10):
				print sess.run(test_label_batch)

			# stop our queue threads and properly close the session
			coord.request_stop()
			coord.join(threads)
			sess.close()
"""


class RNNFunctions:

	"""	
	def lstmCellDefnition(self,sizeOfLSTM):
		return tf.contrib.rnn.BasicLSTMCell(sizeOfLSTM)

	def stackingOfLSTMCell(self,lstmCellDef,noOfLayers):
		return tf.contrib.rnn.MultiRNNCell([lstmCellDef for _ in range(noOfLayers)])

	def LSTMModelDefnition_(self,inputX,sizeOfLSTM,noOfLayers):
		lstmCellDef = self.lstmCellDefnition(sizeOfLSTM)
		stackedLSTMLayers = self.stackingOfLSTMCell(lstmCellDef,noOfLayers)
		finalOutput, finalState = tf.nn.dynamic_rnn(cell=stackedLSTMLayers,inputs=inputX,dtype=tf.float32)
		return finalOutput, finalState
	"""

	def LSTMCellsWithDropout(self,layersInfoDictList,isDropoutAllowed = 0):
		if (isDropoutAllowed):
			return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(currentLayerInfo['NoOfCells'],state_is_tuple=True),currentLayerInfo['KeepProbability']) for currentLayerInfo in layersInfoDictList]
		else:
			return [tf.nn.rnn_cell.BasicLSTMCell(currentLayerInfo['NoOfCells'],state_is_tuple=True) for currentLayerInfo in layersInfoDictList]

	def getLSTMLayerConfiguration(self,noOfLSTMCellsInOneLayer,noOfLayers):
		layerInfo = [{'NoOfCells':noOfLSTMCellsInOneLayer,'KeepProbability':1}]*noOfLayers
		return layerInfo

	def LSTMModelDefnition(self,inputX,noOfLSTMCellsInOneLayer,noOfLayers):
		rnnLayerInfo = self.getLSTMLayerConfiguration(noOfLSTMCellsInOneLayer,noOfLayers)
		stackedLSTMlayers = tf.nn.rnn_cell.MultiRNNCell(self.LSTMCellsWithDropout(rnnLayerInfo), state_is_tuple=True)
		#squeezedX = learn.ops.split_squeeze(1, noOfLSTMCellsInOneLayer, X)
		#squeezedX =  tf.unstack(inputX, num=noOfLSTMCellsInOneLayer, axis=1)
		output, layers = tf.nn.dynamic_rnn(stackedLSTMlayers, inputX, dtype=tf.float32)
		return output, layers

	"""
	X = tf.placeholder(tf.int32, [batch_size, num_steps])
	lstmCellDef = lstmCellDefnition(noOfInputs)
	stackedLSTMLayers = stackingOfLSTMCell(lstmCellDef,noOfLayers)
	final_outputs, final_state = tf.nn.dynamic_rnn(cell=stackedLSTMLayers,inputs=X,dtype=tf.float32)

	"""

class TensorflowModels:


	def RNNForeCastModel(self,currentBatchX,currentBatchY,inputSeriesLength,outputSeriesLength,noOfLSTMLayers,LEARNINGRATE):

		RNNFunc= RNNFunctions()
		cTF = CommonTensorflowFunctions()

		noOfLSTMCellsInOneLayer = inputSeriesLength
		#stateSize = noOfLSTMCellsInOneLayer

		#currentBatchX = tf.placeholder(tf.float32,[None,inputSeriesLength,1])
		#currentBatchY = tf.placeholder(tf.float32,[None,outputSeriesLength])

		with tf.name_scope('LSTMLayers'):
			outputFromLSTM, outputStateFromLSTM = RNNFunc.LSTMModelDefnition(currentBatchX,noOfLSTMCellsInOneLayer,noOfLSTMLayers)
		
		with tf.name_scope('WeightsandBias'):
			WeightMatrixForOutputLayer = tf.Variable(np.random.rand(noOfLSTMCellsInOneLayer, outputSeriesLength),dtype=tf.float32)
			biasForOutputLayer = tf.Variable(np.zeros(shape = (1,outputSeriesLength)),dtype=tf.float32)
			cTF.getSummariesForVariables(WeightMatrixForOutputLayer)
			cTF.getSummariesForVariables(biasForOutputLayer)

		with tf.name_scope('RegressionLayer'):
			outputRegressionLayer = tf.map_fn(lambda currentCellLSTMOutput: tf.matmul(currentCellLSTMOutput, WeightMatrixForOutputLayer) + biasForOutputLayer, outputFromLSTM)
			finalOutputLayer = tf.squeeze(tf.split(outputRegressionLayer,noOfLSTMCellsInOneLayer,axis=1)[-1],axis=1)
			tf.summary.histogram('histogramOutputLayer', outputRegressionLayer)


		with tf.name_scope('LossFunction'):
			lossValue = tf.reduce_mean(tf.square(currentBatchY - finalOutputLayer))
			tf.summary.scalar("MeanSquareLoss", lossValue)

		with tf.name_scope('OptimizationTraining'):
			optimizer = tf.train.AdamOptimizer(learning_rate = LEARNINGRATE).minimize(lossValue)

		mergedSummaryOp = tf.summary.merge_all()
		
		return mergedSummaryOp,optimizer,lossValue,finalOutputLayer
