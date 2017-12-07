import tensorflow as tf
import math

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
		else:
			outNormalizedLayer = normalizedLayer
		return outNormalizedLayer
	
	def batchNormalizationLayer(self,inputToBlock,weightSize,epsilon = 1e-3,activation = "None",layerName):
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
		else:
			outNormalizedLayer = normalizedLayer
		return outNormalizedLayer

	
	def convultionLayer(self,inputToBlock,filterSize,noOfFilters,strideNumber,paddingType = "SAME",layerName):
		W = tf.get_variable(name='W',shape=[filterSize[0],filterSize[1],inputToBlock.shape[2],noOfFilters],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
		conv = tf.nn.conv2d(inputToBlock, W, strides = [1, strideNumber, strideNumber, 1], padding=paddingType)
		b = tf.get_variable('b', shape=[noOfFilters], tf.constant_initializer(0.0),dtype=tf.float32)
		preActivationLayer = tf.nn.bias_add(conv, b)
		convLayer = tf.nn.relu(preActivationLayer, name=layerName)
		return convLayer

class optimizerFunctions:
	
	def learningRateDecay(self,decayRate = 0.1,InitialAlpha = 1e-04,epochNum):
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
		allImagePathTensor = ops.convert_to_tensor(imageFilePathCollection, dtype=dtypes.string)
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
		sampleDataBatch, sampleLabelBatch = tf.train.batch([sampleDataQueue, sampleLabelQueue],batch_size=batchSize)#,num_threads=1)
		return sampleDataBatch, sampleLabelBatch
	
	def createPipelineForData(self,xInputDataList,yOutputDataList,testDataRatio):
		allInputDataTensor = ops.convert_to_tensor(xInputDataList, dtype=dtypes.string)
		allLabelsTensor = ops.convert_to_tensor(labelCollection, dtype=dtypes.int32)
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
