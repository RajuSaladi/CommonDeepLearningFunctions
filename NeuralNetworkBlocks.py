import tensorflow as tf

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
		alpha = tf.Variable(tf.ones([weightSize[1]]))
		beta = tf.Variable(tf.zeros([weightSize[1]]))
		zNorm = (z - batchMean) / tf.sqrt(batchVariance + epsilon)
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
		W = tf.get_variable(name='W',shape=[inputToBlock.shape[1],filterSize[0],filterSize[1],noOfFilters],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
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


