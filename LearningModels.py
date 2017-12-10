import tensorflow as tf
from CommonTensorflowFunctions import NeuralNetworkBlocks




layerConfig = {"Conv":{"ConvType":CONVTYPE,"FilterSize":FILTERSIZE,"NoOfFilters":NOOFFILTERS,"Stride":STRIDENUMBER,"PaddingType":PADDINGTYPE},
				"Pool":{"PoolingType":POOLINGTYPE,"PoolSize":POOLSIZE,"Strides":POOLSTRIDENUMBER},
				"Activation":{"ActivationType":ACTIVATIONTYPE},
			}

class LearningModelsCollection:

	def getDefaultLayerConfiguartion(self):
		defaultLayerConfig = {"Conv":{"ConvType":"1D","FilterSize":3,"NoOfFilters":20,"Stride":1,"PaddingType":"SAME"},
				"Pool":{"PoolingType":"MAXPOOL","PoolSize":2,"Strides":1},
				"Activation":{"ActivationType":"LeakyRelu"},
			}
		return defaultLayerConfig
	
	def deepFinModel(self,inputX,outputY):
		
		
		NNBlock = NeuralNetworkBlocks()
		
		tf.reset_default_graph()
		inputX = tf.placeholder(tf.float32,[None,INPUTWINDOWSIZE,len(keyDataList)])
		outputY = tf.placeholder(tf.float32,[None,len(keyDataList)])
		
		
		layerConfig = self.getDefaultLayerConfiguartion()
		conv1_out = NNBlock.convultionBlock(inputX,layerConfig,layerName = "ConvBlock_1")
		normalizedOutput1 = NNBlock.batchNormalizationLayerTF(conv1_out,weightSize,epsilon = 1e-3,activation = "None")
		
		
		
		