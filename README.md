# deeplabv3_tflite_for_nnapi_delegate
DeepLab V3 TFLite models that be fully delegated to NNAPI.

When running `./benchmark_model --graph=frozen_inference_graph.tflite --use_nnapi=1`, where the `frozen_inference_graph.tflite` is from the [quantized model](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_8bit_2019_04_26.tar.gz) by Google. We can see that the graph is divided into 8 nodes, becasue `AVERAGE_POOL_2D`, `RESIZE_BILINEAR`, and `ARG_MAX` cannot be delegated.

```
Operator-wise Profiling Info for Regular Benchmark Runs:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	     TfLiteNnapiDelegate	            0.000	   11.665	   11.422	 21.939%	 21.939%	     0.000	        1	[MobilenetV2/expanded_conv_16/project/add_fold, aspp0/Relu]:71
	         AVERAGE_POOL_2D	           11.422	    0.075	    0.080	  0.153%	 22.092%	     0.000	        1	[AvgPool2D/AvgPool]:61
	     TfLiteNnapiDelegate	           11.503	    0.549	    0.654	  1.256%	 23.348%	     0.000	        1	[image_pooling/Relu]:72
	         RESIZE_BILINEAR	           12.157	    0.733	    0.731	  1.404%	 24.753%	     0.000	        1	[ResizeBilinear]:63
	     TfLiteNnapiDelegate	           12.888	    1.468	    1.386	  2.663%	 27.416%	     0.000	        1	[logits/semantic/BiasAdd]:73
	         RESIZE_BILINEAR	           14.275	    0.093	    0.095	  0.182%	 27.597%	     0.000	        1	[ResizeBilinear_1]:68
	         RESIZE_BILINEAR	           14.370	   22.826	   22.369	 42.965%	 70.562%	     0.000	        1	[ResizeBilinear_2]:69
	                 ARG_MAX	           36.740	   15.440	   15.326	 29.438%	100.000%	     0.000	        1	[ArgMax]:70
```

| model name| dataset | input size| output stride| note |
| --------- |:-------:| ---------:|-------------:|------|
|[deeplabv3_mnv2_pascal_513_os8_quant](pascal_voc_2012/deeplabv3_mnv2_pascal_513_os8_quant.tflite)| pascal_voc_2012 | 513x513 | 8 | |
|[deeplabv3_mnv2_pascal_513_os16_quant](pascal_voc_2012/deeplabv3_mnv2_pascal_513_os16_quant.tflite)| pascal_voc_2012 | 513x513 | 16 | |
|[deeplabv3_mnv2_pascal_257_os8_quant](pascal_voc_2012/deeplabv3_mnv2_pascal_257_os8_quant.tflite)| pascal_voc_2012 | 257x257 | 8 | |
|[deeplabv3_mnv2_pascal_257_os16_quant](pascal_voc_2012/deeplabv3_mnv2_pascal_257_os16_quant.tflite)| pascal_voc_2012 | 257x257 | 16 | |
|[deeplabv3_mnv2_cityscapes_513_os8_quant](cityscapes/deeplabv3_mnv2_cityscapes_513_os8_dummy_quant.tflite)| cityscapes | 513x513 | 8 | dummy quant |
|[deeplabv3_mnv2_cityscapes_513_os16_quant](cityscapes/deeplabv3_mnv2_cityscapes_513_os16_dummy_quant.tflite)| cityscapes | 513x513 | 16 | dummy quant |
|[deeplabv3_mnv2_ade20k_513_os8_quant](ade20k/deeplabv3_mnv2_ade20k_513_os8_dummy_quant.tflite)| ade20k | 513x513 | 8 | dummy quant |
|[deeplabv3_mnv2_ade20k_513_os16_quant](ade20k/deeplabv3_mnv2_ade20k_513_os16_dummy_quant.tflite)| ade20k | 513x513 | 16 | dummy quant |


## How to get fully delegatable .tflite. Why?
TFLite models from Google, such as those in [mobilenetv2_coco_voc_trainaug_8bit](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_8bit_2019_04_26.tar.gz), are from MobilenetV2 input to ArgMax. However, there are 3 types of ops preventing them from fully delegated to NNAPI.

### Resize_bilinear: align_corners not supported by NNAPI
How to fix it: change to False in python source code. 
### Argmax: output_type=int64, which is default value
How to fix it: change it to int32 in source.
### Average_pool_2d: filter H*W > 256
How to fix it: 1. Do something like `--disable_nnapi_cpu=1` or `--nnapi_accelerator_name=neuron-ann` in benchmark_model. 2. Simply skip the constraint in NNAPI delegate source code.

## Generating pb files / exporting files:
### 513x513, OS = 8, quant:
```
PYTHONPATH=`pwd`:`pwd`/slim python deeplab/export_model.py --checkpoint_path=/tmp/deeplabv3_mnv2_pascal_train_aug_8bit/model.ckpt --export_path=/tmp/deeplab_export/deeplabv3_mnv2_pascal_513_os8_quant.pb --model_variant="mobilenet_v2" --quantize_delay_step=0
```
### 513x513, OS =16, quant: 
```
PYTHONPATH=`pwd`:`pwd`/slim python deeplab/export_model.py --checkpoint_path=/tmp/deeplabv3_mnv2_pascal_train_aug_8bit/model.ckpt --export_path=/tmp/deeplab_export/deeplabv3_mnv2_pascal_513_os16_quant.pb --model_variant="mobilenet_v2" --quantize_delay_step=0 --output_stride=16
```
### 257x257, OS = 8, quant:  
```
PYTHONPATH=`pwd`:`pwd`/slim python deeplab/export_model.py --checkpoint_path=/tmp/deeplabv3_mnv2_pascal_train_aug_8bit/model.ckpt --export_path=/tmp/deeplab_export/deeplabv3_mnv2_pascal_257_os8_quant.pb --model_variant="mobilenet_v2" --quantize_delay_step=0 --crop_size=257 --crop_size=257
```
### 513x513, OS =16, quant: 
```
PYTHONPATH=`pwd`:`pwd`/slim python deeplab/export_model.py --checkpoint_path=/tmp/deeplabv3_mnv2_pascal_train_aug_8bit/model.ckpt --export_path=/tmp/deeplab_export/deeplabv3_mnv2_pascal_257_os16_quant.pb --model_variant="mobilenet_v2" --quantize_delay_step=0 --crop_size=257 --crop_size=257 --output_stride=16
```
