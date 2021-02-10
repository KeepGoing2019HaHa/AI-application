1. python test_standalone_torch_fix_128.py: to get onnx file
2. python -m onnxsim deoldify.onnx deoldify-sim.onnx: to remove some ops
3. ./onnx2ncnn deoldify-sim.onnx deoldify.256.param deoldify.256.bin: convert to ncnn model
4. 修改param中的reshape

converted models: https://drive.google.com/drive/folders/1tywoozrH76DjSFNFhQGjnRzZGD1EPKm7?usp=sharing
more infos can be seen in  https://zhuanlan.zhihu.com/p/350332071







############ for Chinese users ###########################


1. python test_standalone_torch_fix_128.py 得到onnx文件
2. python -m onnxsim deoldify.onnx deoldify-sim.onnx去除胶水op
3. ./onnx2ncnn deoldify-sim.onnx deoldify.256.param deoldify.256.bin转化成ncnn模型
4. 修改param中的reshape

转化后模型地址见 https://drive.google.com/drive/folders/1tywoozrH76DjSFNFhQGjnRzZGD1EPKm7?usp=sharing
更多信息可见https://zhuanlan.zhihu.com/p/350332071
