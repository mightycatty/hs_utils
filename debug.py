from tklib.tf_graph_toolkit import convert_frozen_pb_to_onnx

onnx_dir = r'F:\heshuai\proj\stylegan_trt_convert\utils\stylegan2_generator_trainingFalse.opt.pb'
result = convert_frozen_pb_to_onnx(onnx_dir)