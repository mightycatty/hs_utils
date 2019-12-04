from openvino.inference_engine import IENetwork, IECore
import os
import logging
import sys


class InferenceWithIE:
    def __init__(self, model_xml_dir, device='CPU', cpu_extension=None, **kwargs):
        if cpu_extension is None:
            cpu_extension = r'C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll'
        self.cpu_extension = cpu_extension
        self.model_xml = model_xml_dir
        self.device = device
        logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
        self._model_init()

    def _model_init(self):
        model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        logging.info("Creating Inference Engine")
        ie = IECore()
        found_device = ie.available_devices
        logging.info("found devices:\n{}".format(found_device))
        ie.add_extension(self.cpu_extension, "CPU")
        # Read IR
        logging.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, model_bin))
        net = IENetwork(model=self.model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        # resize network
        # net.reshape({self.input_blob: (1, 3, 256, 256)})
        if "CPU" in self.device:
            supported_layers = ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                logging.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.device, ', '.join(not_supported_layers)))
                logging.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologgingies"
        assert len(net.outputs) == 1, "Sample supports only single output topologgingies"
        logging.info("Preparing input blobs")
        net.batch_size = 1
        # Loading model to the plugin
        logging.info("Loading model to the plugin")
        config = {}
        # config['CPU_THREADS_NUM'] = '1'
        # config['CLDNN_PLUGIN_PRIORITY'] = '0'
        config = None
        self.exec_net = ie.load_network(network=net, device_name=self.device, config=config)

    def predict(self, image):
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res = res[self.out_blob]
        return res