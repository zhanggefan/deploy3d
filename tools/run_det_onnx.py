import glob
import io, os
import numpy as np
from deploy3d.symfun.trt_utils import TRTOnnxModule 

    
if __name__ == '__main__':
    input_path = './data/pc_out'
    result_path = './data/pc_bboxes'
    
    root_path = './models'
    model_path = './models/share/model/det3d_ruby'
    onnx_file = glob.glob(os.path.join(model_path, '*.onnx'))[0]
    config_path = glob.glob(os.path.join(model_path, '*.cfg'))[0]
    yaml_file = glob.glob(os.path.join(root_path, '*.yaml'))[0]
    
    trt_module = TRTOnnxModule(onnx_file, config_path, yaml_file)
    
    pts_files = glob.glob(os.path.join(input_path, '*.npy'))
    for pts_file in pts_files:
        bbox_file = os.path.join(result_path, os.path.split(pts_file)[-1])
        bboxes = trt_module.forward(pts_file)
        np.save(bbox_file, bboxes)