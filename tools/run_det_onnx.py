import glob
import io, os
from deploy3d.symfun.trt_utils import TRTOnnxModule 

    
if __name__ == '__main__':
    input_path = './data/pc_out'
    result_path = './data/pc_bboxes'
    
    onnx_file = './models/det3d_ruby/yolox3d_voxel_ruby_cowadataset_1f.onnx'
    trt_module = TRTOnnxModule(onnx_file)
    
    pts_files = glob.glob(os.path.join(input_path, '*.npy'))
    for pts_file in pts_files:
        trt_module.forward(pts_file)