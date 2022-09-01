
import cv2

from openvino.inference_engine import IECore
import openvino_models as model_openvino

from classifier_urtils import image_collect

plugin_config = {'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
ie= IECore()
model = model_openvino.OpenPose(ie, "human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml", target_size=None, aspect_ratio=1,
                                prob_threshold=0.1)
input=model.image_blob_name
out_pool=model.pooled_heatmaps_blob_name
out_ht=model.heatmaps_blob_name
out_paf=model.pafs_blob_name
n,c,h,w = model.net.inputs[input].shape
exec_net = ie.load_network(network=model.net,config=plugin_config,device_name="CPU",num_requests = 1)


cap = cv2.VideoCapture('/home/neosoft/Desktop/projects/automatic_object_annotation/collect_image/testing_videos/test.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#output = cv2.VideoWriter('output_class2.mp4', fourcc, fps, (1000,1000))
count=0
count_=0
prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
count=0
collect = True
while True:
    _,frame = cap.read()

    frame = cv2.resize(frame, (1000,1000))
    frame_dummy = frame.copy()

    
    output_transform = model_openvino.OutputTransform(frame.shape[:2], None)
    output_resolution = (frame.shape[1], frame.shape[0])
    inputs, preprocessing_meta = model.preprocess(frame)
    infer_res = exec_net.start_async(request_id=0,inputs={input:inputs["data"]})
    status=infer_res.wait()
    results_pool = exec_net.requests[0].outputs[out_pool]
    results_ht = exec_net.requests[0].outputs[out_ht]
    results_paf = exec_net.requests[0].outputs[out_paf]
    results={"heatmaps":results_ht,"pafs":results_paf,"pooled_heatmaps":results_pool}
    poses,scores=model.postprocess(results,preprocessing_meta)
    #points = output_transform.scale(poses)
    print(frame.shape)

    frame,count = image_collect(frame,poses,0.1,output_transform,count,collect)
    count+=1
    #output.write(frame)
    #cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
#output.release()
cv2.destroyAllWindows()
