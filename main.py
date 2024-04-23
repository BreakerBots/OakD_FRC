from depthai_sdk import OakCamera, ArgsParser
from depthai import NNData
from depthai_sdk.classes import DetectionPacket
from depthai_sdk.classes import Detections
from depthai_sdk.classes.packets import TrackerPacket
import depthai as dai
import argparse
import cv2
import ntcore
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='model/best.json', type=str)
args = ArgsParser.parseArgs(parser)
ntInst = ntcore.NetworkTableInstance.getDefault()

# start a NT4 client
ntInst.startClient4("example client")

# connect to a roboRIO with team number TEAM
ntInst.setServerTeam(5104)	

import cv2

# starting a DS client will try to get the roboRIO address from the DS application
ntInst.startDSClient()

# connect to a specific host/port
ntInst.setServer("host", ntcore.NetworkTableInstance.kDefaultPort4)
ntTable = ntInst.getTable("OAKD")

ntTopicNumTargets = ntTable.getIntegerTopic("numTargets")
ntTopicCaptureTimestamp = ntTable.getDoubleTopic("captureLatency")
ntTopicArraySpatialX = ntTable.getDoubleArrayTopic("spatialX")
ntTopicArraySpatialY = ntTable.getDoubleArrayTopic("spatialY")
ntTopicArraySpatialZ = ntTable.getDoubleArrayTopic("spatialZ")
ntTopicArrayObjectId = ntTable.getIntegerArrayTopic("objectID")
ntTopicArrayConf = ntTable.getDoubleArrayTopic("confidence")
ntTopicArrayTargetClass = ntTable.getStringArrayTopic("class")

numTargetsPub = ntTopicNumTargets.publish()
captureTimestampPub = ntTopicCaptureTimestamp.publish()
xArrPub = ntTopicArraySpatialX.publish()
yArrPub = ntTopicArraySpatialY.publish()
zArrPub = ntTopicArraySpatialZ.publish()
objIdPub = ntTopicArrayObjectId.publish()
confArrPub = ntTopicArrayConf.publish()
classArrPub = ntTopicArrayTargetClass.publish()

def captureTracklets(packet : TrackerPacket):

    numTargets = 0
    spatialX = []
    spatialY = []
    spatialZ = []
    objectID = []
    objectClass = []
    conf = []
    for track in packet.daiTracklets.tracklets:
        if (track.status == dai.Tracklet.TrackingStatus.TRACKED or track.status == dai.Tracklet.TrackingStatus.NEW):
            numTargets+=1
            track.spatialCoordinates
            spatialX.append(track.spatialCoordinates.x)
            spatialY.append(track.spatialCoordinates.y)
            spatialZ.append(track.spatialCoordinates.z)
            objectID.append(track.id)
            objectClass.append(nn.get_labels()[track.label])
            conf.append(track.srcImgDetection.confidence)
    numTargetsPub.setDefault(0)
    numTargetsPub.set(numTargets)
    if (numTargets > 0): 
        ts = dai.Clock.now().__sub__(packet.msg.getTimestamp()).total_seconds()
        captureTimestampPub.setDefault(0.0)
        captureTimestampPub.set(ts)
        print(ts)
        confArrPub.setDefault([])
        confArrPub.set(conf)
        classArrPub.setDefault([])
        classArrPub.set(objectClass)
        objIdPub.setDefault([])
        objIdPub.set(objectID)
        xArrPub.setDefault([])
        xArrPub.set(spatialX)
        yArrPub.setDefault([])
        yArrPub.set(spatialY)
        zArrPub.setDefault([])
        zArrPub.set(spatialZ)
    frame = packet.visualizer.draw(packet.decode())
    cv2.imshow("tracker", frame)
    return



with OakCamera(args=args, usb_speed=dai.UsbSpeed.SUPER_PLUS) as oak:
    color = oak.create_camera('color', encode=False)
    stereo = oak.create_stereo('480p')
    oak.pipeline.setXLinkChunkSize(0)
    stereo.config_stereo(subpixel=False, lr_check=True)
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=stereo, tracker=True)
    nn.config_tracker(tracker_type=dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM, assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID, forget_after_n_frames=1, apply_tracking_filter=True, calculate_speed=False, threshold=0.4)
    nn.node.setNumInferenceThreads(1)
    nn.node.setNumNCEPerInferenceThread(2)
    nn.node.input.setBlocking(False)
    nn.node.input.setQueueSize(1)
    # nn.config_spatial(
    #     bb_scale_factor=0.75, # Scaling bounding box before averaging the depth in that ROI
    #     lower_threshold=300, # Discard depth points below 30cm
    #     upper_threshold=10000, # Discard depth pints above 10m
    #     # Average depth points before calculating X and Y spatial coordinates:
    #     calc_algo=dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    # )

    visualizer = oak.visualize(nn, fps=True)
    visualizer = oak.visualize(nn.out.tracker, callback=captureTracklets)
    oak.start(blocking=True)
