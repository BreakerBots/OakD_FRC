from depthai_sdk import OakCamera, ArgsParser
from depthai import NNData
from depthai_sdk.classes import DetectionPacket
from depthai_sdk.classes import Detections
import depthai as dai
import argparse
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
ntTopicArrayConf = ntTable.getDoubleArrayTopic("confidence")
ntTopicArrayTargetClass = ntTable.getStringArrayTopic("class")

numTargetsPub = ntTopicNumTargets.publish()
captureTimestampPub = ntTopicCaptureTimestamp.publish()
xArrPub = ntTopicArraySpatialX.publish()
yArrPub = ntTopicArraySpatialY.publish()
zArrPub = ntTopicArraySpatialZ.publish()
confArrPub = ntTopicArrayConf.publish()
classArrPub = ntTopicArrayTargetClass.publish()

def captureDetections(packet : DetectionPacket):
    detections = packet.detections
    numTgts = len(detections)
    numTargetsPub.setDefault(0)
    numTargetsPub.set(numTgts)
    if (numTgts > 0):
        ts = dai.Clock.now().__sub__(packet.img_detections.getTimestamp()).total_seconds()
        captureTimestampPub.setDefault(0.0)
        captureTimestampPub.set(ts)
        spatialX = []
        spatialY = []
        spatialZ = []
        conf = []
        classStr = []
        for i in range(numTgts):
            det = detections[i]
            conf.append(det.confidence)
            classStr.append(det.label_str)
            spatialX.append(det.img_detection.spatialCoordinates.x / 1000.0)
            spatialY.append(det.img_detection.spatialCoordinates.y / 1000.0)
            spatialZ.append(det.img_detection.spatialCoordinates.z / 1000.0)
        confArrPub.setDefault([])
        confArrPub.set(conf)
        classArrPub.setDefault([])
        classArrPub.set(classStr)
        xArrPub.setDefault([])
        xArrPub.set(spatialX)
        yArrPub.setDefault([])
        yArrPub.set(spatialY)
        zArrPub.setDefault([])
        zArrPub.set(spatialZ)
        return

with OakCamera(args=args) as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
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

    oak.visualize(nn, fps=True, scale=2/3)
    oak.visualize(nn.out.passthrough, fps=True)
    oak.callback(nn, callback=captureDetections)
    oak.start(blocking=True)
