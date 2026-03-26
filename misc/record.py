import pybullet as p
import time

p.connect(p.GUI)

log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/jay/Project_IISc_Intern/test.mp4")

time.sleep(5)

for i in range(300):
    p.stepSimulation()
    time.sleep(1./240.)

p.stopStateLogging(log_id)
p.disconnect()