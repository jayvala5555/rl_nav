import pybullet as p
import pybullet_data
import time
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# robot = p.loadURDF("r2d2.urdf")

robot1 = p.loadURDF("r2d2.urdf", [2,0,0.1])
robot2 = p.loadURDF("r2d2.urdf", [4,0,0.1])

human = p.loadURDF("sphere2.urdf", [1,1,0.5])

goal = [5.0,8.0,0.0]

# Create a visual shape (sphere in this case)
# baseVisualShapeIndex is the key to creating a non-physical visual object
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.1,  # adjust size as needed
    rgbaColor=[1, 0, 0, 1] # Red color (R, G, B, Alpha)
)

# Create a multi-body with no collision shape (baseCollisionShapeIndex=-1)
# This ensures it is a static, non-physical marker
marker_id = p.createMultiBody(
    baseMass=0, # zero mass makes it static/immovable
    baseCollisionShapeIndex=-1, # no collision shape
    baseVisualShapeIndex=visual_shape_id,
    basePosition=goal
)

t = 0
while p.isConnected():

    posX = math.sin(t) + 3.0
    posY = math.cos(t) + 6.0
    p.resetBasePositionAndOrientation(human, [posX, posY, 0.5], [0.0,0.0,0.0,1.0])

    p.stepSimulation()
    time.sleep(1/10)
    
    t += 0.05