import pybullet as p
from bipedalwalker3d import BipedalWalker3DEnv
import time


if __name__ == "__main__":
    robot = BipedalWalker3DEnv(render_mode="gui")
    p.performCollisionDetection(robot.physicsClientId)

    target_positions = [] 
    forces = [100] * len(robot.jointIds)
    for j in range(len(robot.jointIds)): target_positions.append(0)
    #print(len(target_positions))
    #print(len(robot.jointIds))
    for i in range(10000):
        print("Reward: " + str(robot.reward())) 

        for i in range(len(robot.paramIds)):
            c = robot.paramIds[i]
            targetPos = p.readUserDebugParameter(c)
            p.setJointMotorControl2(robot.robotId, robot.jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)

        p.stepSimulation()
        time.sleep(1./240.)

    robot.close()