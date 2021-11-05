import math
import random
import gym
from gym.spaces import Discrete, Box

import xlsxwriter

import numpy as np
import pygame
from Box2D import b2Vec2, b2Color
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

# not necessary
BOXSIZE = 1.0
DENSITY = 5.0

FAULTTOLERANCE = 0.0001 # for the angle of the bar
STEPSIZE = 0.001

EPISODES = 8 # number of documentations to be documented

workbook = xlsxwriter.Workbook('observations.xlsx')

worksheet = workbook.add_worksheet("random mass(es)")

worksheet.write(0, 0, "BoxA Coordinate")
worksheet.write(0, 1, "BoxA Density")
worksheet.write(0, 2, "BoxB Coordinate")
worksheet.write(0, 3, "BoxB Density")
worksheet.write(0, 4, "Bar Angle")

class Scale(Framework):
    """You can use this class as an outline for your tests."""
    name = "Scale"  # Name of the class to display

    def __init__(self):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(Scale, self).__init__()

        # Initialize all of the objects
        self.y, L, a, b = 16.0, 12.0, 1.0, 2.0
        self.counter = 0
        self.episode = 0

        print(f"Episode {self.episode + 1}")

        # fixed paramters: weight of object A and the positions of both boxes
        self.fixedPositionX1 = - 5
        self.fixedPositionX2 = 6
        self.fixedDensityA = 5.0
        self.fixedBoxSize = 1.0

        # -1: move BoxA to the right
        # 0: don't move any boxes
        # +1: move BoxB to the right
        self.action_space = Discrete(3)

        # setting up the objects on the screen
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        # create Box A
        self.randomPositionA = -4. - 2 * random.random() # between -4 and -6
        self.randomDensityA = 4. + 2 * random.random() # between 4 and 6
        self.boxA = self.world.CreateDynamicBody(
            #position=(self.fixedPositionX1, self.y),
            #fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)), density=self.fixedDensityA, friction=1.),
            position=(self.randomPositionA, self.y),
            fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)),
                                density=self.randomDensityA, friction=1.),
        )

        # create Box B
        self.randomPositionB = 4. + 2 * random.random()
        self.randomDensityB = 4. + 2 * random.random()
        self.boxB = self.world.CreateDynamicBody(
            #position=(self.fixedPositionX2, self.y),
            position=(self.randomPositionB, self.y),
            fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)),
                                density=self.randomDensityB, friction=1.),
        )

        topCoordinate = Vec2(0,6)
        self.triangle = self.world.CreateStaticBody(
            position=(0,0),
            fixtures=fixtureDef(shape=polygonShape(vertices=[(-1, 0), (1, 0), topCoordinate]), density=100),
        )

        #TODO: set triangle green when the scale is leveled, red when the angle is not 0°

        self.bar = self.world.CreateDynamicBody(
            position=topCoordinate,
            fixtures=fixtureDef(shape=polygonShape(box=(15, 0.3)), density=1),
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate) #, anchor=topCoordinate)

        self.state = [self.boxA, self.boxB, self.bar] #?

    def Keyboard(self, key):
        """
        The key is from Keys.K_*
        (e.g., if key == Keys.K_z: ... )
        """
        if key == Keys.K_s:
            x, y = pygame.mouse.get_pos()
            pos, _ = self.ConvertScreenToWorld(x, y)
            newBox = self.world.CreateDynamicBody(
                position=(pos, 30.0),
                fixtures=fixtureDef(shape=polygonShape(box=(0.75*BOXSIZE, 0.75*BOXSIZE)),
                                    density=0.75**3*DENSITY, friction=1.),
            )
        if key == Keys.K_b:
            x, y = pygame.mouse.get_pos()
            pos, _ = self.ConvertScreenToWorld(x, y)
            newBox = self.world.CreateDynamicBody(
                position=(pos, 30.0),
                fixtures=fixtureDef(shape=polygonShape(box=(BOXSIZE, BOXSIZE)),
                                    density=0.75**3*DENSITY, friction=1.),
            )


    def ConvertScreenToWorld(self, x, y):
        """
        Returns a b2Vec2 indicating the world coordinates of screen (x,y)
        """
        return Vec2((x + self.viewOffset.x) / self.viewZoom,
                      ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def Step(self, settings, action = None):
        """Called upon every step.
        You should always call
         -> super(Your_Test_Class, self).Step(settings)
        at the beginning or end of your function.

        If placed at the beginning, it will cause the actual physics step to happen first.
        If placed at the end, it will cause the physics step to happen after your code.
        """
        super(Scale, self).Step(settings)

        # do stuff
        #self.description = f"{float(self.boxA.position.x)},{float(self.boxA.position.y)})"
        self.description = f"Iteration {self.episode + 1}, Angle: {self.joint.angle * 180/math.pi}°"

        # Placed after the physics step, it will draw on top of physics objects
        #self.Print("*** Base your own testbeds on me! ***")
        if (abs(self.bar.angle) > 0.39
                or self.boxA.position[0] > 0
                or self.boxB.position[0] < 0):
            self.counter = 0
            self.reset()


        def boxesOnScale():
            # TODO: fix
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            """if len(self.boxA.contacts) > 0:
                print(self.boxA.contacts[0].contact)"""
            return val
        print(self.boxA.mass, self.boxB.mass)
        # not working
        # perform action if and only if both boxes are still on the scale
        if boxesOnScale():
            if self.bar.angle < -FAULTTOLERANCE and boxesOnScale():
                x = self.boxB.position[0] - STEPSIZE
                y = self.boxB.position[1] - math.tan(-(self.bar.angle)) * STEPSIZE
                self.world.DestroyBody(self.boxB)
                self.boxB = self.world.CreateDynamicBody(
                    # position=(-10, y),
                    position=(x, y),
                    fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)),
                                        density=self.randomDensityB, friction=1.),

                )
            if self.bar.angle > FAULTTOLERANCE and boxesOnScale():
                x = self.boxA.position[0] + STEPSIZE
                y = self.boxA.position[1] + math.tan(-(self.bar.angle)) * STEPSIZE
                self.world.DestroyBody(self.boxA)
                self.boxA = self.world.CreateDynamicBody(
                    # position=(-10, y),
                    position=(x, y),
                    fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)),
                                        density=self.randomDensityA, friction=1.),
                                        #density=self.fixedDensityA, friction=1.),
                )
            else:
                # TODO: scale is horizontal --> restart with new random weights
                if self.counter > 200: # wait to see if it stays on balance
                    self.counter = 0
                    self.episode += 1
                    print(f"Episode {self.episode + 1}")
                    if self.episode < EPISODES:
                        worksheet.write(self.episode, 0, self.boxA.position[0])
                        worksheet.write(self.episode, 1, self.randomDensityA)
                        worksheet.write(self.episode, 2, self.boxB.position[0])
                        worksheet.write(self.episode, 3, self.randomDensityB)
                        worksheet.write(self.episode, 4, self.bar.angle)
                        self.reset()
                    if self.episode == EPISODES:
                        workbook.close()
                    else:
                        self.reset()
                else:
                    self.counter += 1
                pass

        state = [self.bar, self.boxA, self.boxB]
        # state = self.bar.angle

        # Calculate reward (Scale in balance?)
        velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        reward = 1 if (abs(self.bar.angle) < FAULTTOLERANCE) else 0

        # no movement and in balance --> done
        done = True if (all(vel == b2Vec2(0, 0) for vel in velocities) and abs(self.bar.angle) < FAULTTOLERANCE) else False
        # done = True if (all(vel == b2Vec2(0, 0) for vel in velocities)) else False

        # placeholder for info
        info = {}
        if done:
            print(self.boxA.position[0], self.boxB.position[0])
            print(self.randomDensityA, self.randomDensityB)
        return state, reward, done, info


    def ShapeDestroyed(self, shape):
        """
        Callback indicating 'shape' has been destroyed.
        """
        pass

    def JointDestroyed(self, joint):
        """
        The joint passed in was removed.
        """
        pass

    # TODO: fix it
    def reset(self):
        # reset positons of Box A and Box B
        # self.boxA.position = (self.fixedPositionX1, self.y)
        self.world.DestroyBody(self.boxA)
        self.world.DestroyBody(self.boxB)

        self.randomPositionA = -4. - 2 * random.random()
        self.randomDensityA = 4. + 2 * random.random()
        self.boxA = self.world.CreateDynamicBody(
            position=(self.randomPositionA, self.y),
            fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)),
                                density=self.randomDensityA, friction=1.),
        )

        self.randomPositionB = 4. - 2 * random.random()
        self.randomDensityB = 4. + 2 * random.random()
        self.boxB = self.world.CreateDynamicBody(
            position=(self.randomPositionB, self.y),
            fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)),
                                density=self.randomDensityB, friction=1.),
        )

        #TODO: function for movement (paramters: box and distance/direction)

        # rearrange the bar to 0 degree
        self.bar.angle = 0

        # Reset the reward
        # TODO

        # Determine a new weight for the Box B (?)
        #TODO

        # return the observation
        return self.bar.angle # ?? self.canvas


# More functions can be changed to allow for contact monitoring and such.
    # See the other testbed examples for more information.

if __name__ == "__main__":
    main(Scale)