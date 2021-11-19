import math
import random
import sys
from time import time, sleep

import gym

import numpy as np
import pygame
from Box2D import b2Color, b2Vec2, b2_addState, b2_persistState, b2DrawExtended, b2_epsilon
from gym.spaces import Discrete, Dict, Box
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

from ScaleEnvironment.settings import fwSettings

BOXSIZE = 1.0
DENSITY = 5.0
BOXSIZE = 1.0

FAULTTOLERANCE = 0.001  # for the angle of the bar
STEPSIZE = 0.001


class Scale(Framework, gym.Env):
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

        # x: Determines the x-coordinate to place the box
        # y: y-coordinate of the box
        # box: 1 --> choose BoxA, 2 --> BoxB
        self.action_space = Dict({
            "x": Box(low=-10., high=-1., shape=(1, 1), dtype=float),
            "y": Box(low=1., high=10., shape=(1, 1), dtype=float),
            "box": Discrete(2)  # 1: BoxA, 2: BoxB
        })

        # setting up the objects on the screen
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxes = []

        # create Box A
        randomPositionA = -4. - 2 * random.random()  # between -4 and -6
        randomDensityA = 4. + 2 * random.random()  # between 4 and 6
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = 4. + 2 * random.random()
        randomDensityB = 4. + 2 * random.random()
        self.boxB = self.createBox(randomPositionB, self.y, DENSITY, BOXSIZE)

        topCoordinate = Vec2(0, 6)
        self.triangle = self.world.CreateStaticBody(
            position=(0, 0),
            fixtures=fixtureDef(shape=polygonShape(vertices=[(-1, 0), (1, 0), topCoordinate]), density=100)
        )

        # TODO: set triangle green when the scale is leveled, red when the angle is not 0°

        self.bar = self.world.CreateDynamicBody(
            position=topCoordinate,
            fixtures=fixtureDef(shape=polygonShape(box=(15, 0.3)), density=1),
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle,
                                                    anchor=topCoordinate)

        self.state = [self.boxA, self.boxB, self.bar]  # ?

    def ConvertScreenToWorld(self, x, y):
        """
        Returns a b2Vec2 indicating the world coordinates of screen (x,y)
        """
        return Vec2((x + self.viewOffset.x) / self.viewZoom,
                    ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def createBox(self, pos_x, pos_y=None, density=DENSITY, boxsize=BOXSIZE):
        if not pos_y:
            pos_y = self.y

        newBox = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            fixtures=fixtureDef(shape=polygonShape(box=(boxsize, boxsize)),
                                density=density, friction=1.),
            userData=boxsize,  # save this because you somehow cannot access fixture data later
        )
        self.boxes.append(newBox)
        return newBox

    def deleteBox(self, box):  # todo: fix, maybe ID for every b2Body object
        "Delete a box from the world"
        if box not in self.boxes:
            print("Box not found")
            return
        pos = self.boxes.index(box)
        self.boxes.pop(pos)
        self.world.DestroyBody(box)
        return

    def deleteAllBoxes(self):
        """deletes every single box in the world"""
        for box in self.boxes:
            try:
                self.world.DestroyBody(box)
            except:
                pass
        self.boxes = []

    def moveBox(self, box, deltaX, deltaY):
        "Move a box in the world along a given vector (deltaX,deltaY)"
        x = box.position[0] + deltaX
        y = box.position[1] + deltaY
        return self.moveBoxTo(box, x, y)

    def moveBoxTo(self, box, x, y):
        "Place a box at a specific position on the field"
        self.deleteBox(box)
        boxsize = box.userData  # self.fixedBoxSize
        density = box.mass / (4 * boxsize)  # DENSITY
        movedBox = self.createBox(x, y, density, boxsize)
        return movedBox

    def step(self, action=None, settings=None):
        # Don't do anything if the setting's Hz are <= 0
        if not settings:
            settings = fwSettings
        #super(Framework, self).Step(settings)

        hz = 60.0
        velocityIterations = 8
        positionIterations = 3

        if hz > 0.0:
            timeStep = 1.0 / hz
        else:
            timeStep = 0.0

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        t_step = time()
        self.world.Step(timeStep, velocityIterations, # todo: Step function called after moving the boxes
                        positionIterations)
        self.world.ClearForces()
        t_step = time() - t_step

        # Update the debug draw settings so that the vertices will be properly
        # converted to screen coordinates
        t_draw = time()

        # catch special case that no action was executed
        if not action:
            self.render()
            return [self.bar, self.boxA, self.boxB], 0, False, {}

        # extract information from actions
        x = action["x"][0, 0]
        y = action["y"][0, 0]
        box = action["box"]

        self.description = f"{self.joint.angle * 180 / math.pi}°"

        # check if test failed --> reward = -1
        if (abs(self.bar.angle) > 0.39
                or self.boxA.position[0] > 0
                or self.boxB.position[0] < 0):
            self.reset()
            return self.state, -1, True, {}

        def boxesOnScale():
            # TODO: fix
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            return val

        # perform action
        if box == 1:
            self.boxA = self.moveBoxTo(self.boxA, x, y)
        elif box == 2:
            self.boxB = self.moveBoxTo(self.boxB, x, y)

        state = [self.bar, self.boxA, self.boxB]
        # Calculate reward (Scale in balance?)
        # reward = 1 if (abs(self.bar.angle) < FAULTTOLERANCE) else 0
        reward = (0.390258252620697 - self.bar.angle) / 0.390258252620697

        # no movement and in balance --> done
        velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        done = True if (
                all(vel == b2Vec2(0, 0) for vel in velocities) and abs(self.bar.angle) < FAULTTOLERANCE) else False
        # done = True if (all(vel == b2Vec2(0, 0) for vel in velocities)) else False

        # placeholder for info
        info = {}

        if (abs(self.bar.angle) - 0.390258252620697) > 0.000000001:
            print("done")

        #self.render()

        return state, reward, done, info

    def Step(self, settings, action=None):  # todo : delete
        """Called upon every step.
        You should always call
         -> super(Your_Test_Class, self).Step(settings)
        at the beginning or end of your function.

        If placed at the beginning, it will cause the actual physics step to happen first.
        If placed at the end, it will cause the physics step to happen after your code.
        """
        # super(Scale, self).Step(settings)
        self.stepCount += 1
        # Don't do anything if the setting's Hz are <= 0
        if settings.hz > 0.0:
            timeStep = 1.0 / settings.hz
        else:
            timeStep = 0.0

        action = self.action_space.sample()
        self.step(action)
        self.render()

        # do stuff
        self.description = f"{self.joint.angle * 180 / math.pi}°"

        # Placed after the physics step, it will draw on top of physics objects
        # self.Print("*** Base your own testbeds on me! ***")

        state = [self.bar, self.boxA, self.boxB]
        # state = self.bar.angle

        if (abs(self.bar.angle) > 0.39
                or self.boxA.position[0] > 0
                or self.boxB.position[0] < 0):
            self.reset()

        def boxesOnScale():
            # TODO: fix
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            return val

        # not working
        # perform action if and only if both boxes are still on the scale
        """if action in {-1,0,1}:
            if not boxesOnScale():
                self.reset()
            if action == -1:
                deltaX = - STEPSIZE
                deltaY = - math.tan(-self.bar.angle) * STEPSIZE
                self.moveBox(self.boxB, deltaX, deltaY)
            elif action == 0:
                if abs(self.bar.angle) < FAULTTOLERANCE:
                    reward = 1 #?
                pass
            elif action == 1:
                deltaX = STEPSIZE
                deltaY = math.tan(-self.bar.angle) * STEPSIZE
                self.moveBox(self.boxA, deltaX, deltaY)
"""
        if False:
            pass

        else:
            if boxesOnScale():
                if self.bar.angle < -FAULTTOLERANCE and boxesOnScale():
                    deltaX = - STEPSIZE
                    deltaY = - math.tan(-self.bar.angle) * STEPSIZE
                    self.moveBox(self.boxB, deltaX, deltaY)
                if self.bar.angle > FAULTTOLERANCE and boxesOnScale():
                    deltaX = STEPSIZE
                    deltaY = math.tan(-self.bar.angle) * STEPSIZE
                    self.moveBox(self.boxA, deltaX, deltaY)
                else:
                    if self.counter > 200:
                        self.reset()
                    self.counter += 1

        state = [self.bar, self.boxA, self.boxB]
        # Calculate reward (Scale in balance?)
        velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        reward = 1 if (abs(self.bar.angle) < FAULTTOLERANCE) else 0

        # no movement and in balance --> done
        done = True if (
                    all(vel == b2Vec2(0, 0) for vel in velocities) and abs(self.bar.angle) < FAULTTOLERANCE) else False
        # done = True if (all(vel == b2Vec2(0, 0) for vel in velocities)) else False

        # placeholder for info
        info = {}

        if (abs(self.bar.angle) - 0.390258252620697) > 0.000000001:
            print("done")

        return state, reward, done, info

    def close(self):
        pygame.quit()
        sys.exit()

    def render(self, mode="human"):  # todo
        from gym.envs.classic_control import rendering
        renderer = self.renderer
        clock = pygame.time.Clock()

        # Set the flags based on what the settings show
        if renderer:
            # convertVertices is only applicable when using b2DrawExtended.  It
            # indicates that the C code should transform box2d coords to screen
            # coordinates.
            is_extended = isinstance(renderer, b2DrawExtended)

            """renderer.flags = dict(drawShapes=settings.drawShapes,
                                  drawJoints=settings.drawJoints,
                                  drawAABBs=settings.drawAABBs,
                                  drawPairs=settings.drawPairs,
                                  drawCOMs=settings.drawCOMs,
                                  convertVertices=is_extended,
                                  )"""
            renderer.flags = dict(drawShapes=True,
                                  drawJoints=False,  # True
                                  drawAABBs=False,
                                  drawPairs=False,
                                  drawCOMs=False,
                                  convertVertices=is_extended,
                                  )

        # Set the other settings that aren't contained in the flags
        """self.world.warmStarting = settings.enableWarmStarting  
        self.world.continuousPhysics = settings.enableContinuous 
        self.world.subStepping = settings.enableSubStepping"""

        self.world.warmStarting = True
        self.world.continuousPhysics = True
        self.world.subStepping = False

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        """self.world.Step(timeStep, settings.velocityIterations,
                        settings.positionIterations)"""
        self.world.ClearForces()

        if renderer is not None:
            renderer.StartDraw()

        self.world.DrawDebugData()

        # Take care of additional drawing (fps, mouse joint, slingshot bomb,
        # contact points)

        if renderer:
            renderer.EndDraw()

            pygame.display.flip()
            clock.tick(60)

    # TODO: fix it
    def reset(self):
        self.deleteAllBoxes()

        randomPositionA = -4. - 2 * random.random()
        # randomDensityA = 4. + 2 * random.random()
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = 4. - 2 * random.random()
        # randomDensityB = 4. + 2 * random.random()
        self.boxB = self.createBox(randomPositionB, self.y, DENSITY, BOXSIZE)

        self.boxes = [self.boxA, self.boxB]

        # TODO: function for movement (paramters: box and distance/direction)

        # rearrange the bar to 0 degree
        self.bar.angle = 0
        self.bar.angularVelocity = 0.

        self.counter = 0

        # Reset the reward
        # TODO

        # Determine a new weight for the Box B (?)
        # TODO

        # return the observation
        return self.bar.angle  # ?? self.canvas


# More functions can be changed to allow for contact monitoring and such.
# See the other testbed examples for more information.

if __name__ == "__main__":
    main(Scale)
