#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
#
# Implemented using the pybox2d SWIG interface for Box2D (pybox2d.googlecode.com)
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
import math
import random
from time import time

import gym

import numpy as np
import pygame
from Box2D import b2Color, b2Vec2
from gym.spaces import Discrete, Dict, Box
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

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

        # fixed paramters: weight of object A and the positions of both boxes
        # ??

        # x: Determines the x-coordinate to place the box
        # y: y-coordinate of the box
        # box: 1 --> choose BoxA, 2 --> BoxB
        self.action_space = Dict({
            "x": Box(low=-10., high=10., shape=(1,1), dtype=float),
            "y": Box(low=1., high=10., shape=(1, 1), dtype=float),
            "box": Discrete(2) # 1: BoxA, 2: BoxB
        })

        # setting up the objects on the screen
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxes = []

        # create Box A
        randomPositionA = -4. - 2 * random.random()  # between -4 and -6
        #randomDensityA = 4. + 2 * random.random()  # between 4 and 6
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = 4. + 2 * random.random()
        #randomDensityB = 4. + 2 * random.random()
        self.boxB = self.createBox(randomPositionB, self.y, DENSITY, BOXSIZE)

        topCoordinate = Vec2(0,6)
        self.triangle = self.world.CreateStaticBody(
            position=(0,0),
            fixtures=fixtureDef(shape=polygonShape(vertices=[(-1, 0), (1, 0), topCoordinate]), density=100)
        )

        # TODO: set triangle green when the scale is leveled, red when the angle is not 0°

        self.bar = self.world.CreateDynamicBody(
            position=topCoordinate,
            fixtures=fixtureDef(shape=polygonShape(box=(15, 0.3)), density=1),
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate) #, anchor=topCoordinate)

        self.state = [self.boxA, self.boxB, self.bar]  # ?

    def Keyboard(self, key):
        """
        The key is from Keys.K_*
        (e.g., if key == Keys.K_z: ... )
        """
        if key == Keys.K_s:
            x, y = pygame.mouse.get_pos()
            pos_x, _ = self.ConvertScreenToWorld(x, y)
            self.createBox(pos_x, 30.0, 0.75 ** 3 * DENSITY, 0.75 * BOXSIZE)
        if key == Keys.K_b:
            x, y = pygame.mouse.get_pos()
            pos_x, _ = self.ConvertScreenToWorld(x, y)
            self.createBox(pos_x, 30.0, DENSITY, BOXSIZE)

    def ConvertScreenToWorld(self, x, y):
        """
        Returns a b2Vec2 indicating the world coordinates of screen (x,y)
        """
        return Vec2((x + self.viewOffset.x) / self.viewZoom,
                      ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def createBox(self, pos_x, pos_y = None, density = None, boxsize = None):
        if not pos_y:
            pos_y = self.y
        if not density:
            density = DENSITY
        if not boxsize:
            boxsize = self.fixedBoxSize
        newBox = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            fixtures=fixtureDef(shape=polygonShape(box=(boxsize, boxsize)),
                                density=density, friction=1.),
            userData=boxsize, # save this because you somehow cannot access fixture data later
        )
        self.boxes.append(newBox)
        return newBox

    def deleteBox(self, box): # todo: fix, maybe ID for every b2Body object
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
        self.deleteBox(box)
        x = box.position[0] + deltaX
        y = box.position[1] + deltaY
        boxsize = box.userData  #self.fixedBoxSize
        density = box.mass/(4*boxsize) #DENSITY
        movedBox = self.createBox(x, y, density, boxsize)
        return movedBox

    def moveBoxTo(self, box, x, y):
        "Place a box at a specific position on the field"
        self.deleteBox(box)
        boxsize = box.userData  # self.fixedBoxSize
        density = box.mass / (4 * boxsize)  # DENSITY
        movedBox = self.createBox(x, y, density, boxsize)
        return movedBox

    def step(self, action):
        self.stepCount += 1 #?
        # Don't do anything if the setting's Hz are <= 0
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
        self.world.Step(timeStep, velocityIterations,
                        positionIterations)
        self.world.ClearForces()
        t_step = time() - t_step

        # Update the debug draw settings so that the vertices will be properly
        # converted to screen coordinates
        t_draw = time()

        # extract informations from action
        x = action["x"][0,0]
        y = action["y"][0,0]
        box = action["box"]

        # do stuff
        self.description = f"{self.joint.angle * 180 / math.pi}°"

        # Placed after the physics step, it will draw on top of physics objects
        # self.Print("*** Base your own testbeds on me! ***")

        if (abs(self.bar.angle) > 0.39
                or self.boxA.position[0] > 0
                or self.boxB.position[0] < 0):
            self.reset()
            return self.state, 0, True, {}

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
        pass

    def close(self):
        pass

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
        self.description = f"{self.joint.angle * 180/math.pi}°"

        # Placed after the physics step, it will draw on top of physics objects
        #self.Print("*** Base your own testbeds on me! ***")

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
        if action in {-1,0,1}:
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
        done = True if (all(vel == b2Vec2(0, 0) for vel in velocities) and abs(self.bar.angle) < FAULTTOLERANCE) else False
        # done = True if (all(vel == b2Vec2(0, 0) for vel in velocities)) else False

        # placeholder for info
        info = {}

        if (abs(self.bar.angle) - 0.390258252620697) > 0.000000001:
            print("done")

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

    def render(self, mode="human"): #todo
        renderer = self.renderer

        if renderer is not None:
            renderer.StartDraw()

        self.world.DrawDebugData()
        print("done")
        pass

    # TODO: fix it
    def reset(self):
        self.deleteAllBoxes()

        randomPositionA = -4. - 2 * random.random()
        #randomDensityA = 4. + 2 * random.random()
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = 4. - 2 * random.random()
        #randomDensityB = 4. + 2 * random.random()
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