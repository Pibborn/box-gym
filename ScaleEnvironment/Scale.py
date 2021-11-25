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
import sys
from time import time, sleep

import numpy as np
import pygame
from Box2D import b2Color, b2Vec2, b2DrawExtended
from gym.spaces import Discrete, Dict, Box
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

from ScaleEnvironment.settings import fwSettings

import gym

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
        self.y, L, a, b = 6.0 + BOXSIZE, 12.0, 1.0, 2.0
        self.counter = 0  # ?

        # fixed parameters: weight of object A and the positions of both boxes
        # ??

        # x: Determines the x-coordinate to place the box
        # y: y-coordinate of the box
        # box: 1 --> choose BoxA, 2 --> BoxB
        self.action_space = Dict({
            "x": Box(low=-10., high=1., shape=(1, 1), dtype=float),
            "y": Box(low=1., high=10., shape=(1, 1), dtype=float),
            "box": Discrete(2)  # 1: BoxA, 2: BoxB
        })

        self.observation_space = Dict(spaces = {
            "x1": Box(low=-20., high=20., shape=(1,), dtype=float),
            "y1": Box(low=0., high=15., shape=(1,), dtype=float),
            "x2": Box(low=-20., high=20., shape=(1,), dtype=float),
            "y2": Box(low=0., high=15., shape=(1,), dtype=float),
            "angle": Box(low=-390258252620697, high=390258252620697, shape=(1,), dtype=float),  # 1: BoxA, 2: BoxB
        })

        # setting up the objects on the screen
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxes = []

        # create Box A
        randomPositionA = -4. - 2 * random.random()  # between -4 and -6
        # randomDensityA = 4. + 2 * random.random()  # between 4 and 6
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = 4. + 2 * random.random()
        # randomDensityB = 4. + 2 * random.random()
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

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate)

        self.state = [self.boxA.position[0], self.boxA.position[1], self.boxB.position[0], self.boxB.position[1], self.bar.angle]  # ?

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

    def placeBox(self, box, pos):
        "Place a box on the scale"
        x = math.cos(self.bar.angle) * pos
        y = 6 + math.tan(self.bar.angle) * pos + BOXSIZE
        placedBox = self.moveBoxTo(box, x, y)
        placedBox.angle = self.bar.angle
        return placedBox

    def step(self, action=None, settings=None):
        # Don't do anything if the setting's Hz are <= 0
        hz = 60.0
        velocityIterations = 8
        positionIterations = 3

        if hz > 0.0:
            timeStep = 1.0 / hz
        else:
            timeStep = 0.0

        if not settings:
            settings = fwSettings

        # check if test failed --> reward = -1
        if (abs(self.bar.angle) > 0.390258252620697
                or self.boxA.position[0] > 0
                or self.boxB.position[0] < 0):
            self.reset()
            return self.state, -1, True, {}

        # catch special case that no action was executed
        if not action:
            self.world.Step(timeStep, velocityIterations,
                            positionIterations)
            self.world.ClearForces()
            self.render()
            return [self.boxA.position[0], self.boxA.position[1], self.boxB.position[0], self.boxB.position[1], self.bar.angle], 0, False, {}

        # extract information from action
        x = action["x"][0, 0]
        y = action["y"][0, 0]
        box = action["box"]

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        self.world.Step(timeStep, velocityIterations,  # todo: Step function called after moving the boxes
                        positionIterations)
        self.world.ClearForces()

        self.description = f"{self.joint.angle * 180 / math.pi}°"

        def boxesOnScale():
            # TODO: fix
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            return val

        # perform action
        if box == 0:
            #self.boxA = self.moveBoxTo(self.boxA, x, y)
            self.boxA = self.placeBox(self.boxA, x)
        elif box == 1:
            #self.boxB = self.moveBoxTo(self.boxB, x, y)
            self.boxB = self.placeBox(self.boxB, x)

        self.state = [self.boxA.position[0], self.boxA.position[1], self.boxB.position[0], self.boxB.position[1], self.bar.angle]
        # Calculate reward (Scale in balance?)
        if abs(self.bar.angle) < FAULTTOLERANCE and boxesOnScale():
            reward = 1
        #elif not boxesOnScale():
        #    reward = -1
        else:
            reward = 0
        # alternatively: reward = (0.390258252620697 - self.bar.angle) / 0.390258252620697

        # no movement and in balance --> done
        velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        done = True if (  # TODO: wait a few seconds/look at forces to be 100% sure the scale won't move again
                all(vel == b2Vec2(0, 0) for vel in velocities) and abs(self.bar.angle) < FAULTTOLERANCE) else False
        # done = True if (all(vel == b2Vec2(0, 0) for vel in velocities)) else False

        # placeholder for info
        info = {}

        #self.render()

        #print(self.state, reward, done, info)

        return self.state, reward, done, info

    def close(self):
        pygame.quit()
        sys.exit()

    def render(self, mode="human"):  # todo
        from gym.envs.classic_control import rendering
        renderer = self.renderer

        self.screen.fill((0, 0, 0))

        # Set the flags based on what the settings show
        if renderer:
            # convertVertices is only applicable when using b2DrawExtended.  It
            # indicates that the C code should transform box2d coords to screen
            # coordinates.
            is_extended = isinstance(renderer, b2DrawExtended)
            renderer.flags = dict(drawShapes=True,
                                  drawJoints=False,  # True
                                  drawAABBs=False,
                                  drawPairs=False,
                                  drawCOMs=False,
                                  convertVertices=is_extended,
                                  )

        self.world.warmStarting = True
        self.world.continuousPhysics = True
        self.world.subStepping = False

        # Reset the collision points
        self.points = []

        if renderer is not None:
            renderer.StartDraw()

        self.world.DrawDebugData()

        if renderer:
            renderer.EndDraw()
            pygame.display.flip()

    def reset(self):
        self.deleteAllBoxes()

        randomPositionA = -4. - 2 * random.random()
        randomDensityA = 4. + 2 * random.random()
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = 4. + 2 * random.random()
        randomDensityB = 4. + 2 * random.random()
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
        return [self.boxA.position[0], self.boxA.position[1], self.boxB.position[0], self.boxB.position[1], self.bar.angle]

    def seed(self, number): # todo:
        pass

# More functions can be changed to allow for contact monitoring and such.
# See the other testbed examples for more information.

if __name__ == "__main__":
    main(Scale)