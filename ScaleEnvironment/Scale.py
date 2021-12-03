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
from gym.utils import seeding
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

import gym

BOXSIZE = 1.0
DENSITY = 5.0
BOXSIZE = 1.0
BARLENGTH = 15

FAULTTOLERANCE = 0.001  # for the angle of the bar
STEPSIZE = 0.001

WAITINGITERATIONS = 20  # maximum iterations to wait per episode

class Scale(Framework, gym.Env):
    """You can use this class as an outline for your tests."""
    name = "Scale"  # Name of the class to display

    def __init__(self, rendering = True):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(Scale, self).__init__(rendering)

        self.seed()

        # Initialize all of the objects
        self.y, L, a, b = 6.0 + BOXSIZE, 12.0, 1.0, 2.0
        self.counter = 0  # ?

        self.rendering = rendering

        # fixed parameters: weight of object A and the positions of both boxes
        # ??

        """
        # pos: Determines the x-coordinate to place the box on the bar
        # box: 0 --> choose BoxA, 1 --> BoxB
        self.action_space = Dict({
            "pos": Box(low=-10., high=-0.5 - BOXSIZE, shape=(1, 1), dtype=float),
            "box": Discrete(2)  # 0: BoxA, 1: BoxB
        })
        """

        #########################################################################
        # delta_pos: move box along the bar with this value
        # box: 0 --> BoxA, 1 --> BoxB
        self.action_space = Dict({
            "delta_pos": Box(low=-1., high=+1., shape=(1, 1), dtype=float),
            "box": Discrete(2)  # 0: BoxA, 1: BoxB
        })

        self.observation_space = Dict(spaces={
            "x1": Box(low=-20., high=20., shape=(1,), dtype=float),
            "y1": Box(low=0., high=15., shape=(1,), dtype=float),
            "x2": Box(low=-20., high=20., shape=(1,), dtype=float),
            "y2": Box(low=0., high=15., shape=(1,), dtype=float),
            "angle": Box(low=-390258252620697, high=390258252620697, shape=(1,), dtype=float),  # 0: BoxA, 1: BoxB,
            # angular velocity of the bar, negative: moves to the right, positive: moves to the left
            "vel": Box(low=-2., high=2., shape=(1,), dtype=float)
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
            fixtures=fixtureDef(shape=polygonShape(box=(BARLENGTH, 0.3)), density=1),
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=self.bar, bodyB=self.triangle, anchor=topCoordinate)

        self.state = [self.boxA.position[0], self.boxA.position[1],
                      self.boxB.position[0], self.boxB.position[1],
                      self.bar.angle, self.bar.angularVelocity]

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
        """Delete a box from the world"""
        if box not in self.boxes:
            print("Box not found")
            return
        pos = self.boxes.index(box)
        self.boxes.pop(pos)
        self.world.DestroyBody(box)
        return

    def deleteAllBoxes(self):
        """Deletes every single box in the world"""
        for box in self.boxes:
            try:
                self.world.DestroyBody(box)
            except:
                pass
        self.boxes = []

    def moveBox(self, box, deltaX, deltaY):
        """Move a box in the world along a given vector (deltaX,deltaY)"""
        x = box.position[0] + deltaX
        y = box.position[1] + deltaY
        return self.moveBoxTo(box, x, y)

    def moveBoxTo(self, box, x, y):
        """Place a box at a specific position on the field"""
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

    def moveAlongBar(self, box, delta_pos):
        """Take a box and move it along the bar with a """
        # recalculate the position on the bar
        pos = box.position[0] / math.cos(self.bar.angle)
        pos += delta_pos
        return self.placeBox(box, pos)

    def resetState(self):
        """Resets and returns the current values of the state"""
        self.state = [self.boxA.position[0], self.boxA.position[1],
                      self.boxB.position[0], self.boxB.position[1],
                      self.bar.angle, self.bar.angularVelocity]
        return self.state

    def step(self, action):
        state, _, _, _ = self.internal_step(action)
        done = False
        for _ in range(50):
            if not done:
                state, reward, done, info = self.internal_step()
            else:
                return state, reward, done, info
        return state, reward, done, info

    def internal_step(self, action=None):
        """Simulates the program with the given action and returns the observations"""
        # Don't do anything if the setting's Hz are <= 0
        hz = 60.
        velocityIterations = 8
        positionIterations = 3
        velocityIterations *= 1
        positionIterations *= 1


        self.counter += 1

        if hz > 0.0:
            timeStep = 1.0 / hz
        else:
            timeStep = 0.0

        # check if test failed --> reward = -1
        if (abs(self.bar.angle) > 0.390
                or self.boxA.position[0] > 0
                or self.boxB.position[0] < 0):
            state = self.resetState().copy()
            self.reset()
            self.render()
            return state, -1, True, {}

        # check if no movement anymore
        if self.state[5] == 0.0:
            # check if time's up
            if self.counter > WAITINGITERATIONS:
                # self.render()
                self.state = self.resetState()
                return self.state, 1, True, {}
        else:  # no movement --> reset counter
            self.counter = 0

        # catch special case that no action was executed
        if not action:
            self.world.Step(timeStep, velocityIterations,
                            positionIterations)
            self.world.ClearForces()
            self.render()
            self.state = self.resetState()
            return self.state, 0, False, {}

        # extract information from action
        """pos = action["pos"][0, 0]
        box = action["box"]"""

        delta_pos = action["delta_pos"][0, 0]
        box = action["box"]

        def boxesOnScale():
            # TODO: fix
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            return val

        # perform action
        """if box == 0:
            self.boxA = self.placeBox(self.boxA, pos)
        elif box == 1:
            self.boxB = self.placeBox(self.boxB, pos)"""

        if box == 0:
            self.boxA = self.moveAlongBar(self.boxA, delta_pos)
        elif box == 1:
            self.boxB = self.moveAlongBar(self.boxB, delta_pos)

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        self.world.Step(timeStep, velocityIterations, positionIterations)
        self.world.ClearForces()

        self.description = f"{self.joint.angle * 180 / math.pi}°"

        self.state = self.resetState()

        # Calculate reward (Scale in balance?)
        if abs(self.bar.angle) < FAULTTOLERANCE and boxesOnScale():
            reward = 1
        # elif not boxesOnScale():
        #    reward = -1
        else:
            reward = 0

        if boxesOnScale():
            end_of_bar = BARLENGTH * abs(math.cos(self.bar.angle))
            if self.boxA.position[0] > - BOXSIZE or self.boxA.position[0] < - end_of_bar or \
                    self.boxB.position[0] < 0 or self.boxB.position[0] > end_of_bar:
                reward = -1
            else:
                reward = (0.390258252620697 - abs(self.bar.angle)) / 0.390258252620697
        else:
            reward = - 1

        # no movement and in balance --> done
        # velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        done = False

        # placeholder for info
        info = {}

        self.render()

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

        randomPositionA = self.np_random.uniform(-6, -4)
        #sleep(1)
        randomDensityA = self.np_random.uniform(4, 6)
        self.boxA = self.createBox(randomPositionA, self.y, DENSITY, BOXSIZE)

        randomPositionB = self.np_random.uniform(4, 6)
        randomDensityB = self.np_random.uniform(4, 6)
        self.boxB = self.createBox(randomPositionB, self.y, DENSITY, BOXSIZE)

        self.boxes = [self.boxA, self.boxB]

        # rearrange the bar to 0 degree
        self.bar.angle = 0
        self.bar.angularVelocity = 0.

        self.counter = 0

        # Reset the reward
        # TODO

        # return the observation
        self.resetState()
        return self.step(None)[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# More functions can be changed to allow for contact monitoring and such.
# See the other testbed examples for more information.

if __name__ == "__main__":
    main(Scale)
