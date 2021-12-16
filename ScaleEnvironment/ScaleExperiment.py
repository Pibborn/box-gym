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
BARLENGTH = 15

FAULTTOLERANCE = 0.001  # for the angle of the bar
ANGLE_TRESHHOLD = 0.80

WAITINGITERATIONS = 20  # maximum iterations to wait per episode
MAXITERATIONS = 1000

def rescale_movement(original_interval, value, to_interval=(-BARLENGTH, +BARLENGTH)):
    a, b = original_interval
    c, d = to_interval
    return c + ((d-c) / (b-a)) * (value-a)

class ScaleExperiment(Framework, gym.Env):
    """You can use this class as an outline for your tests."""
    name = "ScaleExperiment"  # Name of the class to display

    def __init__(self, rendering = True, randomness = True):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(ScaleExperiment, self).__init__(rendering)

        self.seed()

        # Initialize all of the objects
        self.y, L, a, b = 6.0 + BOXSIZE, 12.0, 1.0, 2.0

        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        self.rendering = rendering
        self.randomness = randomness

        #########################################################################
        limit1, limit2 = BARLENGTH - 2 * BOXSIZE, 2 * BOXSIZE
        self.action_space = Dict({
            "box1_pos": Box(low=-limit1, high=-limit2, shape=(1, 1), dtype=float),
            "box2_pos": Box(low=limit2, high=limit1, shape=(1, 1), dtype=float)
        })

        self.observation_space = Dict(spaces={
            "pos1": Box(low=-20., high=20., shape=(1,), dtype=float),
            "pos2": Box(low=-20., high=20., shape=(1,), dtype=float),
            "angle": Box(low=-390258252620697, high=390258252620697, shape=(1,), dtype=float),
            # angular velocity of the bar, negative: moves to the right, positive: moves to the left
            "vel": Box(low=-2., high=2., shape=(1,), dtype=float),
            "density1": Box(low=4., high=6., shape=(1,), dtype=float),
            "density2": Box(low=4., high=6., shape=(1,), dtype=float),
        })

        # setting up the objects on the screen
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxes = []

        # create Box A
        startingPositionA = - BARLENGTH - 3
        if randomness:
            randomDensityA = 4. + 2 * random.random()  # between 4 and 6
            self.boxA = self.createBox(pos_x=startingPositionA, pos_y=BOXSIZE, density=randomDensityA, boxsize=BOXSIZE)
        else:
            self.boxA = self.createBox(pos_x=startingPositionA, pos_y=BOXSIZE, density=DENSITY, boxsize=BOXSIZE)

        startingPositionB = BARLENGTH + 3
        if randomness:
            randomDensityB = 4. + 2 * random.random()
            self.boxB = self.createBox(pos_x=startingPositionB, pos_y=BOXSIZE, density=randomDensityB, boxsize=BOXSIZE)
        else:
            self.boxB = self.createBox(pos_x=startingPositionB, pos_y=BOXSIZE, density=DENSITY, boxsize=BOXSIZE)

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

        pos1 = self.boxA.position[0] / math.cos(self.bar.angle)
        pos2 = self.boxB.position[0] / math.cos(self.bar.angle)
        self.state = [pos1, pos2,
                      self.bar.angle, self.bar.angularVelocity,
                      DENSITY, DENSITY]

    def ConvertScreenToWorld(self, x, y):
        """
        Returns a b2Vec2 indicating the world coordinates of screen (x,y)
        """
        return Vec2((x + self.viewOffset.x) / self.viewZoom,
                    ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def createBox(self, pos_x, pos_y=None, density=DENSITY, boxsize=BOXSIZE):
        """Create a new box on the screen
        Input values: position as x and y coordinate, density and size of the box"""
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
            except Exception as e:
                print(e)
        self.boxes = []
        #return

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
        pos1 = self.boxA.position[0] / math.cos(self.bar.angle)
        pos2 = self.boxB.position[0] / math.cos(self.bar.angle)
        self.state = [pos1, pos2,
                      self.bar.angle, self.bar.angularVelocity,
                      self.state[4], self.state[5]]  # densities cannot be accessed through the box object ...
        return self.state

    def step(self, action):
        """Actual step function called by the agent"""
        done = False
        while not done:
            self.state, reward, done, info = self.internal_step(action)
            action = None
        return self.state, reward, done, info

    def internal_step(self, action=None):
        """Simulates the program with the given action and returns the observations"""
        def boxesOnScale():
            """Utility function to check if both boxes are still on the scale"""
            val = len(self.boxA.contacts) >= 1 and len(self.boxB.contacts) >= 1 and len(self.bar.contacts) == 2
            return val

        def getReward():
            """Calculates the reward and adds it to the self.reward value"""
            # Calculate reward (Scale in balance?)
            if boxesOnScale():
                # both boxes on one side: negative reward
                if (self.boxA.position[0] < 0 and self.boxB.position[0] < 0) \
                        or (self.boxA.position[0] > 0 and self.boxB.position[0] > 0):
                    reward = - (0.390258252620697 - abs(self.bar.angle)) / 0.390258252620697
                    self.timesteps -= 2
                # box on balance
                elif abs(self.bar.angle) < FAULTTOLERANCE and boxesOnScale():
                    reward = 1
                else:
                    reward = (0.390258252620697 - abs(self.bar.angle)) / 0.390258252620697
                self.reward += reward
            else:   # one or both boxes not on the scale
                reward = - 1
            return reward

        # Don't do anything if the setting's Hz are <= 0
        hz = 60.
        velocityIterations = 8
        positionIterations = 3
        velocityIterations *= 1
        positionIterations *= 1

        if hz > 0.0:
            timeStep = 1.0 / hz
        else:
            timeStep = 0.0

        self.counter += 1
        self.timesteps += 1

        # check if test failed --> return reward
        if (abs(self.bar.angle) > 0.390
                #or self.boxA.position[0] > 0
                #or self.boxB.position[0] < 0
                or self.timesteps > MAXITERATIONS):
            state = self.resetState().copy()
            #reward = self.reward / self.timesteps
            reward = self.timesteps
            self.render()
            self.reset()
            return state, reward, True, {}

        # check if no movement anymore
        if self.state[3] == 0.0 and boxesOnScale():
            # check if time's up
            if self.counter > WAITINGITERATIONS:
                # self.render()
                state = self.resetState()
                print("Match:", self.boxA.position[0], self.boxB.position[0], self.bar.angle, getReward())
                reward = getReward()
                self.reset()
                return state, 2 * MAXITERATIONS, True, {}
        else:  # no movement --> reset counter
            self.counter = 0

        # catch special case that no action was executed
        if not action:
            self.world.Step(timeStep, velocityIterations,
                            positionIterations)
            self.world.ClearForces()
            self.render()
            getReward()
            reward = self.timesteps
            #reward = self.reward / self.timesteps
            self.state = self.resetState()
            return self.state, reward, False, {}

        # extract information from action
        box1_pos = action["box1_pos"][0, 0]
        box2_pos = action["box2_pos"][0, 0]

        # perform action
        self.boxA = self.placeBox(self.boxA, box1_pos)
        self.boxB = self.placeBox(self.boxB, box2_pos)

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        self.world.Step(timeStep, velocityIterations, positionIterations)
        self.world.ClearForces()

        self.description = f"{self.joint.angle * 180 / math.pi}°"

        self.state = self.resetState()

        # Calculate reward (Scale in balance?)
        getReward()
        reward = self.timesteps
        #reward == self.reward / self.timesteps

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

    def render(self, mode="human"):
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

        startingPositionA = - BARLENGTH - 3
        if self.randomness:
            randomDensityA = 4. + 2 * random.random()  # between 4 and 6
            self.boxA = self.createBox(pos_x=startingPositionA, pos_y=BOXSIZE, density=randomDensityA, boxsize=BOXSIZE)
        else:
            self.boxA = self.createBox(pos_x=startingPositionA, pos_y=BOXSIZE, density=DENSITY, boxsize=BOXSIZE)

        startingPositionB = BARLENGTH + 3
        if self.randomness:
            randomDensityB = 4. + 2 * random.random()
            self.boxB = self.createBox(pos_x=startingPositionB, pos_y=BOXSIZE, density=randomDensityB, boxsize=BOXSIZE)
        else:
            self.boxB = self.createBox(pos_x=startingPositionB, pos_y=BOXSIZE, density=DENSITY, boxsize=BOXSIZE)

        self.boxes = [self.boxA, self.boxB]

        # rearrange the bar to 0 degree
        self.bar.angle = 0
        self.bar.angularVelocity = 0.

        # Reset the reward and the counters
        self.counter = 0
        self.timesteps = 0
        self.reward = 0

        # return the observation
        self.resetState()
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# More functions can be changed to allow for contact monitoring and such.
# See the other testbed examples for more information.

if __name__ == "__main__":
    main(ScaleExperiment)
