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
import gym

import numpy as np
import pygame
from Box2D import b2Color
from pyglet.math import Vec2

from ScaleEnvironment.framework import (Framework, Keys, main)
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)

BOXSIZE = 1.0
DENSITY = 5.0

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
        y, L, a, b = 16.0, 12.0, 1.0, 2.0
        self.pressed = 0 #?
        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxA = self.world.CreateDynamicBody(
            #position=(-10, y),
            position=(random.random() * 7 - 12, y),
            fixtures=fixtureDef(shape=polygonShape(box=(BOXSIZE, BOXSIZE)), density=DENSITY, friction=1.),
        )
        self.boxB = self.world.CreateDynamicBody(
            #position=(10, y),
            position=(random.random() * 7 + 5, y),
            fixtures=fixtureDef(shape=polygonShape(box=(BOXSIZE, BOXSIZE)), density=DENSITY, friction=1.),
        )

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

    def Step(self, settings):
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
        self.description = f"{self.joint.angle * 180/math.pi}°"

        # Placed after the physics step, it will draw on top of physics objects
        self.Print("*** Base your own testbeds on me! ***")

        """delta_x = 0.01
        delta_y =  1

        if self.bar.angle > 0:
            self.boxA.position.x += delta_x
            self.boxA.position.y -= delta_y
        if self.bar.angle < 0:
            self.boxA.position.x -= delta_x"""

        if (abs(self.bar.angle) - 0.390258252620697) > 0.000000001:
            print("done")

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



# More functions can be changed to allow for contact monitoring and such.
    # See the other testbed examples for more information.

if __name__ == "__main__":
    main(Scale)