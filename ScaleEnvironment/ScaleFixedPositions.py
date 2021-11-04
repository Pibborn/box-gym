import math
import random
import gym

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

        # fixed paramters: weight of object A and the positions of both boxes
        self.fixedPositionX1 = - 5
        self.fixedPositionX2 = 6
        self.fixedDensityA = 5.0
        self.fixedBoxSize = 1.0

        # The ground
        self.ground = self.world.CreateStaticBody(
            shapes=[Box2D.b2.edgeShape(vertices=[(-40, 0), (40, 0)])]
        )

        self.boxA = self.world.CreateDynamicBody(
            #position=(-10, y),
            position=(self.fixedPositionX1, self.y),
            fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)), density=self.fixedDensityA, friction=1.),
        )
        self.boxB = self.world.CreateDynamicBody(
            #position=(10, y),
            position=(self.fixedPositionX2, self.y),
            fixtures=fixtureDef(shape=polygonShape(box=(self.fixedBoxSize, self.fixedBoxSize)), density=4.160, friction=1.),
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

        #if (abs(self.bar.angle) - 0.390258252620697) > 0.000000001:
        #    print("done")

        velocities = [self.bar.linearVelocity, self.boxA.linearVelocity, self.boxB.linearVelocity]
        print(all(vel == b2Vec2(0,0) for vel in velocities) and abs(self.bar.angle) < 0.001)


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
        print(self.boxA.position)
        self.boxA.position = (self.fixedPositionX1, self.y)
        self.boxB.position = (self.fixedPositionX2, self.y)

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