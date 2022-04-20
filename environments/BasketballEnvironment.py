from abc import ABC
from time import sleep

import gym
import pygame
from Box2D import b2Vec2
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, edgeShape, fixtureDef)
import numpy as np
from gym.utils import seeding
from environments.EnvironmentInterface import EnvironmentInterface, rescale_movement


class Ball:
    def __init__(self, world, x, y, angle, velocity, radius=1.0, density=1.0, delta_x=0, delta_y=0):
        self.world = world

        self.x, self.y = x, y
        self.angle = angle
        self.velocity = velocity
        self.radius = radius
        self.density = density

        self.delta_x, self.delta_y = delta_x, delta_y

        self.ball = self.create_ball()
        pass

    def create_ball(self):
        self.ball = self.world.CreateKinematicBody(
            position=(self.x, self.y),
            fixtures=fixtureDef(shape=circleShape(pos=(self.x, self.y), radius=self.radius),
                                density=self.density),
            linearVelocity=b2Vec2(self.delta_x, self.delta_y),
            userData=None,
        )
        return self.ball

    def get_state(self):
        position1 = self.ball.position[0]
        position2 = self.ball.position[1]
        angle = self.ball.angle
        velocity = float(self.ball.angularVelocity)
        return np.array([position1, position2, angle, velocity, self.radius, self.density])

    def __del__(self):
        try:
            self.world.DestroyBody(self.ball)
        except:
            pass
        del self


class Basket:
    def __init__(self, world, x=38, y=20, radius=1.5, width=0.2):
        self.world = world
        self.x = x
        self.y = y
        self.radius = radius
        self.width = width

        self.basket = self.create_basket()

    def _create_vertices(self):
        x1, y1 = self.x - self.radius - self.width, self.y
        x2, y2 = x1, self.y - 2 * self.radius - self.width
        x3, y3 = self.x + self.radius + self.width, y2
        x4, y4 = x3, self.y + 2 * self.radius
        x5, y5 = self.x + self.radius, y4
        x6, y6 = x5, self.y - 2 * self.radius
        x7, y7 = self.x - self.radius, y6
        x8, y8 = x7, y1
        # print([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8)])
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8)]

    def create_basket(self):
        p1, p2, p3, p4, p5, p6, p7, p8 = self._create_vertices()
        rectangle1 = [p1, p2, p7, p8]
        rectangle2 = [p2, p3, p6, p7]
        rectangle3 = [p3, p4, p5, p6]
        self.basket = self.world.CreateStaticBody(
            position=(self.x, self.y),
            # fixtures=fixtureDef(shape=polygonShape(vertices=self._create_vertices()), density=100),
            fixtures=[fixtureDef(shape=polygonShape(vertices=rectangle1), density=100),
                      fixtureDef(shape=polygonShape(vertices=rectangle2), density=100),
                      fixtureDef(shape=polygonShape(vertices=rectangle3), density=100)],
            userData=None,
        )
        return self.basket

    def get_state(self):
        return np.array([self.x, self.y, self.radius])

    def __del__(self):
        self.world.DestroyBody(self.basket)
        del self


class BasketballEnvironment(EnvironmentInterface):
    def __init__(self, seed=None, normalize=False, rendering=True, raw_pixels=False):
        super().__init__(seed=seed, normalize=normalize, rendering=rendering, raw_pixels=raw_pixels)

        # gym action and observation spaces
        self.ball = None
        self.basket = None

        if rendering:
            pygame.display.set_caption('Basketball Environment')

        # action space and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]))
        # self.action_space = gym.spaces.Box(low=np.array([-np.pi, 0]), high=np.array([np.pi, 100]))
        # first, we have the ball information: x coordinate, y coordinate, angle, velocity, radius/size, density
        # then, we have the basket information: x coordinate, y coordinate (both of the center of the ring), radius of the ring
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5]),
                                                high=np.array([self.world_width, self.world_height, np.pi, 10, 1.5, 6,
                                                               self.world_width, self.world_height, 3]))

        self.reset()

    def success(self):
        # idea: check, if box x coordinates in the range of the position of the basket
        # and if the ball is completely under the line
        # self.state = self.updateState()
        ball_x, ball_y = self.ball.ball.position
        """if (self.basket.x - self.basket.radius <= ball_x <= self.basket.x + self.basket.radius and
                self.basket.y - 2 * self.basket.radius + self.ball.radius <= ball_y <= self.basket.y - self.ball.radius):
            pass"""

        if (self.basket.x - self.basket.radius <= self.ball.x - self.ball.radius
                and self.basket.x + self.basket.radius >= self.ball.x + self.ball.radius
                and self.basket.y >= self.ball.y + self.ball.radius
                and self.basket.y - 2 * self.basket.radius <= self.ball.y - self.ball.radius):
            return True
        return False

    def getReward(self):
        if self.success():
            return 100
        if self.ball in self.basket.basket.contacts:
            return 10
        else:
            return 0

    def performAction(self, action):
        delta_x, delta_y = float(action[0]), float(action[1])
        x, y = self.ball.ball.position
        density = self.ball.density
        radius = self.ball.radius
        #self.world.DestroyBody(self.ball.ball)
        del self.ball
        self.ball = Ball(self.world, x=x, y=y, angle=0, velocity=0, radius=radius, density=density, delta_x=delta_x,
                         delta_y=delta_y)
        # self.ball.ball.linearVelocity = b2Vec2(delta_x, delta_y)
        pass

    def updateState(self):
        self.state = np.concatenate((self.ball.get_state(), self.basket.get_state()))
        return self.state

    def rescaleState(self, state=None):
        if not state:
            state = self.updateState()
        if self.raw_pixels:
            normalized_state = rescale_movement([0, 255], self.state, [0., 1.])

        else:
            normalized_state = rescale_movement([self.observation_space.low, self.observation_space.high], self.state,
                                                [0, 1])
        return normalized_state

    def reset(self):
        if self.state is not None:
            del self.ball, self.basket
            """del self.basket
            self.world.DestroyBody(self.ball)"""
        # reset the ball
        ball_x = self.np_random.uniform(0.1 * self.world_width, 0.9 * self.world_width)
        ball_y = self.np_random.uniform(0.2 * self.world_height, 0.8 * self.world_height)
        ball_radius = self.np_random.uniform(0.5, 1.5)
        ball_density = self.np_random.uniform(4, 6)
        self.ball = Ball(self.world, x=ball_x, y=ball_y, angle=0, velocity=0, radius=ball_radius, density=ball_density)

        # reset the basket
        basket_y = self.np_random.uniform(0.2 * self.world_height, 0.8 * self.world_height)
        basket_radius = self.np_random.uniform(ball_radius, 2 * ball_radius)
        basket_width = self.np_random.uniform(0.1, 1.0)
        basket_x = self.world_width - basket_width - basket_radius
        self.basket = Basket(self.world, x=basket_x, y=basket_y, radius=basket_radius, width=basket_width)
        return


env = BasketballEnvironment(123, False)
env.reset()
env.rescaleState()
print(env.state)
env.reset()
while True:
    env.render()
    oldstate = env.state
    env.step([1, 1])
    # print(env.state == oldstate)
    # env.reset()
    # sleep(1)
