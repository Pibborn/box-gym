from time import sleep

import gym
from gym.spaces import Box
import pygame
from Box2D import b2Vec2
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, edgeShape, fixtureDef)
import numpy as np
from gym.utils import seeding
from environments.EnvironmentInterface import EnvironmentInterface, rescale_movement


# copy paste from EnvironmentInterface file
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS


class Ball:
    def __init__(self, world, x, y, radius=1.0, density=1.0):
        self.world = world

        self.x, self.y = x, y
        self.radius = radius
        self.density = density

        self.ball = self.create_ball()
        pass

    def create_ball(self):
        self.ball = self.world.CreateDynamicBody(
            position=(self.x, self.y),
            fixtures=fixtureDef(shape=circleShape(radius=self.radius),
                                density=self.density, friction=0.4,
                                restitution=0.6),  # restitution makes the ball bouncy
            #linearVelocity=b2Vec2(self.delta_x, self.delta_y),
            userData=None,
        )
        return self.ball

    def get_state(self):
        # height = self.ball.position[1]
        height = self.ball.position[1] - self.radius
        velocity = self.ball.linearVelocity.y
        return np.array([height, velocity, self.radius, self.density])

    def get_world(self):
        return self.world

    def __del__(self):
        try:
            self.world.DestroyBody(self.ball)
        except:
            pass
        del self


class FreeFallEnvironment(EnvironmentInterface):
    def __init__(self, seed=None, normalize=False, rendering=True, raw_pixels=False, random_ball_size=True,
                 random_density=False, random_gravity=False, random_ball_height=True, use_seconds=False):
        # should the size of the ball be fixed or random?
        self.random_ball_size = random_ball_size
        # should the density of the ball be fixed or random?
        self.random_density = random_density
        # should the world's gravity be random or not?
        self.random_gravity = random_gravity
        # start height of the basketball random or not?
        self.random_ball_height = random_ball_height
        # use seconds as time measure or count time steps
        self.use_seconds = use_seconds

        # create Environment with interface
        super().__init__(seed=seed, normalize=normalize, rendering=rendering, raw_pixels=raw_pixels)

        self.ball = None
        self.last_height = None
        self.starting_height = None
        self.starting_distance = None
        self.last_velocity = None
        self.prediction = None  # for reward determination later
        self.passed_time_steps = 0

        if rendering:
            pygame.display.set_caption('Free Fall Environment')

        # action space and observation space
        self.max_time_steps = 300
        self.max_time = self.max_time_steps * TIME_STEP if self.use_seconds else self.max_time_steps

        self.action_space = Box(low=0, high=self.max_time if not self.normalize else 1, shape=(1,))

        # first, we have the ball information: height, velocity, radius, density
        # then, we have the time information and the gravity
        self.observation_space = Box(low=np.array([0, 0, 0.5, 4, 0, 3] if not self.normalize
                                                  else [0 for _ in range(6)]),
                                     high=np.array([self.world_height, 10, 1.5, 6, self.max_time, 20] if not self.normalize
                                                   else [1 for _ in range(6)]))

        self.reset()

    def success(self):
        # idea: check, if the ball either is height 0 or its last height was lower than the current one
        # if so, the ball must have touched the ground
        # self.state = self.updateState()
        _, ball_height = self.ball.ball.position

        self.passed_time_steps += 1

        """if self.last_height < ball_height or ball_height == 0:
            print(self.last_height, ball_height)
            sleep(100)
            return True"""
        if len(self.ball.ball.contacts) > 0:
            return True
        self.last_height = ball_height
        self.last_velocity = self.state[1]
        return False

    def getReward(self):
        if self.use_seconds:
            time_passed = self.passed_time_steps * TARGET_FPS
        else:
            time_passed = self.passed_time_steps
        time_predicted = self.prediction

        time_difference = abs(time_predicted / time_passed - 1)
        if time_difference < 0.01:  # 1% or better
            return 150
        elif time_difference < 0.02:
            return 100
        elif time_difference < 0.03:
            return 80
        elif time_difference < 0.04:
            return 60
        elif time_difference < 0.05:
            return 40
        elif time_difference < 0.10:
            return 10
        return 0


    def testFailed(self):
        # test cannot fail
        return False

    def printSuccess(self):
        decimal_places = 2
        prediction = self.prediction
        if self.use_seconds:
            time_passed = self.passed_time_steps * TARGET_FPS
        else:
            time_passed = self.passed_time_steps

        difference = self.prediction / time_passed - 1

        if abs(difference) <= 0.03:

            return_message = f"Success!\t " \
                             f"| Gravity: {f'%.{decimal_places}f' % abs(self.world.gravity[1])} m/s^2 " \
                             f"| Start height: {f'%.{decimal_places}f' % self.starting_height} " \
                             f"| Prediction: {f'%.{decimal_places}f' % prediction} " \
                             f"| Time passed: {f'%.{decimal_places}f' % time_passed} " \
                             f"| Difference: {f'%.{decimal_places}f' % difference}"
        else:
            return_message = ""
        return return_message

    def performAction(self, action):
        #max_time = self.max_time_steps * TIME_STEP if self.use_seconds else self.max_time_steps
        if self.normalize:
            action = rescale_movement(np.array([0, 1]), action, np.array([0], [self.max_time]))
        # save the predicted time
        self.prediction = action
        """print(self.prediction)
        sleep(10)"""
        pass

    def updateState(self):
        if not (self.ball):  # check if basket and ball exist in the world
            return None
        self.state = np.concatenate((self.ball.get_state(), np.array([self.passed_time_steps, abs(self.world.gravity[1])])))
        if self.normalize:
            self.normalized_state = self.rescaleState(np.concatenate(
                (self.ball.get_state(), self.passed_time_steps, abs(self.world.gravity[1]))))
        return self.state

    def rescaleState(self, state=None):
        if state is None:
            state = self.updateState()
        if self.raw_pixels:
            self.normalized_state = rescale_movement([0, 255], self.state, [0., 1.])

        else:
            max_time = self.max_time_steps * TARGET_FPS

            self.normalized_state = rescale_movement(
                # [self.observation_space.low, self.observation_space.high],
                np.array([[0, 0, 0.5, 4, 0, 3],
                          [self.world_height, 10, 1.5, 6, max_time, 20]]),
                state,
                np.array([np.array([0 for _ in range(6)]), np.array([1 for _ in range(6)])]))
        return self.normalized_state

    def reset(self):
        # delete old ball
        if self.state is not None:
            del self.ball

        # create new world with new gravity
        gravity = self.np_random.uniform(3, 20) if self.random_gravity else 9.80665
        self.world = self.createWorld(walls=0, gravity=-gravity)

        # reset the ball
        if self.random_ball_height:
            # ball_x = self.np_random.uniform(0.1 * self.world_width, 0.5 * self.world_width)
            ball_x = self.world_width * 0.5
            ball_y = self.np_random.uniform(0.4 * self.world_height, 0.7 * self.world_height)
        else:
            ball_x, ball_y = self.world_width * 0.4, self.world_height * 0.6
        ball_radius = self.np_random.uniform(0.5, 1.5) if self.random_ball_size else 1.0
        ball_density = self.np_random.uniform(4, 6) if self.random_density else 5.0
        self.ball = Ball(world=self.world, x=ball_x, y=ball_y, radius=ball_radius, density=ball_density)

        self.world = self.ball.get_world()

        self.last_height = ball_y
        self.starting_height = ball_y
        self.starting_distance = ball_y - ball_radius
        self.last_velocity = 0
        self.prediction = None
        self.passed_time_steps = 0

        self.state = self.updateState()
        return self.rescaleState() if self.normalize else self.state
