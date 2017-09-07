# Copyright 2017 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple test "game" with a partially observable state.

The state of the game is a 1-D circular buffer of integers, randomly
initialized between 0 and K. The player can perform the following
actions:

  NOOP  - do nothing.
  RIGHT - move right.
  LEFT  - move left.
  UP    - increment (mod K) the integer at the current location.
  DOWN  - decrement (mod K) the integer at the current location.

the game's observations are a one-hot representation of the value at
the current location.

For example, here's a game with width = 4 and K = 3:

Step 0 (initial state).
  Buffer:      [1, 2, 1, 0]
  Position:     ^

Step 1.
  Action:      RIGHT
  Buffer:      [1, 2, 1, 0]
  Position:        ^
  Observation: [0, 0, 1]

Step 2.
  Action:      UP
  Buffer:      [1, 0, 1, 0]
  Position:        ^
  Observation: [1, 0, 0]

Step 3.
  Action:      RIGHT
  Buffer:      [1, 0, 1, 0]
  Position:           ^
  Observation: [0, 1, 0]
"""

import numpy as np

ACTIONS = ["NOOP", "RIGHT", "LEFT", "UP", "DOWN"]
(NOOP, RIGHT, LEFT, UP, DOWN) = range(len(ACTIONS))


class Game(object):

    def __init__(self, width, classes):
        self.width = width
        self.classes = classes
        self.reset()

    def reset(self):
        self.counters = np.random.randint(0, self.classes, size=self.width)
        self.position = 0

    def render(self):
        ret = np.zeros(self.classes, dtype=np.int32)
        ret[self.counters[self.position]] = 1
        return ret

    def take_action(self, action):
        if action == NOOP:
            return
        elif action == RIGHT:
            self.position += 1
        elif action == LEFT:
            self.position -= 1
        elif action == UP:
            self.counters[self.position] += 1
        elif action == DOWN:
            self.counters[self.position] -= 1
        self.position %= self.width
        self.counters[self.position] %= self.classes
