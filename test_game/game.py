import numpy as np

ACTIONS = ["NOOP", "RIGHT", "LEFT", "UP", "DOWN"]
(NOOP, RIGHT, LEFT, UP, DOWN) = range(len(ACTIONS))


def action_name(action):
    return ACTIONS[action]


def print_sequence(actions, observed):
    print '== BEGIN =='
    for a, o in zip(actions, observed):
        print '%5s' % action_name(a), list(o)
    print '== END ===='


def produce_batches(game, batch_size, sequence_size):
    while True:
        actions = []
        observed = []
        for b in xrange(batch_size):
            game.reset()
            a, o = game.sample_sequence(sequence_size)
            actions.append(a)
            observed.append(o)
        actions = np.array(actions, dtype=np.int32)
        observed = np.array(observed, dtype=np.float32)
        yield actions, observed

        
class Game(object):
    def __init__(self, width, classes):
        self.width = width
        self.classes = classes
        self.reset()
        
    def reset(self):
        self.counters = np.random.randint(0, self.classes, size=self.width)
        self.position = 0
    
    def render(self):
        ret = np.zeros(self.classes)
        ret[self.counters[self.position]] = 1.
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
            
    def sample_sequence(self, n):
        actions = []
        observed = []
        for i in xrange(n):
            action = np.random.randint(0, len(ACTIONS))
            actions.append(action)
            self.take_action(action)
            observed.append(self.render())
        return actions, observed
