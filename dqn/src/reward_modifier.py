def space_invaders(action, reward):
    return reward if action == 1 and reward > 0 else -10