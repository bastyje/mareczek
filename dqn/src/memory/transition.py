from collections import namedtuple


Transition = namedtuple(
    typename='Transition',
    field_names=('state', 'action', 'reward', 'next_state', 'done'))