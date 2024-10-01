import itertools
from player import *
import itertools

# Create instances of the class
instance1 = AlwaysDefect()
instance2 = AlwaysCooperate()
instance3 = LearningPlayer()

# Put the instances into a list
instances = [instance1, instance2, instance3]

# Generate combinations with replacement
combinations = list(itertools.combinations_with_replacement(instances, 2))

# Print the combinations
for combo in combinations:
    print(combo)