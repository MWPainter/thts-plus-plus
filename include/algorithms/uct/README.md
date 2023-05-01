# Code Overview - include/algorithms/uct/

TODO:
1. Add links to papers for algorithms
2. Add description of algorithm at high level
3. Integrate small notes made below into final version

UCT nodes subclass from THTS nodes directly
UCT uses custom manager, subclassed from THTSManager
UCT uses custom logger, subclassed from THTSLogger

Implementation of UCT also includes the option for 'Prioritised' UCT, which integrates prior policies into the UCB 
formula, as used by AlphaGo/AlphaZero. Confusingly, both prioritised UCT and polynomial uct are referred to as puct. 
In this code puct == polynomial. If this leads to confusion, can change naming from PuctXXX to PolyUctXXX and so on.

Include link to PROST paper and the auto bias that can use

PUCT subclasses from UCT
PUCT has custom manager, subclassed from UCT manager
PUCT uses the UCT logger

