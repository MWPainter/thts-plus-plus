# Code Overview - include/algorithms/est/

TODO:
1. Add links to papers for algorithms
2. Add description of algorithm at high level
3. Integrate small notes made below into final version



EST = energy search for trees, at least that's what I'm calling it for now
Its essentially using the energy based policy for action selection
And then DP or average value backups depending on what flavour you want

Currently:
EST nodes are defined ontop of Dents nodes
EST has uses the Dents manager
EST doesn't really have a logger

The implementation essentially ignores the entropy part of the node, so ignores all of the entropy options in the 
manager too