# Code Overview - include/algorithms/ments/

TODO:
1. Add links to papers for algorithms
2. Add description of algorithm at high level
3. Integrate small notes made below into final version

DENTS nodes subclass from DB-MENTS nodes
DENTS uses custom manager, subclassed from MentsManager
DENTS doesn't have a custom logger implemented, but should really


DENTS - everything to do with the entropy computed with entropy backups called value_temp, as opposed to the temperature
used in the policy, which is just temp, but could be called policy_temp

value_temp is the temperature used to compute V_soft(s) = V(s) + value_temp * H(s), all other temp parameters refer to 
the temperature used in the energy based policy (and MENTS soft backups if relevant)

Note that if want to run MENTS using empirical values (or a seperated dp value + entropy) then DENTS can be used, with 
the value_temp set to not decay.