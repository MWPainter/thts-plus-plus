# Code Overview - include/algorithms/ments/

TODO:
1. Add links to papers for algorithms
2. Add description of algorithm at high level
3. Integrate small notes made below into final version

MENTS nodes subclass from THTS nodes directly
MENTS uses custom manager, subclassed from THTSManager
MENTS uses custom logger, subclassed from THTSLogger

MENTS includes a lot of additional options over the original design of MENTS
- mixing prior policy into search policy
- decaying the search temperature (default is to not decay)
- recommending the most visited node


DB-MENTS nodes subclass from MENTS nodes
DB-MENTS uses MENTS manager
DB-MENTS uses logger subclassed from MENTS logger

DB-MENTS provides and additional 



## dents/

Contains DENTS implementation

## rents/

Contains RENTS implementation (MENTS with Relative entropy)

## tents/

Contains TENTS implementation (MENTS with Tsallis entropy)