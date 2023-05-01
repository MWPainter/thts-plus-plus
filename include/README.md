# Code Overview - include/

In each subdirectory in the include folder we have a README file giving an overview of the code contained in it. In 
each header there should be docstrings describing the use of each class/function, including full argument and 
return value descriptions. 

If details on the implementation of a function are needed, there are often (but not always) function level docstrings 
describing the implementation in the `.cc` files in the `src` directory.

Some notes on specific wordings used in this code:
sink state
: The word *sink* is used to refer to a final state in the 



## THTS Overview

Eventually this section will contain a diagram showing the overall THTS schema used. For now a wordy description will 
do. A `ThtsManager` object is used to provide a space for 'global' variables used through the algorithm. This includes 
things like a pointer to the environment and options like the maximum search depth.

A trial consists of the following steps:
1. Each function call will be passed the instance of `ThtsManager` containing the global variables for the algorithm.
2. Sample a context. This context will be passed to every future function call in the trial.
3. Start at the root `decision node`
4. Call `visit` and then `select_action` functions on current `decision node`
5. Select next `chance node` using the selected action from step 4
6. Call `visit` and then `sample_observation` functinos on current `chance node`
7. Select next `decision node` using the sampled observation from step 6
8. Repeat steps 4-7 until a new decision node is made, or until a sink state or the maximum depth is reached
9. Call the `backup` function on every node visited, from bottom to top (i.e. the last node visited will be the first 
    to call `backup`)

Note that some of the options provided in `ThtsManager` may alter how a trial is run slightly, but the above is how a 
trial will run with default options selected.



## algorithms/

This directory contains implementations of specific THTS algorithms. Each implementation will subclass the `ThtsDNode`, 
`ThtsCNode`, `ThtsManager` and `ThtsLogger` types.

## templates/

This directory contains templates for subclassing the `ThtsDNode`, `ThtsCNode`, and `ThtsManager` classes. Each 
contains boilerplate code which can be filled out by using `ctrl-f` to find and replace typenames.

One of the main uses of these templates is to use custom `State` and `Action` types, and use boilerplate code to 
implement the `XXX_itfc` functions (i.e. the interface versions of the functions).

Note that each of these `.h` files contains both the `.h` and `.cpp` portions of the template.

## helper.h

Helper functions. `helper.h` just contains a default zero heuristic function.

## helper_templates.h

More helper functions, but these ones are templated. Things to do with hashing, sampling and pretty printing.

Originally I wanted to keep the interfaces and implementations seperated, which is pretty hard with templated code. The 
current solution involves implementing the templated functions in a `.cc` file, and having the makefile only compile 
`.cpp` files. All of the templated code is actually contained in the header file, as the last line of 
`helper_templates.h` is `#include "helper_templates.cc"`.

## mc_eval.h

Classes to help run a Monte-Carlo evaluation of a tree.

## thts_chance_node.h

Defines the base chance node type `ThtsCNode`. Defines the THTS interface that subclasses need to implement, and 
provides additional utility functions.

## thts_decision_node.h

Defines the base chance node type `ThtsDNode`. Defines the THTS interface that subclasses need to implement, and 
provides additional utility functions. Most of these should be 

One nuance is the create child functions. The `create_child_node_itfc` is the interface that should be used to create 
a child node. as it handles the logic to maintain the map of `children` nodes, and handles any logic with respect to 
transposition tables. The `create_child_node_helper_itfc` function should just construct a child node and return a 
smart pointer to that node.

In the templated subclass in `templates/thts_decision_node_template.h` there are four functions related to creating 
child nodes. The ones that should be called to create a child node are `create_child_node` or `create_child_node_itfc`. 
The functions are called in the following order: `create_child_node` -> `create_child_node_itfc` -> 
`create_child_node_helper_itfc` -> `create_child_node_helper`. The purpose of each

## thts_env_context.h

`ThtsEnvContext` implements a dictionary datatype, keying from string objects to arbitrary data types (i.e. 
`std::shared_ptr<void>`). This can also be subclasses if more specific behaviour is required by any algorithm or 
environment.

## thts_manager.h

The `ThtsManager` class provides a 'global' space to store variables to be used a THTS algorithm, and defines options 
to control some specifics on how a trial runs. 

## thts.h

Defines a class called `ThtsPool`, which is a specialised version of a ThreadPool, where each thread will run trials 
continuously until some end condition is met (either max time or max number of trials reached). The constructor takes a 
root node and a manager object. The root node type will specify the specific algorithm that will be used, and 
the manager object contains the environment to plan for, and specifies options for thts to use.

To run the Thts algorithm only needs one main funciton:
- `run_trials`, which signals the `ThtsPool` to start running trials, and specifies how many trials to run and/or a max 
    time to run for. By default this call is blocking
- `join`, causes the current thread to wait for a `run_trials` call to be completed, which may be usefule with 
    non-blocking calls to `run_trials`

The trials run follow the description from the 'THTS Overview' section. As the `ThtsPool` runs trials in a 
multithreaded environment, the decision and chance nodes are locked around any functions calls.

