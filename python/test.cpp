/**
 * Test script
 */
#include <Python.h>

#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <thread>
#include <mutex>

// Defines an error object that our functions can use to throw exceptions of type thts.Error in python
// Created in PyInit_thts later
static PyObject* thts_error;

static PyObject* spam_system(PyObject* self, PyObject* args) {
    const char* command;
    int status;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    status = system(command);
    if (!WIFEXITED(status)) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", WEXITSTATUS(status));
}

static PyObject* spam_check_system(PyObject* self, PyObject* args) {
    const char* command;
    int status;
    int exit_status;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    status = system(command);
    if (!WIFEXITED(status)) {
        return PyErr_Format(thts_error, "Command could not be executed");
    }
    exit_status = WEXITSTATUS(status);
    if (exit_status != 0) {
        return PyErr_Format(
            thts_error, "Command returned non-zero exit status %d", exit_status);
    }
    Py_RETURN_NONE;
}

static void run_py_fn(PyObject* fn, std::mutex* lock) {
    std::cout << "in thread, about to call py function" << std::endl;
    std::lock_guard lg(lock);
    PyObject* _unused_1 = PyObject_CallObject(fn, NULL);
}

static PyObject* test_py_parallel(PyObject* self, PyObject* args) {
    PyObject* foo_fn;
    PyObject* bar_fn;

    // todo, want to use: PyArg_ParseTupleAndKeywords() to use keywords
    if (!PyArg_ParseTuple(args, "OO", &foo_fn, &bar_fn)) {
        PyErr_SetString(PyExc_TypeError, "Expecting two arguments");
        return NULL;
    }

    if (!PyCallable_Check(foo_fn) || !PyCallable_Check(bar_fn)) {
        PyErr_SetString(PyExc_TypeError, "Expecting foo and bar to be callable");
        return NULL;
    }

    Py_INCREF(foo_fn);
    Py_INCREF(bar_fn);

    std::cout << "about to run threads" << std::endl;

    bool multiprocess = true;

    if (multiprocess) {
        std::mutex l;
        pid_t pid = fork();
        if (pid > 0) {
            run_py_fn(bar_fn, &l);
        } else {
            run_py_fn(foo_fn, &l);
            // exit child thread
            std::cout << "exiting child thread, hopefully it cleans up nicely *shrugs*" << std::endl;
            exit(0);
        }
    }

    if (!multiprocess) {
        std::mutex l;
        std::thread foo_thread(run_py_fn, foo_fn, &l);
        std::thread bar_thread(run_py_fn, bar_fn, &l);
        foo_thread.join();
        bar_thread.join();
    }

    std::cout << "finished run threads" << std::endl;

    Py_DECREF(foo_fn);
    Py_DECREF(bar_fn);

    Py_INCREF(Py_None);
    return Py_None;
}



/**
 * PyInit_thts
 * 
 * Initialises the python module. First we define an array of method definitions and the module definition:
 * - thts_methods: 
 *      A list of (method_name (C string), method (C function ptr), python_args (usually METH_VARARGS), 
 *      description (C string)) tuples that define each method, null terminated.
 * - thts_module:
 *      A tuple defining the module object. It contains the following: 
 *          python object header, 
 *          the name of the module, 
 *          any module documentation string, 
 *          the 'size of per-interpreter state of the module, or -1 if the module keeps state in global variables' (I 
 *              think this will be -1 for most cases, as objects either used internally (i.e. globally?), or is 
 *              returned as python object, but the module), and finally
 *          'thts_methods' passing the array of method definitions.
 * 
 * Once thts_module C object exists, the PyInit_thts method pretty much runs 'PyModule_Create' to add the module to 
 * pythons list of modules in sys.modules.
*/
static PyMethodDef thts_methods[] = {
    {
        "system",
        spam_system,
        METH_VARARGS,
        "babExecute a system command and return its exit status.\n\n"
        "If the command cannot be executed, return -1.",
    },
    {
        "check_system",
        spam_check_system,
        METH_VARARGS,
        "Execute a system command.\n\n"
        "If the command cannot be executed or returns a non-zero exit status, "
        "raise Error.",
    },
    {
        "test_py_parallel",
        test_py_parallel,
        METH_VARARGS,
        "Testing python can be run in parallel from c++",
    },
    {NULL, NULL, 0, NULL},  // sentinel
};

static PyModuleDef thts_module = {
    PyModuleDef_HEAD_INIT,
    "thts",
    "A thts (mcts) package for python.",
    -1,
    thts_methods,
};

PyMODINIT_FUNC PyInit_thts() {
    PyObject* py_module;

    py_module = PyModule_Create(&thts_module);

    // error checking
    if (py_module == NULL) {
        return NULL;
    }

    // make an exception that the module can throw when something goes wrong
    thts_error = PyErr_NewException("thts.Error", NULL, NULL);
    Py_INCREF(thts_error);
    PyModule_AddObject(py_module, "Error", thts_error);

    // return module
    return py_module;
}