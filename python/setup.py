from distutils.core import Extension
from distutils.core import setup

thts_module = Extension('thts',
                    # define_macros = [('MAJOR_VERSION', '0'),
                    #                  ('MINOR_VERSION', '1')],
                    # include_dirs = ['/usr/local/include'],
                    # libraries = ['tcl83'],
                    # library_dirs = ['/usr/local/lib'],
                    sources = ['test.cpp'])#,'uct.cpp'])

setup (name = 'thts',
       version = '0.1',
       description = 'A thts (mcts) package for python.',
       author = 'Michael Painter',
       author_email = 'michaelpainter.1994@gmail.com',
       url = 'https://github.com/MWPainter/thts-plus-plus/',
       long_description = '''
A thts/mcts package for python.

The main purpose of this package is to provide access to faster monte-carlo tree-search (MCTS), and more generally 
trial-based heuristic tree-search (THTS) methods, implemented in C++. It's main advantages are running the tree search 
in a faster language, and providing true multi-treaded implementations not possible in python.
''',
       ext_modules = [thts_module])