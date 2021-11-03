%module python_module
%{
   /* Put header files here or function declarations like below */
   extern char const *test();
%}

extern char const *test();
