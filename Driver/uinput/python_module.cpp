extern "C" {
#include "uinput_api.h"
}

#include <boost/python.hpp>

BOOST_PYTHON_MODULE(uinput_api) {
   using namespace boost::python;
   def("test", test);
}