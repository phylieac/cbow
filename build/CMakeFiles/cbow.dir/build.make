# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.14.5/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.14.5/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/panhongyan/cbow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/panhongyan/cbow/build

# Include any dependencies generated for this target.
include CMakeFiles/cbow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cbow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cbow.dir/flags.make

CMakeFiles/cbow.dir/main.cpp.o: CMakeFiles/cbow.dir/flags.make
CMakeFiles/cbow.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/panhongyan/cbow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cbow.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cbow.dir/main.cpp.o -c /Users/panhongyan/cbow/main.cpp

CMakeFiles/cbow.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cbow.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/panhongyan/cbow/main.cpp > CMakeFiles/cbow.dir/main.cpp.i

CMakeFiles/cbow.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cbow.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/panhongyan/cbow/main.cpp -o CMakeFiles/cbow.dir/main.cpp.s

# Object files for target cbow
cbow_OBJECTS = \
"CMakeFiles/cbow.dir/main.cpp.o"

# External object files for target cbow
cbow_EXTERNAL_OBJECTS =

cbow: CMakeFiles/cbow.dir/main.cpp.o
cbow: CMakeFiles/cbow.dir/build.make
cbow: /Users/panhongyan/libtorch/lib/libc10.dylib
cbow: /Users/panhongyan/libtorch/lib/libtorch.dylib
cbow: /Users/panhongyan/libtorch/lib/libc10.dylib
cbow: CMakeFiles/cbow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/panhongyan/cbow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cbow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cbow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cbow.dir/build: cbow

.PHONY : CMakeFiles/cbow.dir/build

CMakeFiles/cbow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cbow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cbow.dir/clean

CMakeFiles/cbow.dir/depend:
	cd /Users/panhongyan/cbow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/panhongyan/cbow /Users/panhongyan/cbow /Users/panhongyan/cbow/build /Users/panhongyan/cbow/build /Users/panhongyan/cbow/build/CMakeFiles/cbow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cbow.dir/depend

