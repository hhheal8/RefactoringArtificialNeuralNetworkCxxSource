# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build

# Include any dependencies generated for this target.
include CMakeFiles/ArtificialNeuralNetworkCpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ArtificialNeuralNetworkCpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make

CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make
CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/includes_CXX.rsp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj: ../main.cpp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj -MF CMakeFiles\ArtificialNeuralNetworkCpp.dir\main.cpp.obj.d -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\main.cpp.obj -c C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\main.cpp

CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\main.cpp > CMakeFiles\ArtificialNeuralNetworkCpp.dir\main.cpp.i

CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\main.cpp -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\main.cpp.s

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/includes_CXX.rsp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj: ../sources/layer.cpp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj -MF CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\layer.cpp.obj.d -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\layer.cpp.obj -c C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\layer.cpp

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\layer.cpp > CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\layer.cpp.i

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\layer.cpp -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\layer.cpp.s

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/includes_CXX.rsp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj: ../sources/matrix.cpp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj -MF CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\matrix.cpp.obj.d -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\matrix.cpp.obj -c C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\matrix.cpp

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\matrix.cpp > CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\matrix.cpp.i

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\matrix.cpp -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\matrix.cpp.s

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/includes_CXX.rsp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj: ../sources/utils/ANN_math.cpp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj -MF CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\utils\ANN_math.cpp.obj.d -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\utils\ANN_math.cpp.obj -c C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\utils\ANN_math.cpp

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\utils\ANN_math.cpp > CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\utils\ANN_math.cpp.i

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\utils\ANN_math.cpp -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\utils\ANN_math.cpp.s

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/includes_CXX.rsp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj: ../sources/neural_network.cpp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj -MF CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neural_network.cpp.obj.d -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neural_network.cpp.obj -c C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\neural_network.cpp

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\neural_network.cpp > CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neural_network.cpp.i

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\neural_network.cpp -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neural_network.cpp.s

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/flags.make
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/includes_CXX.rsp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj: ../sources/neuron.cpp
CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj: CMakeFiles/ArtificialNeuralNetworkCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj -MF CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neuron.cpp.obj.d -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neuron.cpp.obj -c C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\neuron.cpp

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\neuron.cpp > CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neuron.cpp.i

CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\sources\neuron.cpp -o CMakeFiles\ArtificialNeuralNetworkCpp.dir\sources\neuron.cpp.s

# Object files for target ArtificialNeuralNetworkCpp
ArtificialNeuralNetworkCpp_OBJECTS = \
"CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj" \
"CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj" \
"CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj" \
"CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj" \
"CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj" \
"CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj"

# External object files for target ArtificialNeuralNetworkCpp
ArtificialNeuralNetworkCpp_EXTERNAL_OBJECTS =

ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/main.cpp.obj
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/layer.cpp.obj
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/matrix.cpp.obj
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/utils/ANN_math.cpp.obj
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neural_network.cpp.obj
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/sources/neuron.cpp.obj
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/build.make
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/linklibs.rsp
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/objects1.rsp
ArtificialNeuralNetworkCpp.exe: CMakeFiles/ArtificialNeuralNetworkCpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable ArtificialNeuralNetworkCpp.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\ArtificialNeuralNetworkCpp.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ArtificialNeuralNetworkCpp.dir/build: ArtificialNeuralNetworkCpp.exe
.PHONY : CMakeFiles/ArtificialNeuralNetworkCpp.dir/build

CMakeFiles/ArtificialNeuralNetworkCpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\ArtificialNeuralNetworkCpp.dir\cmake_clean.cmake
.PHONY : CMakeFiles/ArtificialNeuralNetworkCpp.dir/clean

CMakeFiles/ArtificialNeuralNetworkCpp.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build C:\Users\ictgi\Desktop\ArtificialNeuralNetworkCxxSource\build\CMakeFiles\ArtificialNeuralNetworkCpp.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ArtificialNeuralNetworkCpp.dir/depend

