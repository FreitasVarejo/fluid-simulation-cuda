# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build

# Include any dependencies generated for this target.
include CMakeFiles/serial.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/serial.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/serial.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/serial.dir/flags.make

CMakeFiles/serial.dir/src/main_serial.c.o: CMakeFiles/serial.dir/flags.make
CMakeFiles/serial.dir/src/main_serial.c.o: /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/src/main_serial.c
CMakeFiles/serial.dir/src/main_serial.c.o: CMakeFiles/serial.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/serial.dir/src/main_serial.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/serial.dir/src/main_serial.c.o -MF CMakeFiles/serial.dir/src/main_serial.c.o.d -o CMakeFiles/serial.dir/src/main_serial.c.o -c /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/src/main_serial.c

CMakeFiles/serial.dir/src/main_serial.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/serial.dir/src/main_serial.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/src/main_serial.c > CMakeFiles/serial.dir/src/main_serial.c.i

CMakeFiles/serial.dir/src/main_serial.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/serial.dir/src/main_serial.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/src/main_serial.c -o CMakeFiles/serial.dir/src/main_serial.c.s

# Object files for target serial
serial_OBJECTS = \
"CMakeFiles/serial.dir/src/main_serial.c.o"

# External object files for target serial
serial_EXTERNAL_OBJECTS = \
"/home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build/CMakeFiles/solver.dir/src/solver.c.o"

serial: CMakeFiles/serial.dir/src/main_serial.c.o
serial: CMakeFiles/solver.dir/src/solver.c.o
serial: CMakeFiles/serial.dir/build.make
serial: CMakeFiles/serial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable serial"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/serial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/serial.dir/build: serial
.PHONY : CMakeFiles/serial.dir/build

CMakeFiles/serial.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/serial.dir/cmake_clean.cmake
.PHONY : CMakeFiles/serial.dir/clean

CMakeFiles/serial.dir/depend:
	cd /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build /home/freitaspinhe/Desktop/unicamp/mc970/fluid-simulation-cuda/build/CMakeFiles/serial.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/serial.dir/depend

