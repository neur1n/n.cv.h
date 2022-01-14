<!------------------------------------------------------------------ HEADER -->
<pre align="center">
  ▄▀▀▀▀▄▀▀▄ █▀▀▀▀▀▀▀█ █▀▀▀█▀▀▀█    ▄▀▀▀▀▀▀▀▄ █▀▀▀█▀▀▀█    █▀▀▀█▀▀▀█
  █    █  █ █   ▄▄▄▄█ █   █   █    █   ▄▖  █ █   █   █    █   █   █
  █  █ █  █ █       █ █   █   █    █   █████ █   █   █    █       █
  █  █    █ █   ▀▀▀▀█ █   ▀   █    █   ▀▘  █  ▀▄   ▄▀     █   █   █
  ▀▄▄▀▄▄▄▄▀ █▄▄▄▄▄▄▄█ ▀▄▄▄▄▄▄▄▀ ██ ▀▄▄▄▄▄▄▄▀    ▀▄▀    ██ █▄▄▄█▄▄▄█
</pre>


<p align="center">
  <img src="https://img.shields.io/badge/C++11-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++11">

  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-mit-green.svg?style=for-the-badge" alt="license-mit">
  </a>
</p>

<p align="center">
  Some C++ utilities for computer vision.
</p>


<!------------------------------------------------------- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


<!--------------------------------------------------------- GETTING STARTED -->
## Getting Started
### Prerequisites
These are tested prerequisites, they might be not mandatory:
- C++ compiler: >= C++11
- CMake: >= 3.18
- Eigen: >= 3.3.7


### Installation
This library contains only one header, `neu.cv.h`. For end users, it is
recommended to grab the file and drop it into your project. For contributors,
you may want to clone the entire repository for testing purpose:

```bash
git clone https://github.com/Neur1n/neu.cv.h.git
```


<!------------------------------------------------------------------- USAGE -->
## Usage
For CMake users, download and put `neu.cv.h` into your project and add a line
to to `INCLUDE_DIRECTORIES()` in your project's main `CMakeLists.txt`:

```cmake
INCLUDE_DIRECTORIES(
  path/to/directory/containing/neu.cv.h
)
```

Additionally, add some dependency configurations required by `neu.cv.h` to
`CMakeLists.txt` (you may refer to the `CMakeLists.txt` in this repository).


<!----------------------------------------------------------- DOCUMENTATION -->
## Documentation
TODO


<!----------------------------------------------------------------- LICENSE -->
## License
Distributed under the MIT license. See [LICENSE](LICENSE) for more information.
