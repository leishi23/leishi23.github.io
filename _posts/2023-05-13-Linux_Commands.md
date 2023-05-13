---
layout: post
title:  "Linux Useful Commands"
date:   2023-05-13
desc: "Linux Useful Commands"
keywords: "Linux"
categories: [Linux]
tags: [Linux]
icon: icon-html
---

# Linux Useful Commands
## _Linux_DebugNotes_03_

[![image](https://fossbytes.com/wp-content/uploads/2020/02/Ubuntu-18.04.4-release.jpg)](https://releases.ubuntu.com/18.04/)

- `sudo`: Super User Do
  - `sudo -i`: Switch to root user

- `apt-get`: Advanced Packaging Tool
  - `sudo apt-get update`: Update software list
  - `sudo apt-get upgrade`: Upgrade software
  - `sudo apt-get install <package>`: Install package
  - `sudo apt-get remove <package>`: Remove package
  - `sudo apt-get autoremove`: Remove unused packages
  - `sudo apt-get clean`: Clean cache
  - `sudo apt-get autoclean`: Clean cache and unused packages
  - `sudo apt-get purge <package>`: Remove package and configuration files

- `dpkg`: Debian Package
  - `sudo dpkg -i <package>`: Install package
  - `sudo dpkg -r <package>`: Remove package

- File manipulation:
  - `ls`: List files
    - `ls -a`: List all files including hidden files
    - `ls -l`: List files in detail
  
  - `cd`: Change directory
    - `cd ..`: Go to parent directory
    - `cd ~`: Go to home directory
    - `cd -`: Go to previous directory
  - `pwd`: Print working directory
  - `mkdir`: Make directory
    - Syntax: `mkdir dir1 dir2 dir3`
  - `touch`: Create file
    - Syntax: `touch file1 file2 file3`
  - `cp`: Copy file
    - `cp -r`: Copy directory
    - syntax: `cp <source> <destination>`
  - `mv`: Move file
    - `mv -r`: Move directory
    - `mv <old_name> <new_name>`: Rename file
    - `mv <file> <directory>`: Move file to directory
  - `rm`: Remove file
    - `rm -r`: Remove directory
    - `rm -f`: Force remove
    - `rm -rf`: Force remove directory

- `chmod`: Change mode
  - `chmod 777 <file>`: Give all permissions to file

- `&&`: Logical AND
  - `command1 && command2`: Execute command2 if command1 is successful

- `-y`: Yes
  - `sudo apt-get install -y <package>`: Install package without asking

- `|`: Pipe
  - `command1 | command2`: Pipe the output of command1 to command2

- `grep`: Global Regular Expression Print
  - `ls -l | grep <pattern>`: Search for pattern in output of ls -l


Strongly recommend the **Missing Semester** course from MIT. It is a very good course for learning Linux and shell [Link](https://missing.csail.mit.edu/). My notes for this course can be found [here](https://github.com/leishi23/Missing_Learning_Notes/tree/main). 
