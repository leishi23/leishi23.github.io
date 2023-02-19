---
layout: post
title:  "Hello World Program in C (Day 02_1)"
date:   2023-02-19
desc: "Intermediate Programming Course Day 02_1"
keywords: "C++"
categories: [Html]
tags: [C++]
icon: icon-html
---

# Hello World Program in C
## _Intermediate Programming Course Day 02_1_

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://jhu-ip.github.io/cs220-sp23/material.html)

## Keypoints 

- Compiling and running a simple C program 


## Hello world 

```c
// hello_world.c
#include <stdio.h>

// Print "Hello, world!" followed by newline and exit
int main(void){
    printf("Helllo, world\n");
    return 0;
}

// Compilation command line 
$ gcc hello_world.c -std=c99 -pedantic -Wall -Wextra
$ ./a.out
// Output 
Hello, world!
```

- ==#include== is a preprocessor directive, similar to import in Python 
- ==<stdio.h>== means it's defined in stdio.h(standard input output package); ==.h== is "header file"
- ==main== is a function, every program has exactly one main 
- ==int== is the return type 
- ==(void)== says the main function takes no inputs/parameters 
- ==printf== prints a string to "standard out" (terminal)
- ==\n== denotes the newline character 
- ==return 0== means "program completed without error"
- ==gcc== is the compiler; ==hello_world.c== is the cpp file name; 
- Switches in compilation command: ==-std=c99== is to use the c99 standard from 1999 year; ==-pedantic -Wall -Wextra== is to get full feedback, i.e. error and warning (potential error)
- ==./== is the execution sign; ==.== is current directory 
- ==a.out== is the compilation default file name 

## Basic C/C++ programming workflow 

- Edit file (using EMACS or VIM)
- Compile using GNUplot C compiler (gcc) to compile, link, and create excut
    - If compile-time errors reported, edit .c file and re-compile 
- Run the excutable file 
    - If run-time errors reported/detected, go back to edit step  


### Inside the compiler 

- Step 1：preprocessor 
    - Bring together all the code that belongs together 
    - Process the directives that start with #, such as #include 
        - also #define (to ensure library exists)
- Step 2：compiler 
    - Turn human-readable source code into [object code](https://en.wikipedia.org/wiki/Object_code) 
    - Might yield warnings and errors if mistakes are "visible" to compiler 
- Step 3：linker 
    - Bring together all the relevant object code into a single excutable file 
    - Might yield warnings and errors if relevant code is missing, naming conflicts, etc
```
many .c files --compiler--> many .o files --linker--> one excutable file 
```
