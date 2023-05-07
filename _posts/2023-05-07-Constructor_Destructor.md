---
layout: post
title:  "Constructor and Destructor"
date:   2023-05-06
desc: "Constructor and Destructor in C++"
keywords: "C"
categories: [C++]
tags: [C++]
icon: icon-html
---

# Constructor and Destructor in C++

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z?p=106)

> **_Keypoints:_**
- Initialization and cleanup of objects
- Constructor and destructor

## Initialization and cleanup of objects
- Initialization: set the initial value of an object
- Cleanup: release the resources of an object

## Constructor and destructor
- Constructor is a special member function that is executed when an object of that class is created. Destructor is a special member function that is executed when an object of that class is destroyed.
  
- If we do not specify a constructor, the compiler generates a default constructor. But a void one. Same for destructor.
- Syntax:
  - Constructor
    - No return type
    - Same function name as class name
    - Can be overloaded
    - Automatically called when an object is created
    ```cpp
    class_name::class_name() {
        // body of constructor
    }
    ```

  - Destructor
    - No return type
    - Same function name as class name, but with a tilde (~) symbol
    - Cannot be overloaded
    - Automatically called when an object is destroyed
    ```cpp
    class_name::~class_name() {
        // body of destructor
    }
    ```

- Example:
    ```cpp
    #include <iostream>
    using namespace std;

    class Person{
        public:
            Person() {
                cout << "Constructor called" << endl;
            }
            ~Person() {
                cout << "Destructor called" << endl;
            }
    };

    int main() {
        Person p1;  // Constructor called automatically when p1 is created
        return 0;   // p1 is a local variable inside main function, so it is destroyed when main function ends
    }  // Destructor called automatically when p1 is destroyed/ main function ends
    ```