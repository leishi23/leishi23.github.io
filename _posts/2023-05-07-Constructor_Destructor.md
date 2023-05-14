---
layout: post
title:  "Constructor and Destructor"
date:   2023-05-06
desc: "Constructor and Destructor in C++"
keywords: "C++"
categories: [C++]
tags: [C++]
icon: icon-html
---

# Constructor and Destructor in C++

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z?p=106)

> **_Keypoints:_**
- Initialization and cleanup of objects
- Constructor and destructor
- Constructor classification and calling 
- When to use copy constructor
- Constructor call rules
- Deep copy and shallow copy

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

## Constructor classification and calling
- By parameter
  - Default constructor: no parameter
  - Parameterized constructor: with parameters
- By type
  - Copy constructor: copy an object to another
  - Normal constructor: no copy

  ```cpp
  #include <iostream>
  using namespace std;

  class Person{
      public:

          int age;
          Person() {
              cout << "Default constructor called" << endl;
          }

          Person(int a) {
              age = a;
              cout << "Parameterized constructor called" << endl;
          }

          Person(const Person &p) { // const is used to avoid changing the value of p, & is used to avoid copying the whole object
              age = p.age;  // copy age from p
              cout << "Copy constructor called" << endl;
          }

          ~Person() {
              cout << "Destructor called" << endl;
          }
  }

  void test01(){
      // 1. bracket calling
      Person p1;  // Default constructor called. 
                  // Don't add () after class name here or it will be treated as a function declaration
      Person p2(10);  // Parameterized constructor called
      Person p3(p2);  // Copy constructor called
      cout << "p2.age = " << p2.age << endl;  // p2.age = 10
      cout << "p3.age = " << p3.age << endl;  // p3.age = 10

      // 2. explicit calling
      Person p4;  // Default constructor called
      Person p5 = Person(10);  // Parameterized constructor called
      Person p6 = Person(p5);  // Copy constructor called

      Person(10);  // Annymous object, will be destroyed immediately after this line. Add tje `Person p5 = ` before it to keep it explicitly

      // Don't use copy constructor to initialize an object
      Person (p5);  // Wrong, will be treated as a `Person p5;` 

      // 3. implicit calling
      Person p7 = 10;  // Parameterized constructor called. Equivalent to `Person p7(10);`
      Person p8 = p7;  // Copy constructor called. Equivalent to `Person p8(p7);`
  }

  int main(){
      test01();
      return 0;
  }

## When to use copy constructor
- When an object is initialized by another object of the same class
- When an object is passed by value as a parameter to a function
- When an object is returned from a function by value

  ```cpp
  #include <iostream>
  using namespace std;

  class Person{
      public:
          Person() {
              cout << "Default constructor called" << endl;
          }

          Person(int a) {
              age = a;
              cout << "Parameterized constructor called" << endl;
          }

          Person(const Person &p) { 
              age = p.age; 
              cout << "Copy constructor called" << endl;
          }

          ~Person() {
              cout << "Destructor called" << endl;
          }

          int age;
  }

  // 1. When an object is initialized by another object of the same class
  void test01(){
      Person p1(10);  // Parameterized constructor called
      Person p2(p1);  // Copy constructor called
      cout << "p2.age = " << p2.age << endl;  // p2.age = 10
  }

  // 2. When an object is passed by value as a parameter to a function
  void doWork(Person p) {  // p is a local variable inside this function, it copies the value of the object passed in, equivalent to `Person p = p1;`
      cout << "p.age = " << p.age << endl;  // p.age = 10
  }

  void test02(){
      Person p1(10);  // Parameterized constructor called
      doWork(p1);  // Copy constructor called
  }

  // 3. When an object is returned from a function by value
  Person doWork2() {
      Person p1(10);
      cout << "p1.age = " << p1.age << endl;  // p1.age = 10
      return p1;  // Copy constructor called
  }

  void test03(){
      Person p2 = doWork2();  // Copy constructor called, equivalent to `Person p2 = p1;` Two copy constructors are called in total
      cout << "p2.age = " << p2.age << endl;  // p2.age = 10
  }

  int main(){
      test01(); 
      test02();
      test03();
      return 0;
  }
  ```


## Constructor call rules
- If we do not specify a constructor, the compiler will generate at least three constructors: default constructor, parameterized constructor and copy constructor.
- If we specify a parameterized constructor, the compiler will not generate a default constructor but still generate a copy constructor.
- If we specify a copy constructor, the compiler will not generate a default constructor or a parameterized constructor.

  ```cpp
  #include <iostream>
  using namespace std;

  class Person{
      public:
          Person() {
              cout << "Default constructor called" << endl;
          }
          // if we specify a parameterized constructor, the compiler will not generate a default constructor. Then `Person p1;` will be wrong

          Person(int a) {
              age = a;
              cout << "Parameterized constructor called" << endl;
          }

          ~Person() {
              cout << "Destructor called" << endl;
          }

          int age;
  }

  void test01(){
      Person p1;  // Default constructor called
      p1.age = 10;
      Person p2(p1);  // No output, because the copy constructor is generated by the compiler automatically
      cout << "p2.age = " << p2.age << endl;  // p2.age = 10
  }

  ```

## Deep copy and shallow copy
- Shallow copy: copy the value of the pointer, not the value of the pointed object with the copy constructor generated by the compiler defaultly
- Deep copy: copy the value of the pointer and the value of the pointed object with the copy constructor defined by ourselves
- If there is a heap memory in constructor, customized copy constructor needed and deep copy are necessary for avoiding releasing heap memory repeatedly.

  ```cpp
  #include <iostream>
  using namespace std;

  class Person{
      public:
          Person() {
              cout << "Default constructor called" << endl;
          }

          Person(int a, int h) {
              age = a;
              height = new int(h);  // allocate heap memory for height
              cout << "Parameterized constructor called" << endl;
          }

          Person(const Person &p){
            cout << "Copy constructor called" << endl;
            age = p.age;
            height = new int(*p.height) // a different pointer but refers to the same address
          } // Deep copy will allocate a new pointer referring to the same address, so the repeated release is solved.

          ~Person() {
              // release the heap memory
              if (height != NULL) {
                  delete height;
                  height = NULL;  // avoid dangling pointer
              }
              cout << "Destructor called" << endl;
          }

          int age;
          int *height;
  }

  void test01(){
      Person p1(18, 160);
      cout << "p1.age = " << p1.age << endl;  // p1.age = 18
      cout << "p1.height = " << *p1.height << endl;  // p1.height = 160

      Person p2(p1);  // Shallow copy, only copy the value of the pointer
      cout << "p2.age = " << p2.age << endl;  // p2.age = 18
      cout << "p2.height = " << *p2.height << endl;  // p2.height = 160
  } 
  // wrong, the heap memory will be released repeatedly in deconstructor

  int main(){
      test01();
      return 0;
  }
  ```

