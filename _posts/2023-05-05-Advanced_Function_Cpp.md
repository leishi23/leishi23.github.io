---
layout: post
title:  "Adcanced Function"
date:   2023-05-05
desc: "Cpp Learning Notes 10"
keywords: "C++"
categories: [C++]
tags: [C++]
icon: icon-html
---

# Adcanced Function
## _Cpp Learning Notes 10_

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z?p=95&vd_source=d8d0bffc8e5266c19ad61d5b6c71609e)

## Keypoints 

- Default arguments
- Placeholder parameters
- Function overloading

## Default arguments
- Syntax: `return type function_name(type1 arg1, type2 arg2 = default_value);`
  - The default value would be overwritten if the argument is passed in the function call
  - The arguments with default values must be at the end of the argument list
    - `int func(int a, int b = 10, int c);` is not allowed
  - If the declaration and definition are separated, the default value must be specified in the declaration, cannot be in the definition
    - `int func(int a, int b = 10, int c);` in the declaration
    - `int func(int a, int b, int c){...}` in the definition
- Example: 
    ```cpp
    #include <iostream>
    using namespace std;
    int func(int a, int b = 10, int c = 20){
        return a + b + c;
    }

    int main(){
        cout << func(10) << endl; // 40
        cout << func(10, 30) << endl; // 60
        cout << func(10, 30, 40) << endl; // 80
        return 0;
    }
    ```

## Placeholder parameters
- Syntax: `return type function_name(type1 arg1, type2)`
- Example: 
    ```cpp
    #include <iostream>
    using namespace std;
    int func(int a, int){
        return a;
    }

    int main(){
        cout << func(10, 20) << endl; // 10
        return 0;
    }
    ```

## Function overloading
- Function name can be reused if:
  - In the same scope
  - Function name is the same
  - Function **parameters** are different in number, type, or sequence. 
    - Return type is not considered
  - Example:
      ```cpp
      #include <iostream>
      using namespace std;

      void func(){
          cout << "func()" << endl;
      }

      void func(int a){
          cout << "func(int a)" << endl;
      }

      void func(double a){
          cout << "func(double a)" << endl;
      }

      void func(int a, double b){
          cout << "func(int a, double b)" << endl;
      }

      void func(double a, int b){
          cout << "func(double a, int b)" << endl;
      }

      int main(){
          func(); // func()
          func(10);   // func(int a)
          func(3.14); // func(double a)
          func(10, 3.14); // func(int a, double b)
          func(3.14, 10); // func(double a, int b)
          return 0;
      }
      ```

- Reference parameters can be overloaded with value parameters
  - The compiler will always choose the value parameter
  - The compiler will always choose the const reference parameter
  - Example:
  ```cpp
    #include <iostream>
    using namespace std;

    void func(int &a){
        cout << "func(int &a)" << endl;
    }

    void func(const int &a){
        cout << "func(const int &a)" << endl;
    }

    int main(){
        int a = 10;
        func(a); // func(int &a), essentially func(int &a = a)
        func(10); // func(const int &a), essentially func(const int &a = 10)
        return 0;
    }
    ```

- Function overloading and default arguments
  - Example
  ``` cpp
    #include <iostream>
    using namespace std;

    void func(int a, int b = 10){
        cout << "func(int a, int b = 10)" << endl;
    }

    void func(int a){
        cout << "func(int a)" << endl;
    }

    int main(){
        func(10); // Error: call of overloaded 'func(int)' is AMBUGUOUS!
        func(10, 20); // func(int a, int b = 10), no ambiguity here
        return 0;
    }
  ```
