---
layout: post
title:  "Reference in C++"
date:   2023-05-03
desc: "Reference in C++"
keywords: "C++"
categories: [C++]
tags: [C++]
icon: icon-html
---

# Reference in C++

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z?p=89)

## Keypoints 

- Basic syntax
- Attention
- Reference as function parameter
- Reference as function return value
- Essence of reference
- Constant reference

## Basic syntax
- Function: to call other name for a variable
- Syntax: `type &new_name = old_name;`
  - Example: 
    ```cpp
    int x = 5;  // initialize x
    int &y = x;
    y = 10;
    cout << x << endl; // 10
    ```

## Attention
- Reference is not a new variable, it is just another name for the same variable.
- Reference must be initialized when it is created and cannot be changed after initialization.
    ```cpp
    int x = 5;
    int &y; // error: y must be initialized
    int &y = x;
    int z = 10;
    int &y = 10; // error: y cannot be initialized by constant, has to be an address
    int &y = z; // error: y cannot be changed after initialization
    y = z; // y is still x, but with value of z
    cout << x << endl; // 10
    cout << y << endl; // 10
    cout << z << endl; // 10
    ```

## Reference as function parameter
- Pass by reference (reference as function parameter). Similar to pass by pointer and also be able to change the value of the original variable.
  - Syntax: `void func(type &var);`
  - Example:
    ```cpp
    void swap(int &x, int &y) {
        int temp = x;
        x = y;
        y = temp;
    }
    int main() {
        int a = 5, b = 10;
        swap(a, b);
        cout << a << endl; // 10
        cout << b << endl; // 5
        return 0;
    }
    ```

## Reference as function return value
- Return by reference (reference as function return value). Similar to return by pointer and also be able to change the value of the original variable.
  - Syntax: `type &func();`
  - Example:
    ```cpp
    int &func(int &x) {
        x++;
        return x;
    }
    int main() {
        int a = 5;
        func(a) = 10;
        cout << a << endl; // 10
        return 0;
    }
    ```

  - Don't return reference of local variable.
    ```cpp
    int &func() {
        int x = 5;
        return x; // error: x is local variable, it's in stack and will be destroyed after function call
    }
    int main() {
        int &y = func();
        cout << y << endl; // right, but y will be destroyed after once call
        cout << y << endl; // error: y is reference of local variable
        return 0;
    }
    ```

  - Fcuntion calling itself.
    ```cpp
    int &func() {
        static int x = 5;   // static variable isn't in stack, it's in global area. So it won't be destroyed after function call by compiler.
        return x;
    }
    int main() {
        int &y = func();
        cout << y << endl; // 5
        cout << y << endl; // 5

        func() = 10;    // function() return a reference of x
        cout << y << endl; // 10
        cout << y << endl; // 10

        return 0;
    }
    ```

## Essence of reference
- Reference is just a pointer in essence.
  - Example:
    ```cpp
    int x = 5;
    int &y = x; // int *const y = &x;
    int *z = &x;
    y = 10; // *y = 10;
    cout << &x << endl; // 0x7ffeeb0b0a3c
    cout << &y << endl; // 0x7ffeeb0b0a3c
    cout << z << endl;  // 0x7ffeeb0b0a3c
    ```

## Constant reference
- Constant reference: reference of constant variable to avoid changing that value.
  - Syntax: `const type &new_name = old_name;`
  - Example:
    ```cpp
    void showValue(const int &x) {
        x++;    // error: x is constant
        cout << x << endl;
    }
    int main() {
        int a = 5;
        showValue(a); // 5
        return 0;
    }
    ```
