---
layout: post
title:  "C ++ Program Memory Division"
date:   2023-04-30
desc: "C ++ Program Memory Division"
keywords: "C++"
categories: [C++]
tags: [C++]
icon: icon-html
---

# C++ Basics
## _C ++ program memory division_

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z/?p=84&spm_id_from=333.1007.top_right_bar_window_history.content.click)

## Keypoints 

- Memory division classfication
- 

## Memory division classfication
![image](https://th.bing.com/th/id/R.1a1dbbe8b343484a55f3ff386b9cb48a?rik=XWO3jGPhFp1E4A&pid=ImgRaw&r=0)
- Significance: to bring different life cycle for different division, for dexterous programming.
- **Text/Code** division: Store the binary code of the correspondence, managed by the operating system. Assigned before program execution.
    - Shared: for frequently run code, it's sufficient to store only once.
    - Only readable: not changable.
- **Global** division: Store global variables and static variables and constants. Assigned before program execution.
    - Global variable: out of main function.
    - Static variable: its address is near global ones.
        ```cpp
        static data_type variable_name = 10;
        ```
    - Constant: string variable; const global variable. (constant local variable isn't in Global division).
    
- **Stack** division: Automatically by compiler, to store function parameter and local variable.
    - Don't return address of local variable. Because stack data will be released after program execution. 
        ```cpp
        int * func(){
            int a = 10; 
            return &a;
        }
        int main(){
            int *p = func(); 
            cout << *p << endl; // return 10
            cout << *p << endl; // return mess
            cout << *p << endl; // return mess; because compiler store stack data only once, then it'll be released
        }
        ```

- **Heap** division: Assigned and released by programmer. When program is over, OS will release it.
    - Use **new** to start new heap area.
        ```cpp
        int * func(){
            int *p = new int(10); // pointer here is a local variable and in stack area; 
                                  //new int (10) is not a value but an address
            return p;  // the data this pointer refers to is in heap area
        }
        int main(){
            int *p = func(); // right, because 10 is in heap area
            cout << *p << endl;
            cout << *p << endl;
            cout << *p << endl; // because programmer decide the release or not of heap data. It'll not be released automatically by compiler.
        }
        ```
