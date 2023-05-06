---
layout: post
title:  "Encapsulation"
date:   2023-05-06
desc: "Encapsulation in C++"
keywords: "C"
categories: [C++]
tags: [C++]
icon: icon-html
---

# Encapsulation in C++

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z?p=99)

> **_Keypoints:_**
- Significance of encapsulation
- Examples
- Access specifier
- Difference between struct and class


## Significance of encapsulation
- Encapsulation is the process of combining data and functions into a single unit called class.
- Encapsulation is used to hide the internal representation, or state, of an object from the outside.

## Examples
- Example 1: Circle
```cpp
#include <iostream>
using namespace std;

const double PI = 3.14;
class Circle {
    public: // access specifier, public means anyone can access
        double radius;  // data member
        double getPerimeter(){  // member function
            return PI * radius * 2;
        }
};

int main() {

    Circle c1;  // create an object of class Circle, c1 is object name
    c1.radius = 10;
    cout << c1.getPerimeter() << endl;  // 62.8  
    return 0;
}
```

- Example 2: Student
```cpp
#include <iostream>
using namespace std;
#include <string>

class student {
    public:
        string name;
        int number;
        void show() {
            cout << "Name is: " << name << " StuNumber is: " << number << endl;
        }
        void setName(string set_name) {
            name = set_name;
        }
        void setNumber(int set_number) {
            number = set_number;
        }
};

int main() {
    int studentNum;
    cout << "Please enter the number of students: " << endl;
    cin >> studentNum;

    for (int i = 0; i < studentNum; i++) {
        student s;
        cout << "Please enter the name of student " << i + 1 << ": " << endl;
        cin >> name;
        s.setName(name);
        cout << "Please enter the stu_number of student " << i + 1 << ": " << endl;
        cin >> number;
        s.setNumber(number);
        s.show();
    }

    system("pause");
    return 0;
}
```

## Access specifier
- Public: anyone can access
- Private: only member functions can access
- Protected: only member functions and derived classes can access

```cpp
# include <iostream>
using namespace std;
# include <string>

class Person{
    public:
        string name;
    private:
        int code;
    protected:
        string lover;
    
    public:
        void func() {
            name = "Tom";
            code = 123; // accessible inside class
            lover = "Jerry"; // accessible inside class
        }
};

int main(){

    Person p1;
    p1.name = "Tommy"; // public can be accessed
    p1.code = 123; // Error! private cannot be accessed outside of class
    p1.lover = "Jerry"; // Error! protected cannot be accessed outside of class

    system("pause");
    return 0;
}
```

## Difference between struct and class
- The only difference is that struct members are public by default, while class members are private by default.

```cpp
# include <iostream>
using namespace std; 
# include <string>

class C1{
    int m_A;  // default private
};

struct C2{
    int m_A;  // default public
};

int main(){

    C1 c1;
    c1.m_A = 100; // Error! private cannot be accessed outside of class

    C2 c2;
    c2.m_A = 100; // public can be accessed

    return 0;
}
```