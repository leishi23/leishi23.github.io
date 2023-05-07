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
- Examples of encapsulation
- Access specifier
- Difference between struct and class
- Set member function
- Examples of set member function
- How to split class into .h and .cpp files

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

## Set member function
- Set member function is used to set private data members
  - To control the read and write permissions of private data members
  - To control the range of private data members

    ```cpp
    # include <iostream>
    using namespace std;
    # include <string>

    class Person{
        public:

            void setName(string name){
                m_Name = name;
            }
            string getName(){
                return m_Name;
            }

            void setAge(int age){
                if (age < 0 || age > 150){
                    cout << "Invalid age!" << endl;
                    m_Age = 0;
                    return;
                }
                m_Age = age;
            }
            int getAge(){
                return m_Age;
            }

            void setLover(string lover){
                m_Lover = lover;
            }

        private:
            // readable and writable 
            string m_Name;
            // readable and writable
            int m_Age;
            // only writable
            string m_Lover;
    };

    int main(){

        Person p1;
        p1.setName("Tom"); 
        p1.setAge(180); // error! invalid age 
        p1.setLover("Jerry"); 

        cout << "Name: " << p1.getName() << endl;
        cout << "Age: " << p1.getAge() << endl;
        cout << "Lover: " << p1.getLover() << endl; // error! cannot get lover

        return 0;
    }
    ```

## Examples of set member function
- Example: Cube
    ```cpp
    # include <iostream>
    using namespace std;
    # include <string>

    class Cube{
        public:
            // set length
            void setL(int l){
                m_L = l;
            }
            // get length
            int getL(){
                return m_L;
            }

            // set width
            void setW(int w){
                m_W = w;
            }
            // get width
            int getW(){
                return m_W;
            }

            // set height
            void setH(int h){
                m_H = h;
            }
            // get height
            int getH(){
                return m_H;
            }

            // get surface area
            int calculateS(){
                return 2 * m_L * m_W + 2 * m_W * m_H + 2 * m_H * m_L;
            }

            // get volume
            int calculateV(){
                return m_L * m_W * m_H;
            }

            // judge whether equal
            bool isSameByClass(Cube &c){
                if (m_L == c.getL() && m_W == c.getW() && m_H == c.getH()){
                    return true;
                }
                return false;
            }

        private:
            int m_L;
            int m_W;
            int m_H;
    };

    bool isSame(Cube &c1, Cube &c2){
        if (c1.getL() == c2.getL() && c1.getW() == c2.getW() && c1.getH() == c2.getH()){
            return true;
        }
        return false;
    }

    int main(){

        Cube c1;
        c1.setL(10);
        c1.setW(10);
        c1.setH(10);

        cout << "Surface area: " << c1.calculateS() << endl;
        cout << "Volume: " << c1.calculateV() << endl;

        Cube c2;
        c2.setL(10);
        c2.setW(10);
        c2.setH(10);

        bool ret1 = c1.isSameByClass(c2);
        bool ret2 = isSame(c1, c2);

        return 0;
    }
    ```

    - Example: Point and circle 
    ```cpp
    # include <iostream>
    using namespace std;
    # include <string>

    class Point{
        public:
            void setX(int x){
                m_X = x;
            }
            int getX(){
                return m_X;
            }

            void setY(int y){
                m_Y = y;
            }
            int getY(){
                return m_Y;
            }

        private:
            int m_X;
            int m_Y;
    };

    class Circle{
        public:
            void setR(int r){
                m_R = r;
            }
            int getR(){
                return m_R;
            }

            void setCenter(Point center){
                m_Center = center;
            }
            Point getCenter(){
                return m_Center;
            }

        private:
            int m_R;
            Point m_Center;
    };

    int main(){

        Point p1;
        p1.setX(10);
        p1.setY(10);

        Point p2;
        p2.setX(9);
        p2.setY(9);

        Circle c1;
        c1.setR(10);
        c1.setCenter(p2);

        distance = sqrt(pow(c1.getCenter().getX() - p1.getX(), 2) + pow(c1.getCenter().getY() - p1.getY(), 2));

        if distance == c1.getR(){
            cout << "Point is on the circle" << endl;
        }
        elif distance > c1.getR(){
            cout << "Point is outside of the circle" << endl;
        }
        else{
            cout << "Point is inside of the circle" << endl;
        }
    }
    ```

## How to split class into .h and .cpp files
    ```cpp
    // Point.h
    # program once // prevent multiple inclusions
    # include <iostream>
    using namespace std;

    class Point{
        public:
            void setX(int x);
            int getX();

            void setY(int y);
            int getY();

        private:
            int m_X;
            int m_Y;
    };

    // Point.cpp
    # include "Point.h" // include the header file
    void Point::setX(int x){    // Point:: is the scope resolution operator
        m_X = x;
    }
    int Point::getX(){
        return m_X;
    }

    void Point::setY(int y){
        m_Y = y;
    }
    int Point::getY(){
        return m_Y;
    }
    ```