---
layout: post
title:  "Project_InfoSystem"
date:   2023-04-09
desc: "Black horse online course"
keywords: "C++"
categories: [C++]
tags: [C++]
icon: icon-html
---

# Project_InfoSystem

[![image](https://www.freeiconspng.com/thumbs/c-logo-icon/c--logo-icon-0.png)](https://www.bilibili.com/video/BV1et411b73Z?p=73&spm_id_from=pageDriver&vd_source=d8d0bffc8e5266c19ad61d5b6c71609e)

```cpp
# include <iostream>
using namespace std;
# include <string>
# define MAX 1000

void ShowMenu(){
    cout << "1. Add contacts" << endl;
    cout << "2. Show contacts" << endl;
    cout << "3. Delete contacts" << endl;
    cout << "4. Find contacts" << endl;
    cout << "5. Change contacts" << endl;
    cout << "6. Empty contacts" << endl;
    cout << "0. Exis info-system" << endl;
}

void addPerson(AddressBook *abs){
    // to test if abs's size equals to MAX
    if (abs->m_Size == MAX){
        cout << "AddressBook is full!" << endl;
    }
    else{
        string name;
        cout << "Please add a name:" << endl;
        cin >> name;
        abs->personArray[abs->m_Size].m_Name = name;
        
        int sex;
        while (true){
            cout << "Please enter the sex:" << endl;
            cout << "1 -- Male:" << endl;
            cout << "2 -- Female:" << endl;
            cin >> sex;
            if (sex == 1 || sex == 2){
                abs->personArray[abs->m_Size].m_Sex = sex;
                break;      // break when sex input is right 
            }
            cout << "Sex input is wrong, please try again" << endl;
        }

        int age;
        cout << "Please enter the age:" << endl;
        cin >> age;
        abs->personArray[abs->m_Size].m_Age = age;
        
        string phone;
        cout << "Please enter the phone number:" << endl;
        cin >> phone;
        abs->personArray[abs->m_Size].m_Phone = phone;
        
        string addr;
        cout << "Please enter the address:" << endl;
        cin >> addr;
        abs->personArray[abs->m_Size].m_Addr = addr;
        
        abs->m_Size++;
        cout << "Added successfully" << endl;
        
        system("pause");
        system("cls");      // clear screen
    }
}

void showContact(AddressBook *abs){
    if (abs->m_Size == 0){
        cout << "Address book is empty" << endl;
    }
    else{
        for (int i=0; i<abs->m_Size; i++){
            cout << "Name: " << abs->personArr[i].m_Name << "\t";   
            //abs is pointer while abs->personArr[i] is not, so ->/.
            cout << "Sex: " << (abs->personArr[i].m_Sex == 1 ? "Male":"Female") << "\t";
            cout << "Age: " << abs->personArr[i].m_Age << "\t";
            cout << "Phone: " << abs->personArr[i].m_Phone << "\t";
            cout << "Address: " << abs->personArr[i].m_Addr << endl;
        }
    }
    system("pause");
    system("cls");
}

int verifyContact(AddressBook *abs, string name){
    for (int i=0; i<abs->m_Size; i++){
        if (abs->personArr[i].m_Name == name){
            return i;
        }
    }
    return -1;
}

void deletePerson(AddressBook *abs){
    cout << "Please enter the person you want to delete" << endl;
    string name;
    cin >> name;
    int ret = verifyContact(abs, name);
    if (ret != -1){
        for (int i=0; i<abs->m_Size; i++){
            abs->personArr[i] = abs->personArr[i+1]
        }
        abs->m_Size--;
        cout << "Found and deleted" << endl;
    }
    else {
        cout << "This person not exist" << endl;
    }
}

void findPerson(AddressBook *abs){
    cout << "Please enter the person you want to find" << endl;
    string name;
    cin >> name;
    int ret = verifyContact(abs, name);
    if (ret != -1){
        cout << "Name: " << abs->personArr[ret].m_Name << "\t";
        cout << "Sex: " << abs->personArr[ret].m_Sex << "\t";
        cout << "Age: " << abs->personArr[ret].m_Age << "\t";
        cout << "Phone: " << abs->personArr[ret].m_Phone << "\t";
        cout << "Address: " << abs->personArr[ret].m_Addr << endl;
    }
    else{
        cout << "Nobody found" << endl;
    }
    system("pause");
    system("cls");
}

void modifyPerson(Addressbooks * abs){
    cout << " Enter the person you want to modify" <<endl;
    string name;
    cin >> name;
    int ret = isExist(abs, name);
    
    if (ret != -1){
        string name;
        cout << "Enter name: " << endl;
        cin >> name;
        abs->personArray[ret].m_Name = name;
        
        cout << "Enter sex:" << endl;
        cout <<"1 ---- Male" << endl;
        cout <<"2 ---- Female" << endl;
        int sex = 0;
        cin >> sex;
        while (true){
            if (sex==1 || sex==2){
                abs->personArray[ret].m_Sex = sex;
                break;
            }
            cout << "Sex entered is wrong, please try again" << endl;
        }
        
        cout << "Enter age:" << endl;
        int age = 0;
        cin >> age;
        abs->personArray[ret].m_Age = age;
        
        cout << "Enter contact phone:" << endl;
        string phone;
        cin >> phone;
        abs->personArray[ret].m_Phone = phone;
        
        cout << "Enter home address: " << endl;
        string address;
        cin >> address;
        abs->personArray[ret].m_Addr = address;
        
        cout << "Modified successfully" << endl;
    }
    else{
        cout << "No one found" << endl;
    }
    system("pause");
    system("cls");
}

void cleanPerson(Addressbooks * abs){
    abs->m_Size = 0; // Set current contact number 0
    cout << "Addressbook is already emptied." << endl;
    system("pause");
    system("cls");
}

struct Person{
    string m_Name;
    int m_Sex;  // 1 male; 2 female
    int m_Age;
    string m_Phone;
    string m_Addr;
};

struct AddressBook{
    struct Person personArr[MAX];
    int m_Size;
};

int main(){
    
    AddressBook abs;
    abs.m_Size = 0;

    int select = 0;
    cin >> select;
    
    while (true){
        ShowMenu();
        
        switch (select){
            case 1: // Add contacts
                addPerson(&abs);        // address pass to modify value
                break;
            case 2: // Show contacts
                showContact(&abs);
                break;
            case 3: // Delete contacts
                deletePerson(&abs);
                break;
            case 4: // Find contacts
                findPerson(&abs);
                break;
            case 5: // Change contacts
                break;
            case 6: // Empty contacts
                break;
            case 0: // Exis info-system
                cout << "See you!" << endl;
                system('pause');    // Pause the system, and print “Press any key to continue...”
                return 0;   // exit main function
                break;
            default:
                break;
        }
    }
    
    system('pause');
    return 0;   // exit main function
}

```
