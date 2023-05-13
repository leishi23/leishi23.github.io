---
layout: post
title:  "Win10/Ubuntu 18双系统安装/卸载"
date:   2023-05-13
desc: "Win10/Ubuntu 18双系统安装/卸载"
keywords: "Linux"
categories: [Linux]
tags: [Linux]
icon: icon-html
---

# Win10/Ubuntu 18双系统安装/卸载
## _Linux_DebugNotes_01_

[![image](https://fossbytes.com/wp-content/uploads/2020/02/Ubuntu-18.04.4-release.jpg)](https://releases.ubuntu.com/18.04/)

## Keypoints 

- Install Ubuntu 18.04.4 LTS on Windows 10
- Uninstall Ubuntu 18.04.4 LTS on Windows 10


## Install Ubuntu 18.04.4 LTS on Windows 10

- Download Ubuntu 18.04.4 LTS iso file from [Ubuntu](https://releases.ubuntu.com/18.04/)
  - Select desktop image
  
- 烧录iso文件到U盘
  - Download [Rufus](https://rufus.ie/zh/) to burn iso file to USB flash drive
  - When burning:
    - Select USB flash drive
    - Select iso file
    - Other settings remain default
    - Click start
  
- 磁盘管理
  - Win10下按下`Win+X`，选择`磁盘管理`
  - 选择一个磁盘分区，右键选择`压缩卷`
  - 设置分区大小，建议大于20G
  - 点击`压缩`
  - 此时会出现一个未分配的磁盘空间
  
- Restart computer
  - Press `F12` to enter BIOS when computer is starting, select `USB flash drive` to enter Ubuntu installation interface
  - or press `shift` when clicking `restart` in Windows 10. Select `USB flash drive` to enter Ubuntu installation interface

- Installation
  - Select 'Language' and click 'Install Ubuntu'
  - Select 'Keyboard layout' and click 'Continue'
  - Select 'Normal installation' and click 'Continue'
  - Select `Something else` and click `Continue`
  - Select the unallocated disk free space and click `+`
    - Set `/` as mount point
  - Select `Install Now`
  - Select `Continue` to write the changes to disk
  - Select your location and click `Continue`
  - Enter your name, computer's name, username and password


## Uninstall Ubuntu 18.04.4 LTS on Windows 10
- Open `Disk Management` by `win + X` in Windows 10
- Select the partition of Ubuntu and right click to delete it
  - Never delete the partition of Windows 10 or **EFI** System Partition
- `win + R` to open `cmd` to run the following commands
  ```cmd
    diskpart
    list disk // list all disks
    select disk 0 // select the disk where Windows 10 is installed
    list partition // list all partitions
    select partition 1 // select the partition of Windows EFI 
    assign letter=Z // assign a letter to the partition
    exit
    ```
    Then you can see the partition of Windows EFI in `This PC` as `Z:`
- To open `Z:`:
  - 以管理员身份运行记事本
  - 左上角`文件`->`打开`->`计算机`->`Z:`->`EFI`删除`ubuntu`文件夹
  - `remove letter=Z` in `cmd` to remove the letter of Windows EFI partition