---
layout: post
title:  "双系统开机引导修复（Grub界面）"
date:   2023-05-13
desc: "双系统开机引导修复（Grub）"
keywords: "Linux"
categories: [Linux]
tags: [Linux]
icon: icon-html
---

# 双系统开机引导修复（Grub）
## _Linux_DebugNotes_02_

[![image](https://fossbytes.com/wp-content/uploads/2020/02/Ubuntu-18.04.4-release.jpg)](https://releases.ubuntu.com/18.04/)

## Keypoints 

- Repair in Grub interface
- Repair in Windows 10

## Repair in Grub interface
- Repair grub
  ```bash
  grub> ls # you can see the partitions of your hard disk, such as (hd0) (hd0,msdos1) (hd0,msdos2) (hd0,msdos3) (hd0,msdos4)

  grub> ls (hd0,msdos1)/ # to find the partition where the grub files are located, such as (hd0,msdos1)/boot/grub. If you can't find it, try other partitions. If you can't find it, you can only repair it in Windows 10

  grub> set root=(hd0,msdos1) # set the partition where the grub files are located
  grub> set prefix=(hd0,msdos1)/boot/grub # set the path of grub files
  grub> insmod normal # load normal module
  grub> normal # enter normal mode
  ```

- Update grub
  ```bash
  sudo update-grub
  sudo grub-install /dev/sda
  ```

## Repair in Windows 10
- To enter Windows 10 when starting the computer, press `F12` to enter BIOS, select `Windows Boot Manager` to enter Windows 10
- Repair in Windows 10: Just delete the Ubuntu partition and then delete the Ubuntu folder in the EFI partition
- Other methods can be found on the Internet. I haven't tried it. I don't know if it works. 
