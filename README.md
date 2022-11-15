# Curly-Hair-simulation
This project about https://graphics.pixar.com/library/CurlyHairA/paper.pdf use C++ and OpenGL <br>
if you want a test this project you should follow https://pig-tag.tistory.com/6 

# Build sequence

-> unzip HairSimulation(GPU)/nanogui.zip
-> build cmake vs2017, 64x same directory
-> visual studio -> project setting -> projects "nanogui, nanogui-obj, glfw_objects" output directory change to inheritance parent


# Hair model

## Stretch spring

## Bending spring
|<img src="https://user-images.githubusercontent.com/82528291/163705545-6ef91bbc-d293-44de-b119-794f224eafa2.gif"/>|![bending frame](https://user-images.githubusercontent.com/82528291/165365474-5e22d450-238e-42af-916c-a39515be4c2c.gif)|
|:--:|:--:|
|Bending spring|Smoothing function with Parallel transport frame|

## Core spring
|![no_core_normal_bending](https://user-images.githubusercontent.com/82528291/165364832-7883cecf-c7fd-4a0f-8217-320cef452e71.gif)|![no_core_high_bending](https://user-images.githubusercontent.com/82528291/165364857-0b12bf77-8702-43a2-a988-1dbff3097d58.gif)|![core_spring_normal_bendinggif](https://user-images.githubusercontent.com/82528291/165364884-17c5e230-d6c2-4647-a9d9-d7d7f0f3bfe3.gif)|
|:--:|:--:|:--:|
|No core springs, normal bending springs|No core springs, high bending springs|Core springs with, normal bending springs|
