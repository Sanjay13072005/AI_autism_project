# 📹 YOLO Autism Project - Activity & Sleep Monitoring System

## What Is This Project? 🤔

This is a **smart video surveillance system** that watches a person and automatically detects:
- 🚶 **What they're doing** (walking, running, sitting, etc.)
- 😴 **If they're sleeping** (by watching their eyes)

Think of it like a helpful AI assistant that watches someone and tells you their current activity and sleep status in real-time!

---

## What Can It Do? ✨

| Feature | What It Does |
|---------|-------------|
| 🦴 **Pose Detection** | Detects all 17 body points (head, arms, legs, etc.) from video |
| 🏃 **Activity Recognition** | Recognizes if person is walking, running, or sitting |
| 😴 **Sleep Alert** | Detects when eyes are closed for too long |
| 📱 **Phone Camera Support** | Works with your Android phone as a camera |
| 💻 **Webcam Support** | Also works with your laptop's built-in camera |
| 👁️ **Face Detection** | Detects face and eyes to check if person is sleeping |  

---

## Project Folders Explained 📁

```
yolo_autism_project/                    ← Main folder
│
├── main.py                             ← ⭐ Main program (run this!)
├── main_sleep_only.py                  ← Simpler version (sleep only)
├── extract_pose_from_dataset.py        ← Helps train the system
│
├── activity_agent/                     ← Activity detection code
│   ├── activity_Agent.py               ← Brain that detects activities
│   └── memory.py                       ← Remember previous activities
│
├── camera/                             ← Camera connection
│   └── ip_camera.py                    ← Connect to phone camera
│
├── pose/                               ← Body detection
│   └── pose_detector.py                ← Finds body joints in video
│
├── models/                             ← AI Models (smart algorithms)
│   ├── activity_lstm.pt                ← Recognizes activities
│   └── activity_model.pt               ← Backup activity model
│
└── pose_features/                      ← Training data
    ├── X.pt                            ← Sample movements
    └── y.pt                            ← Labels for movements
```

**In Simple Terms:**
- 📁 `activity_agent/` = Brain that decides what activity the person is doing
- 📁 `camera/` = Handles video from phone or laptop camera
- 📁 `pose/` = Finds the person's body parts in the video
- 📁 `models/` = Smart AI models that make predictions

---

## How Does It Work? 🧠

### Step 1️⃣: Get Video from Camera
- System connects to a camera (phone or laptop)
- Gets live video stream

### Step 2️⃣: Find Body Parts (Pose Detection)
```
🎬 Video Frame → 🦴 Find Body Joints (head, arms, legs, etc.)
```
Uses AI to find 17 body points on the person in the video.

### Step 3️⃣: Check If Person Is Sleeping
```
👀 Look at Eyes → 😴 Are Eyes Closed?
If YES for 10 seconds → PERSON IS SLEEPING ⚠️
```
Checks the distance between eyelids. If closed too long = sleeping alert.

### Step 4️⃣: Detect Activity (What They're Doing)
```
🦴 Body Movement → 🤖 AI Brain → 🏃 WALKING / 🪑 SITTING / 🏃‍♂️ RUNNING
```
Looks at how fast the legs are moving and what the body position is.

**Simple Examples:**
- **WALKING**: Medium leg movement → Person is walking
- **RUNNING**: Fast leg movement → Person is running  
- **SITTING**: Little/no leg movement → Person is sitting

---

## Getting Started 🚀

### Step 1: Install Required Software
Copy and paste this in your terminal:
```bash
pip install -r requirements.txt
```

This installs all the tools the program needs to work.

### Step 2: Set Up Your Camera
**Option A: Use Phone Camera** (Recommended)
1. Install "IP Webcam" app on your Android phone
2. Open the app and click "Start server"
3. Note the IP address shown (like `192.168.1.100:8080`)
4. Open [camera/ip_camera.py](camera/ip_camera.py) and change:
   ```python
   url = "http://YOUR_PHONE_IP:8080/video"
   ```
   Replace `YOUR_PHONE_IP` with your actual phone IP

**Option B: Use Laptop Webcam**
- No setup needed! Program will use built-in camera automatically

### Step 3: Run the Program
```bash
python main.py
```

The system will start and show:
```
✅ Activity + Eye-based Sleep Monitoring Started
[1] Activity: WALKING | Sleep: NO
[2] Activity: SITTING | Sleep: NO
[3] Activity: SITTING | Sleep: YES ⚠️ SLEEPING DETECTED
```

---

## Different Ways to Run the System 🎯

### 🔹 Full Monitoring (Activity + Sleep)
```bash
python main.py
```
**What it does:** Detects what activity the person is doing AND checks if they're sleeping.
**Best for:** Complete monitoring of a person's behavior.

---

### 🔹 Sleep Monitoring Only
```bash
python main_sleep_only.py
```
**What it does:** Only checks if the person is sleeping (faster, simpler).
**Best for:** When you only care about sleep detection.

---

### 🔹 Extract Training Data
```bash
python extract_pose_from_dataset.py
```
**What it does:** Pulls movement data from videos for training the AI.
**Best for:** Creating new activity models.

---

## Understanding the Settings ⚙️

These settings control how sensitive the system is:

```python
RUN_THRESHOLD = 0.030      # How fast = running? (Higher = stricter)
WALK_THRESHOLD = 0.008     # How fast = walking? (Higher = stricter)
SLEEP_TIME = 10            # Seconds eyes closed = sleeping? (Higher = waits longer)
EAR_THRESHOLD = 0.20       # Eye opening distance (Higher = needs wider open eyes)
```

**Simple Explanation:**
- If you see too many false "running" detections → **Increase** `RUN_THRESHOLD`
- If sleep detection is too sensitive → **Increase** `SLEEP_TIME` or `EAR_THRESHOLD`
- If it misses activities → **Decrease** the thresholds

---

## Troubleshooting 🔧

### ❌ Problem: Camera Not Working
**Solution:**
1. Check phone and computer are on **same WiFi network**
2. Make sure IP Webcam app is **running** on your phone
3. Copy the correct IP address from the app into `ip_camera.py`
4. Try accessing the camera in your browser: `http://YOUR_IP:8080` (should show video)

---

### ❌ Problem: Bad Activity Detection
**Solution:**
1. Make sure there's **good lighting** in the room
2. Move camera to get **full body** in view
3. Wait a few seconds for system to "warm up"
4. Try increasing `MOTION_AVG_FRAMES` to 15-20 for smoother detection

---

### ❌ Problem: Too Many False Sleep Alerts
**Solution:**
1. Increase `SLEEP_TIME` from 10 to 15-20 seconds
2. Increase `EAR_THRESHOLD` from 0.20 to 0.25-0.30
3. Make sure camera can clearly see the person's eyes

---

### ❌ Problem: Program Running Slow
**Solution:**
1. This is **normal** - the system processes many calculations per frame
2. Make sure no other apps are using the camera
3. Close other programs to free up computer memory
4. No GPU? Don't worry, it still works but slower (15-30 frames/second is normal)

---

## What Files Do What? 📄

| File | Purpose |
|------|---------|
| [main.py](main.py) | Main program - run this to start monitoring |
| [camera/ip_camera.py](camera/ip_camera.py) | Connects to camera |
| [pose/pose_detector.py](pose/pose_detector.py) | Finds body joints in video |
| [activity_agent/activity_Agent.py](activity_agent/activity_Agent.py) | Decides what activity person is doing |
| [models/activity_lstm.pt](models/activity_lstm.pt) | AI brain for activity recognition |
| requirements.txt | List of software needed |

---

## Quick Reference 📋

**Want to start quickly?**
1. `pip install -r requirements.txt` ← Install software
2. Set up your camera in `camera/ip_camera.py`
3. `python main.py` ← Run the system!

**That's it!** 🎉

---

## How Activity Detection Works (Simple Explanation) 🏃

```
Person walking:
  Frame 1: Left foot at position A
  Frame 2: Left foot at position B
  
  Distance from A to B = Big distance → AI says "WALKING"

Person sitting:
  Frame 1: Legs at position C
  Frame 2: Legs still at position C
  
  Distance = No movement → AI says "SITTING"

Person running:
  Frame 1: Legs at position D
  Frame 2: Legs at position E (very far from D!)
  
  Distance = Very big distance → AI says "RUNNING"
```

The AI counts how far the legs move and decides the activity!

---

## How Sleep Detection Works (Simple Explanation) 😴

```
👁️ Look at eyes → Measure distance between eyelids

Big distance = AWAKE (eyes open)
Small distance = SLEEPING (eyes closed)

If eyes stay closed for 10 seconds → ⚠️ ALERT: SLEEPING!
```

---

## Important Notes ⚠️

✅ **Works on:** Windows, Mac, Linux  
✅ **Needs:** Python 3.7+ installed  
✅ **Camera:** Phone camera OR laptop webcam  
⚠️ **Privacy:** No video is saved, only activity predictions are tracked  
⚠️ **Research Use:** Follow your organization's privacy guidelines  

---

## Need Help? 🆘

- Check the **Troubleshooting** section above
- Make sure your camera IP is correct
- Verify good lighting in the room
- Check that Python installed correctly: `python --version`

---

Made with ❤️ for behavior analysis and monitoring
