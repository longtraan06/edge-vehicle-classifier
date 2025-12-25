# 🚀 Quick Reference Card

## 🎯 Performance Highlights

**⚡ Edge Device Optimized**: 
- **25-30 FPS** on Jetson Nano 4GB **WITHOUT GPU**
- CPU-only inference using NCNN
- ~1.5GB RAM, ~70-80% CPU usage
- Production-ready performance

---

## Configuration Files

### 📄 `config/settings.yaml` - Technical Settings
```yaml
camera:
  source: "http://192.168.1.100/stream.m3u8"
  processing_width: 640

model:
  per_class_thresholds:
    car: 0.73
    motorbike: 0.63
    person: 0.15
    truck: 0.3

tracking:
  detection_interval: 3
  min_hits_for_classification: 5
```

### 🔐 `.env` - Sensitive Credentials
```bash
CAMERA_ID=your_camera_id
AI_MODULE_ID=your_ai_module_id
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Common Tasks

### 🎥 Change Camera Source
```yaml
# Edit config/settings.yaml
camera:
  source: "NEW_URL_HERE"
```

### 🎯 Adjust Detection Sensitivity
```yaml
# Edit config/settings.yaml
model:
  per_class_thresholds:
    car: 0.8    # Increase for fewer false positives
    truck: 0.5  # Decrease for more detections
```

### ⚡ Improve Performance (Higher FPS)
```yaml
# Edit config/settings.yaml
camera:
  processing_width: 416  # Reduce from 640 → 35-40 FPS

tracking:
  detection_interval: 5  # Increase from 3 → 30-35 FPS

output:
  save_video: false      # Disable if not needed → +2-3 FPS
```

### 🔔 Setup Telegram Alerts
1. Create bot with `@BotFather`
2. Get token: `123456789:ABCdef...`
3. Get chat ID: Send message to bot, visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Edit `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_CHAT_ID=-1001234567890
```

### 📹 Enable Video Recording
```yaml
# Edit config/settings.yaml
output:
  save_video: true
  video_output_path: "output/videos/recording.avi"
```

---

## Running the System

### First Time Setup (Jetson Nano)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install NCNN (optimized for ARM)
pip install ncnn-python

# 3. Copy environment file
cp .env.example .env

# 4. Edit .env with your credentials
nano .env

# 5. Run with web interface
python main.py

# 6. Access http://<jetson-ip>:5000
# 7. Configure ROI and tripwires
# 8. Click "Start System"
```

### Subsequent Runs
```bash
# Just run (uses saved config)
python main.py
```

---

## File Locations

| What | Where |
|------|-------|
| **Technical config** | `config/settings.yaml` |
| **API credentials** | `.env` (create from `.env.example`) |
| **Model files** | `models/model_best_data/` |
| **Vehicle logs** | `data/logs/vehicle_logs.json` |
| **System logs** | `logs/vehicle_detection.log` |
| **Recorded videos** | `output/videos/` |
| **Old backup code** | `Vehicle_cls_tracking_single.py` |

---

## Comparison: Old vs New

| Configuration | Old Location | New Location |
|--------------|--------------|--------------|
| Camera URL | Hardcoded in code | `config/settings.yaml` → `camera.source` |
| Model paths | Hardcoded in code | `config/settings.yaml` → `model.*` |
| Thresholds | `PER_CLASS_THRESHOLDS` dict | `config/settings.yaml` → `model.per_class_thresholds` |
| CAMERA_ID | Hardcoded (⚠️ security risk!) | `.env` → `CAMERA_ID` |
| BOT_TOKEN | Hardcoded (⚠️ exposed!) | `.env` → `TELEGRAM_BOT_TOKEN` |
| Detection interval | Function argument | `config/settings.yaml` → `tracking.detection_interval` |

---

## Performance Tuning Guide

### 🎯 Jetson Nano 4GB Performance Targets

| Goal | Setting | Value | Expected FPS |
|------|---------|-------|--------------|
| **Maximum FPS** | `processing_width` | 320 | 35-40 FPS |
| **Maximum FPS** | `detection_interval` | 10 | 30-35 FPS |
| **Balanced** (default) | `processing_width` | 640 | **25-30 FPS** ✅ |
| **Balanced** (default) | `detection_interval` | 3 | **25-30 FPS** ✅ |
| **Max Accuracy** | `processing_width` | 640 | 20-25 FPS |
| **Max Accuracy** | `detection_interval` | 1 | 15-20 FPS |

### General Tuning

| Goal | Setting | Recommended Value |
|------|---------|------------------|
| **Higher FPS** | `detection_interval` | 5-10 |
| **More accurate** | `detection_interval` | 1-2 |
| **Fewer false alerts** | `min_hits_for_classification` | 8-10 |
| **More sensitive detection** | `per_class_thresholds` | 0.3-0.5 |
| **Less false positives** | `per_class_thresholds` | 0.7-0.9 |

---

## Troubleshooting Quick Fixes

### ❌ ModuleNotFoundError: No module named 'cv2'
```bash
pip install opencv-python
```

### ❌ ModuleNotFoundError: No module named 'ncnn'
```bash
# For Jetson Nano / ARM
pip install ncnn-python
```

### ❌ Camera not connecting
```bash
# Test camera
python -c "import cv2; print(cv2.VideoCapture('YOUR_URL').isOpened())"
```

### ❌ Low FPS on Jetson Nano
```yaml
# config/settings.yaml
camera:
  processing_width: 416  # Reduce from 640
tracking:
  detection_interval: 5   # Increase from 3
output:
  save_video: false       # Disable recording
```

### ❌ Web interface not loading
```bash
# Check Flask installed
pip install Flask

# Access via Jetson IP, not localhost
# http://192.168.1.X:5000
```

### ❌ Alerts not sending
```bash
# Check .env file exists and has correct values
cat .env
```

### ❌ High CPU usage / thermal throttling
```bash
# Increase detection interval
# config/settings.yaml → tracking.detection_interval: 5

# Monitor temperature
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Ensure proper cooling (heatsink + fan recommended)
```

---

## Edge Device Tips

### Jetson Nano Optimization
- ✅ Use power mode MAXN: `sudo nvpmodel -m 0`
- ✅ Set CPU to max frequency: `sudo jetson_clocks`
- ✅ Install heatsink + 5V fan
- ✅ Use 5V 4A power supply (barrel jack, not micro-USB)
- ✅ Enable swap if needed: `sudo systemctl enable nvzramconfig`

### Raspberry Pi 4 Optimization
- ✅ Overclock CPU to 2.0 GHz (if stable)
- ✅ Use active cooling
- ✅ Reduce `processing_width` to 416
- ✅ Increase `detection_interval` to 5
- ✅ Expected FPS: 8-12 FPS

---

## Need More Help?

- **Full documentation**: See [README.md](README.md)
- **Migration guide**: See [MIGRATION.md](MIGRATION.md)
- **Project structure**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Test system**: Run `python test_system.py`
- **Performance benchmarks**: Check README.md § Performance Benchmarks
