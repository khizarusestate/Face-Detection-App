# ⚙️ Setup Guide - Face Recognition System

## 🚨 Permission Error Fix

**The app requires running on a local server or HTTPS** (webcam access policy).

### Option 1: Using Node.js HTTP Server (RECOMMENDED)

```bash
# Install http-server globally (one time)
npm install -g http-server

# Start the server
cd c:\Users\hasee\OneDrive\Desktop\Portfolio-2-main\face-recognition-app
http-server

# Open in browser:
# http://localhost:8080
```

### Option 2: Using Python

#### Python 3.x

```bash
cd c:\Users\hasee\OneDrive\Desktop\Portfolio-2-main\face-recognition-app
python -m http.server 8000

# Open: http://localhost:8000
```

#### Python 2.x

```bash
python -m SimpleHTTPServer 8000
```

### Option 3: Using VS Code Live Server Extension

1. Install "Live Server" extension in VS Code
2. Right-click on `index.html` → "Open with Live Server"
3. Browser opens automatically (usually on http://127.0.0.1:5500)

---

## ✅ Fixing Permission Issues

### If you see "Permission Denied":

1. Look for the **🎥 camera icon** in your browser's address bar
2. Click it and select **Allow** for camera access
3. Refresh the page

### If you see "Camera In Use":

- Close other apps using the camera (e.g., Zoom, Teams, Discord, OBS, etc.)
- Refresh the page

### If you see "No Camera Found":

- Your device doesn't have a webcam connected
- Connect a USB webcam if available

---

## 🎯 Quick Start

1. **Open Terminal**

   ```bash
   cd c:\Users\hasee\OneDrive\Desktop\Portfolio-2-main\face-recognition-app
   npx http-server
   ```

2. **Open Browser**: http://localhost:8080 (check terminal for the actual port)

3. **Grant Camera Permission** when prompted

4. **Start Using**: Add faces → Recognize faces

---

## 📝 Notes

- **All data is stored locally** in your browser (localStorage) - nothing sent to servers
- **First time only**: Browser will ask for camera permission
- **If permission is denied**: Clear browser cookies/site data and try again
