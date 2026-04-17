// ============================================================================
// Face Recognition System - Main JavaScript
// ============================================================================

// Global variables
let video, canvas, ctx;
let faceDetection;
let handDetection;
let isEnrolling = false;
let isRecognizing = false;
let recognitionModel = null;

// Database for storing face embeddings
const faceDatabase = {
  people: new Map(), // Map<name, {samples: [[embedding]], avgEmbedding: [], age, weight, nationality}>

  addPerson(name, age, weight, nationality) {
    if (!this.people.has(name)) {
      this.people.set(name, {
        samples: [],
        avgEmbedding: null,
        age: age || null,
        weight: weight || null,
        nationality: nationality || null,
      });
    }
  },

  addSample(name, embedding, age, weight, nationality) {
    this.addPerson(name, age, weight, nationality);
    const person = this.people.get(name);
    person.samples.push(embedding);
    person.age = age || person.age;
    person.weight = weight || person.weight;
    person.nationality = nationality || person.nationality;
    this.updateAverage(name);
    this.save();
  },

  updateAverage(name) {
    if (!this.people.has(name)) return;
    const person = this.people.get(name);
    if (person.samples.length === 0) return;

    const dimension = person.samples[0].length;
    const average = new Array(dimension).fill(0);

    for (const sample of person.samples) {
      for (let i = 0; i < dimension; i++) {
        average[i] += sample[i];
      }
    }

    for (let i = 0; i < dimension; i++) {
      average[i] /= person.samples.length;
    }

    person.avgEmbedding = average;
  },

  findClosestMatch(embedding, threshold = 0.6) {
    let closestName = null;
    let closestDistance = Infinity;

    for (const [name, person] of this.people.entries()) {
      if (!person.avgEmbedding) continue;
      const distance = this.euclideanDistance(embedding, person.avgEmbedding);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestName = name;
      }
    }

    // Normalize distance (0-1 scale) - lower is better
    // Using sigmoid-like function to convert distance to similarity score
    const similarity = 1 / (1 + closestDistance);

    if (similarity < threshold) {
      return { name: "Unknown", confidence: 0 };
    }

    return { name: closestName, confidence: similarity };
  },

  euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  },

  cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (normA * normB);
  },

  save() {
    const data = {};
    for (const [name, person] of this.people.entries()) {
      data[name] = {
        samples: person.samples,
        avgEmbedding: person.avgEmbedding,
        age: person.age,
        weight: person.weight,
        nationality: person.nationality,
      };
    }
    localStorage.setItem("faceDatabase", JSON.stringify(data));
  },

  load() {
    const data = localStorage.getItem("faceDatabase");
    if (!data) return;

    try {
      const parsed = JSON.parse(data);
      for (const [name, personData] of Object.entries(parsed)) {
        const person = {
          samples: personData.samples || [],
          avgEmbedding: personData.avgEmbedding || null,
          age: personData.age || null,
          weight: personData.weight || null,
          nationality: personData.nationality || null,
        };
        this.people.set(name, person);
      }
    } catch (e) {
      console.error("Error loading database:", e);
    }
  },

  clear() {
    this.people.clear();
    localStorage.removeItem("faceDatabase");
  },

  getStats() {
    return {
      peopleCount: this.people.size,
      totalSamples: Array.from(this.people.values()).reduce(
        (sum, p) => sum + p.samples.length,
        0,
      ),
    };
  },
};

// ============================================================================
// Embedding Generation with Feature Extraction
// ============================================================================

class EmbeddingGenerator {
  constructor() {
    this.inputSize = 256; // Output embedding size
  }

  // Generate a deterministic embedding from face detection landmarks
  generateEmbedding(landmarks) {
    try {
      if (!landmarks || landmarks.length === 0) {
        return this.generateRandomEmbedding();
      }

      // Use face landmarks to create a feature vector
      const features = this.extractFeatures(landmarks);

      // Normalize features
      const normalized = this.normalize(features);

      return normalized;
    } catch (e) {
      console.error("Error generating embedding:", e);
      return this.generateRandomEmbedding();
    }
  }

  extractFeatures(landmarks) {
    // Extract meaningful features from face landmarks
    const features = [];

    // Add landmark coordinates
    for (const point of landmarks) {
      features.push(point.x, point.y);
    }

    // Add pairwise distances (geometric features)
    for (let i = 0; i < Math.min(landmarks.length, 10); i++) {
      for (let j = i + 1; j < Math.min(landmarks.length, 10); j++) {
        const dx = landmarks[i].x - landmarks[j].x;
        const dy = landmarks[i].y - landmarks[j].y;
        features.push(Math.sqrt(dx * dx + dy * dy));
      }
    }

    // Add angles between points
    for (let i = 0; i < Math.min(landmarks.length - 2, 8); i++) {
      const p1 = landmarks[i];
      const p2 = landmarks[i + 1];
      const p3 = landmarks[i + 2];

      const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
      const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };

      const angle = Math.atan2(v2.y, v2.x) - Math.atan2(v1.y, v1.x);
      features.push(Math.cos(angle), Math.sin(angle));
    }

    // Pad or truncate to fixed size
    while (features.length < this.inputSize) {
      features.push(0);
    }
    features.length = this.inputSize;

    return features;
  }

  // Normalize features to unit length
  normalize(vector) {
    let sum = 0;
    for (const val of vector) {
      sum += val * val;
    }
    const norm = Math.sqrt(sum);

    if (norm === 0) return vector;

    return vector.map((val) => val / norm);
  }

  generateRandomEmbedding() {
    const embedding = new Array(this.inputSize);
    for (let i = 0; i < this.inputSize; i++) {
      embedding[i] = Math.random();
    }
    return this.normalize(embedding);
  }
}

const embeddingGenerator = new EmbeddingGenerator();

// ============================================================================
// MediaPipe Face Detection Setup
// ============================================================================

async function setupFaceDetection() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
  );

  faceDetection = await FaceDetector.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    },
    runningMode: "VIDEO",
  });
}

// ============================================================================
// MediaPipe Hand Tracking Setup
// ============================================================================

async function setupHandDetection() {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
    );

    handDetection = await HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
  } catch (error) {
    console.warn("Hand detection not available:", error);
  }
}

// ============================================================================
// Face Detection and Drawing
// ============================================================================

async function detectFace(image) {
  try {
    if (!faceDetection) return null;

    const detections = faceDetection.detectForVideo(image, performance.now());
    return detections;
  } catch (e) {
    console.error("Face detection error:", e);
    return null;
  }
}

// Detect hands and count fingers
async function detectHands(image) {
  try {
    if (!handDetection) return null;
    const results = handDetection.detectForVideo(image, performance.now());
    return results;
  } catch (e) {
    console.warn("Hand detection error:", e);
    return null;
  }
}

// Count visible fingers from hand landmarks
function countFingers(handLandmarks) {
  if (!handLandmarks || handLandmarks.length === 0) return 0;

  // Finger tip indices in MediaPipe hand landmarks
  const fingertipIndices = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky
  const pipIndices = [3, 6, 10, 14, 18]; // PIP joints (one level before tip)

  let fingerCount = 0;

  // Check each finger
  for (let i = 0; i < fingertipIndices.length; i++) {
    const tipIndex = fingertipIndices[i];
    const pipIndex = pipIndices[i];

    if (handLandmarks[tipIndex] && handLandmarks[pipIndex]) {
      // Finger is extended if tip is above (lower y) than PIP joint
      const tipY = handLandmarks[tipIndex].y;
      const pipY = handLandmarks[pipIndex].y;

      if (tipY < pipY - 0.05) {
        fingerCount++;
      }
    }
  }

  return fingerCount;
}

// Detect facial expression from landmarks
function detectFacialExpression(landmarks) {
  if (!landmarks || landmarks.length < 10) return "neutral";

  try {
    // Simple expression detection based on landmark positions
    // Mouth landmarks: 61, 62, 63, 64 (upper lip), 65, 66, 67, 68, 69, 70 (lower lip)
    // Eye landmarks: 33, 130, 8, 246, 161, 160, 159, 158 (left eye)
    // Eye landmarks: 263, 362, 382, 381, 380, 374, 33, 246 (right eye)

    let smileIndicator = 0;
    let surprised = 0;

    // Check if mouth is open/closed and corners are up (smile)
    if (landmarks.length > 70) {
      const mouthCornerLeft = landmarks[61]; // Left mouth corner
      const mouthCornerRight = landmarks[291]; // Right mouth corner
      const mouthBottom = landmarks[17]; // Bottom mouth
      const mouthTop = landmarks[13]; // Top mouth

      if (mouthCornerLeft && mouthCornerRight && mouthBottom) {
        // Smile detection: mouth corners up
        if (
          mouthCornerLeft.y < mouthBottom.y &&
          mouthCornerRight.y < mouthBottom.y
        ) {
          smileIndicator++;
        }
      }

      if (mouthTop && mouthBottom) {
        const mouthOpenness = Math.abs(mouthTop.y - mouthBottom.y);
        if (mouthOpenness > 0.05) {
          surprised++;
        }
      }
    }

    // Check eye openness
    if (landmarks.length > 250) {
      const leftEyeTop = landmarks[159];
      const leftEyeBottom = landmarks[145];
      const rightEyeTop = landmarks[386];
      const rightEyeBottom = landmarks[374];

      if (leftEyeTop && leftEyeBottom && rightEyeTop && rightEyeBottom) {
        const leftEyeOpen = Math.abs(leftEyeTop.y - leftEyeBottom.y);
        const rightEyeOpen = Math.abs(rightEyeTop.y - rightEyeBottom.y);

        if (leftEyeOpen > 0.03 && rightEyeOpen > 0.03) {
          surprised++;
        }
      }
    }

    if (surprised > 1) return "surprised 😮";
    if (smileIndicator > 0) return "smiling 😊";
    return "neutral 😐";
  } catch (e) {
    return "neutral 😐";
  }
}

function drawDetections(detections, hands, mode) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const faceCount =
    detections && detections.detections.length
      ? detections.detections.length
      : 0;
  updateFaceCount(faceCount);

  // Draw up to 3 faces
  if (detections && detections.detections.length > 0) {
    const facesToDraw = Math.min(detections.detections.length, 3);

    for (let faceIdx = 0; faceIdx < facesToDraw; faceIdx++) {
      const detection = detections.detections[faceIdx];
      const bbox = detection.boundingBox;
      const confidence = detection.categories[0].score;

      const x = bbox.originX * canvas.width;
      const y = bbox.originY * canvas.height;
      const width = bbox.width * canvas.width;
      const height = bbox.height * canvas.height;

      // Draw bounding box
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw confidence
      const label = `Face ${faceIdx + 1} | Confidence: ${(confidence * 100).toFixed(1)}%`;
      ctx.fillStyle = "#00ff88";
      ctx.font = "bold 14px Courier New";
      ctx.fillText(label, x + 5, y - 30);

      // Detect and display facial expression
      if (detection.keypoints && detection.keypoints.length > 0) {
        const expression = detectFacialExpression(detection.keypoints);
        ctx.fillStyle = "#00ffcc";
        ctx.font = "bold 12px Courier New";
        ctx.fillText(`Expression: ${expression}`, x + 5, y - 10);
      }

      // Draw keypoints
      if (detection.keypoints && detection.keypoints.length > 0) {
        ctx.fillStyle = "#ff00ff";
        for (const keypoint of detection.keypoints) {
          const kx = keypoint.x * canvas.width;
          const ky = keypoint.y * canvas.height;
          ctx.beginPath();
          ctx.arc(kx, ky, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
  }

  // Draw hands and finger count
  if (hands && hands.landmarks && hands.landmarks.length > 0) {
    for (let handIdx = 0; handIdx < hands.landmarks.length; handIdx++) {
      const handLandmarks = hands.landmarks[handIdx];
      const fingerCount = countFingers(handLandmarks);

      // Draw hand landmarks
      ctx.fillStyle = "#ffff00";
      for (const landmark of handLandmarks) {
        const lx = landmark.x * canvas.width;
        const ly = landmark.y * canvas.height;
        ctx.beginPath();
        ctx.arc(lx, ly, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw connections between landmarks
      ctx.strokeStyle = "#ffff00";
      ctx.lineWidth = 2;

      // Connection indices for hand
      const connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4], // Thumb
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8], // Index
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12], // Middle
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16], // Ring
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20], // Pinky
      ];

      for (const [start, end] of connections) {
        if (handLandmarks[start] && handLandmarks[end]) {
          const x1 = handLandmarks[start].x * canvas.width;
          const y1 = handLandmarks[start].y * canvas.height;
          const x2 = handLandmarks[end].x * canvas.width;
          const y2 = handLandmarks[end].y * canvas.height;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }

      // Display finger count at hand position
      if (handLandmarks[0]) {
        const handX = handLandmarks[0].x * canvas.width;
        const handY = handLandmarks[0].y * canvas.height;

        ctx.fillStyle = "#ffff00";
        ctx.font = "bold 16px Courier New";
        ctx.fillText(`Fingers: ${fingerCount}`, handX + 10, handY - 20);
      }
    }
  } else {
    // No hands detected
    if (faceCount > 0) {
      ctx.fillStyle = "#ff6600";
      ctx.font = "bold 14px Courier New";
      ctx.fillText("Fingers: 0", 20, canvas.height - 20);
    }
  }
}

function updateFaceCount(count) {
  document.getElementById("faceCount").textContent = `Faces detected: ${count}`;
}

// ============================================================================
// Enrollment Process
// ============================================================================

async function enrollFace() {
  const personName = document.getElementById("personName").value.trim();
  const personAge = document.getElementById("personAge").value.trim();
  const personWeight = document.getElementById("personWeight").value.trim();
  const personNationality = document
    .getElementById("personNationality")
    .value.trim();
  const enrollmentStatus = document.getElementById("enrollmentStatus");

  if (!personName) {
    showStatus(enrollmentStatus, "Please enter a name", "error");
    return;
  }

  if (isEnrolling) {
    showStatus(enrollmentStatus, "Already enrolling...", "warning");
    return;
  }

  isEnrolling = true;
  showStatus(enrollmentStatus, "Looking for face...", "info");

  // Try to detect face
  let detected = false;
  let attempts = 0;
  const maxAttempts = 30; // 3 seconds at 10fps

  while (!detected && attempts < maxAttempts) {
    const detections = await detectFace(video);

    if (detections && detections.detections.length > 0) {
      const detection = detections.detections[0];

      // Extract landmarks from detection
      const landmarks = detection.keypoints || [];
      if (landmarks.length > 0) {
        // Generate embedding
        const embedding = embeddingGenerator.generateEmbedding(landmarks);

        // Add to database with all fields
        faceDatabase.addSample(
          personName,
          embedding,
          personAge || null,
          personWeight || null,
          personNationality || null,
        );

        detected = true;
        showStatus(
          enrollmentStatus,
          `✓ Face enrolled for ${personName}!`,
          "success",
        );
        updateDatabaseDisplay();

        // Clear inputs
        document.getElementById("personName").value = "";
        document.getElementById("personAge").value = "";
        document.getElementById("personWeight").value = "";
        document.getElementById("personNationality").value = "";
      }
    }

    attempts++;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  if (!detected) {
    showStatus(
      enrollmentStatus,
      "No face detected. Please try again.",
      "error",
    );
  }

  isEnrolling = false;
}

// ============================================================================
// Recognition Process
// ============================================================================

async function startRecognition() {
  if (faceDatabase.people.size === 0) {
    showStatus(
      document.getElementById("recognitionStatus"),
      "No enrolled faces. Please add faces first.",
      "error",
    );
    return;
  }

  isRecognizing = true;
  document.getElementById("startRecognitionBtn").disabled = true;
  document.getElementById("stopRecognitionBtn").disabled = false;
  document.getElementById("mode").textContent = "Mode: Recognition";
  showStatus(
    document.getElementById("recognitionStatus"),
    "Recognition started...",
    "info",
  );

  recognitionLoop();
}

function stopRecognition() {
  isRecognizing = false;
  document.getElementById("startRecognitionBtn").disabled = false;
  document.getElementById("stopRecognitionBtn").disabled = true;
  document.getElementById("mode").textContent = "Mode: Idle";
  document.getElementById("detectionDetails").innerHTML = "";
  showStatus(
    document.getElementById("recognitionStatus"),
    "Recognition stopped.",
    "info",
  );
}

async function recognitionLoop() {
  if (!isRecognizing) return;

  const detections = await detectFace(video);
  const hands = await detectHands(video);

  if (detections && detections.detections.length > 0) {
    const detection = detections.detections[0];
    const landmarks = detection.keypoints || [];

    if (landmarks.length > 0) {
      const embedding = embeddingGenerator.generateEmbedding(landmarks);
      const match = faceDatabase.findClosestMatch(embedding, (threshold = 0.5));

      // Detect facial expression
      const expression = detectFacialExpression(landmarks);

      // Count fingers from hands
      let fingerCount = 0;
      if (hands && hands.landmarks && hands.landmarks.length > 0) {
        fingerCount = countFingers(hands.landmarks[0]);
      }

      displayRecognitionResult(match, expression, fingerCount);
    }
  } else {
    document.getElementById("detectionDetails").innerHTML = "";
  }

  drawDetections(detections, hands);
  requestAnimationFrame(recognitionLoop);
}

function displayRecognitionResult(match, expression, fingerCount) {
  // Display in the right-side info panel instead of on camera
  const detailsPanel = document.getElementById("detectionDetails");
  const confidence = (match.confidence * 100).toFixed(1);

  let displayHTML = "";

  if (match.name === "Unknown") {
    displayHTML = `
      <div style="color: #ff6b6b; border: 2px solid #ff6b6b; padding: 10px; margin-top: 10px;">
        <strong>❓ Unknown Face</strong><br/>
        Confidence: ${confidence}%
      </div>
    `;
  } else {
    // Get person data from database
    const person = faceDatabase.people.get(match.name);

    displayHTML = `
      <div style="color: #00ff88; border: 2px solid #00ff88; padding: 10px; margin-top: 10px;">
        <strong>✓ ${match.name}</strong><br/>
        Confidence: ${confidence}%<br/>
    `;

    if (person) {
      if (person.age) displayHTML += `Age: ${person.age}<br/>`;
      if (person.weight) displayHTML += `Weight: ${person.weight} kg<br/>`;
      if (person.nationality)
        displayHTML += `Country: ${person.nationality}<br/>`;
    }

    if (expression) displayHTML += `Expression: ${expression}<br/>`;

    if (fingerCount > 0) {
      displayHTML += `Fingers: ${fingerCount}<br/>`;
    }

    displayHTML += `</div>`;
  }

  detailsPanel.innerHTML = displayHTML;
}

// ============================================================================
// UI Helper Functions
// ============================================================================

function showStatus(element, message, type) {
  element.textContent = message;
  element.className = `status-message show ${type}`;
}

function updateDatabaseDisplay() {
  const stats = faceDatabase.getStats();
  document.getElementById("enrolledCount").textContent = stats.peopleCount;
  document.getElementById("samplesCount").textContent = stats.totalSamples;
}

function openDatabaseModal() {
  const modal = document.getElementById("databaseModal");
  const list = document.getElementById("databaseList");
  list.innerHTML = "";

  if (faceDatabase.people.size === 0) {
    list.innerHTML =
      '<p style="text-align: center; color: #999;">No enrolled faces yet.</p>';
  } else {
    for (const [name, person] of faceDatabase.people.entries()) {
      const item = document.createElement("div");
      item.className = "database-item";
      item.innerHTML = `
                <h4>${name}</h4>
                <p>Age: ${person.age || "N/A"}</p>
                <p>Weight: ${person.weight ? person.weight + " kg" : "N/A"}</p>
                <p>Nationality: ${person.nationality || "N/A"}</p>
                <p>Samples: ${person.samples.length}</p>
                <p>Embedding dimension: ${person.avgEmbedding ? person.avgEmbedding.length : 0}</p>
                <button class="delete-person-btn" onclick="deletePerson('${name}')">Delete</button>
            `;
      list.appendChild(item);
    }
  }

  modal.classList.add("show");
}

function closeDatabaseModal() {
  document.getElementById("databaseModal").classList.remove("show");
}

function deletePerson(name) {
  if (confirm(`Delete ${name} and all their face samples?`)) {
    faceDatabase.people.delete(name);
    faceDatabase.save();
    updateDatabaseDisplay();
    openDatabaseModal(); // Refresh modal
  }
}

function resetDatabase() {
  if (
    confirm(
      "Are you sure you want to delete all enrolled faces? This cannot be undone.",
    )
  ) {
    faceDatabase.clear();
    updateDatabaseDisplay();
    showStatus(
      document.getElementById("recognitionStatus"),
      "Database cleared.",
      "success",
    );
  }
}

// ============================================================================
// Export/Import Functions
// ============================================================================

function exportDatabase() {
  const data = {};
  for (const [name, person] of faceDatabase.people.entries()) {
    data[name] = {
      samples: person.samples,
      avgEmbedding: person.avgEmbedding,
      age: person.age,
      weight: person.weight,
      nationality: person.nationality,
    };
  }

  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `face-database-${new Date().getTime()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  showStatus(
    document.getElementById("recognitionStatus"),
    "✓ Database exported successfully!",
    "success",
  );
}

function importDatabase() {
  document.getElementById("importFileInput").click();
}

function handleFileImport(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);

      if (!confirm("This will replace your current database. Continue?")) {
        return;
      }

      faceDatabase.clear();

      for (const [name, personData] of Object.entries(data)) {
        const person = {
          samples: personData.samples || [],
          avgEmbedding: personData.avgEmbedding || null,
          age: personData.age || null,
          weight: personData.weight || null,
          nationality: personData.nationality || null,
        };
        faceDatabase.people.set(name, person);
      }

      faceDatabase.save();
      updateDatabaseDisplay();
      showStatus(
        document.getElementById("recognitionStatus"),
        "✓ Database imported successfully!",
        "success",
      );
    } catch (error) {
      console.error("Import error:", error);
      showStatus(
        document.getElementById("recognitionStatus"),
        "✗ Invalid file format. Please use a JSON file exported from this app.",
        "error",
      );
    }
  };

  reader.readAsText(file);

  // Reset file input
  event.target.value = "";
}

// ============================================================================
// Initialization
// ============================================================================

async function initializeApp() {
  try {
    // Load saved database
    faceDatabase.load();
    updateDatabaseDisplay();

    // Setup video element
    video = document.getElementById("video");
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");

    // Setup face detection
    await setupFaceDetection();

    // Setup hand detection
    await setupHandDetection();

    // Request camera access
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
    });
    video.srcObject = stream;

    // Adjust canvas size when video is loaded
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    };

    // Setup event listeners
    setupEventListeners();

    // Hide loading indicator
    document.getElementById("loadingIndicator").classList.add("hidden");

    console.log("Face Recognition App initialized successfully");
  } catch (error) {
    console.error("Initialization error:", error);
    alert(
      "Failed to initialize the app. Please check permissions and try again.",
    );
    document.getElementById("loadingIndicator").classList.add("hidden");
  }
}

function setupEventListeners() {
  document.getElementById("addFaceBtn").addEventListener("click", enrollFace);
  document
    .getElementById("startRecognitionBtn")
    .addEventListener("click", startRecognition);
  document
    .getElementById("stopRecognitionBtn")
    .addEventListener("click", stopRecognition);
  document
    .getElementById("viewDatabaseBtn")
    .addEventListener("click", openDatabaseModal);
  document
    .getElementById("exportDatabaseBtn")
    .addEventListener("click", exportDatabase);
  document
    .getElementById("importDatabaseBtn")
    .addEventListener("click", importDatabase);
  document
    .getElementById("importFileInput")
    .addEventListener("change", handleFileImport);
  document
    .getElementById("resetDatabaseBtn")
    .addEventListener("click", resetDatabase);
  document
    .getElementById("closeModalBtn")
    .addEventListener("click", closeDatabaseModal);

  // Close modal on close button
  document
    .querySelector(".close")
    .addEventListener("click", closeDatabaseModal);

  // Close modal when clicking outside
  document.getElementById("databaseModal").addEventListener("click", (e) => {
    if (e.target.id === "databaseModal") {
      closeDatabaseModal();
    }
  });

  // Allow enrollment with Enter key
  document.getElementById("personName").addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      enrollFace();
    }
  });
}

// Start the app when DOM is ready
window.addEventListener("DOMContentLoaded", initializeApp);
