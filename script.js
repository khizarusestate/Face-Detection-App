// ============================================================================
// Face Recognition System - Main JavaScript
// ============================================================================

// Global variables
let video, canvas, ctx;
let faceDetection;
let isEnrolling = false;
let isRecognizing = false;
let recognitionModel = null;

// Database for storing face embeddings
const faceDatabase = {
  people: new Map(), // Map<name, {samples: [[embedding]], avgEmbedding: []}>

  addPerson(name) {
    if (!this.people.has(name)) {
      this.people.set(name, { samples: [], avgEmbedding: null });
    }
  },

  addSample(name, embedding) {
    this.addPerson(name);
    const person = this.people.get(name);
    person.samples.push(embedding);
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

function drawDetections(detections, mode) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!detections || detections.detections.length === 0) {
    updateFaceCount(0);
    return;
  }

  updateFaceCount(detections.detections.length);

  for (const detection of detections.detections) {
    const bbox = detection.boundingBox;
    const confidence = detection.categories[0].score;

    const x = bbox.originX * canvas.width;
    const y = bbox.originY * canvas.height;
    const width = bbox.width * canvas.width;
    const height = bbox.height * canvas.height;

    // Draw bounding box
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);

    // Draw confidence
    const label = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    ctx.fillStyle = "#00ff00";
    ctx.font = "bold 14px Arial";
    ctx.fillText(label, x, y - 10);

    // Draw keypoints if available
    if (detection.keypoints && detection.keypoints.length > 0) {
      ctx.fillStyle = "#ff0000";
      for (const keypoint of detection.keypoints) {
        const kx = keypoint.x * canvas.width;
        const ky = keypoint.y * canvas.height;
        ctx.beginPath();
        ctx.arc(kx, ky, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
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

        // Add to database
        faceDatabase.addSample(personName, embedding);

        detected = true;
        showStatus(
          enrollmentStatus,
          `✓ Face enrolled for ${personName}!`,
          "success",
        );
        updateDatabaseDisplay();

        // Clear input
        document.getElementById("personName").value = "";
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
  document.getElementById("recognitionLabel").classList.remove("active");
  showStatus(
    document.getElementById("recognitionStatus"),
    "Recognition stopped.",
    "info",
  );
}

async function recognitionLoop() {
  if (!isRecognizing) return;

  const detections = await detectFace(video);

  if (detections && detections.detections.length > 0) {
    const detection = detections.detections[0];
    const landmarks = detection.keypoints || [];

    if (landmarks.length > 0) {
      const embedding = embeddingGenerator.generateEmbedding(landmarks);
      const match = faceDatabase.findClosestMatch(embedding, (threshold = 0.5));

      displayRecognitionResult(match);
    }
  } else {
    document.getElementById("recognitionLabel").classList.remove("active");
  }

  drawDetections(detections);
  requestAnimationFrame(recognitionLoop);
}

function displayRecognitionResult(match) {
  const label = document.getElementById("recognitionLabel");
  const confidence = (match.confidence * 100).toFixed(1);

  if (match.name === "Unknown") {
    label.textContent = "❓ Unknown";
    label.style.borderColor = "#ff6b6b";
    label.style.color = "#ff6b6b";
  } else {
    label.textContent = `✓ ${match.name}\n${confidence}%`;
    label.style.borderColor = "#00ff00";
    label.style.color = "#00ff00";
  }

  label.classList.add("active");
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
