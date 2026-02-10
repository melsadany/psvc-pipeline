// main.js

let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let audioContext = null;
let videoEl = null;
let requestMicBtn = null;
let startTaskBtn = null;
let statusEl = null;
let downloadLink = null;

// Hardcoded task video filename (must be in same folder)
const TASK_VIDEO_FILE = "task_video.mp4";

document.addEventListener("DOMContentLoaded", () => {
  videoEl = document.getElementById("taskVideo");
  requestMicBtn = document.getElementById("requestMic");
  startTaskBtn = document.getElementById("startTask");
  statusEl = document.getElementById("status");
  downloadLink = document.getElementById("downloadLink");

  // Load the hardcoded video file
  videoEl.src = TASK_VIDEO_FILE;
  videoEl.load();
  statusEl.textContent = "Task video loaded. Enable microphone to continue.";

  // Ask for microphone permission
  requestMicBtn.addEventListener("click", async () => {
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      statusEl.textContent = "Microphone enabled. Click 'Start task' when ready.";
      startTaskBtn.disabled = false;
      requestMicBtn.disabled = true;
    } catch (err) {
      console.error("Error getting microphone:", err);
      statusEl.textContent = "Microphone access denied or unavailable.";
    }
  });

  // Start task: play video + record audio
  startTaskBtn.addEventListener("click", () => {
    if (!audioStream) {
      statusEl.textContent = "Microphone not enabled.";
      return;
    }

    audioChunks = [];
    try {
      // Use a supported mime type
      const mimeType = getSupportedMimeType();
      mediaRecorder = new MediaRecorder(audioStream, { mimeType });
    } catch (e) {
      console.error("Failed to create MediaRecorder:", e);
      statusEl.textContent = "Recording not supported in this browser.";
      return;
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      statusEl.textContent = "Processing audio... please wait.";
      
      // Create blob from recorded chunks
      const webmBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      
      // Convert to MP3
      try {
        const mp3Blob = await convertToMp3(webmBlob);
        const url = URL.createObjectURL(mp3Blob);
        
        downloadLink.href = url;
        downloadLink.download = "participant_audio.mp3";
        downloadLink.style.display = "inline-block";
        
        statusEl.textContent = "Task complete! Download your MP3 recording below.";
      } catch (err) {
        console.error("MP3 conversion failed:", err);
        statusEl.textContent = "Failed to convert to MP3. Please try again.";
      }
    };

    // Start recording and play video
    mediaRecorder.start();
    videoEl.currentTime = 0;
    videoEl.play();
    statusEl.textContent = "Task running • Recording audio...";
    startTaskBtn.disabled = true;
    downloadLink.style.display = "none";
  });

  // Stop recording when video ends
  videoEl.addEventListener("ended", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  });
});

// Helper: pick supported audio mime type
function getSupportedMimeType() {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg"
  ];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

// Convert WebM/OGG audio blob to MP3 using lamejs
async function convertToMp3(audioBlob) {
  // Decode the recorded audio to raw PCM
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  
  // Get mono channel (or mix stereo to mono)
  const samples = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  
  // Convert float samples to Int16
  const int16Samples = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    int16Samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  
  // Encode to MP3 using lamejs
  const mp3encoder = new lamejs.Mp3Encoder(1, sampleRate, 128); // mono, 128kbps
  const mp3Data = [];
  
  const sampleBlockSize = 1152; // MP3 frame size
  for (let i = 0; i < int16Samples.length; i += sampleBlockSize) {
    const sampleChunk = int16Samples.subarray(i, i + sampleBlockSize);
    const mp3buf = mp3encoder.encodeBuffer(sampleChunk);
    if (mp3buf.length > 0) {
      mp3Data.push(mp3buf);
    }
  }
  
  // Flush remaining data
  const mp3buf = mp3encoder.flush();
  if (mp3buf.length > 0) {
    mp3Data.push(mp3buf);
  }
  
  // Create MP3 blob
  const mp3Blob = new Blob(mp3Data, { type: "audio/mp3" });
  return mp3Blob;
}
