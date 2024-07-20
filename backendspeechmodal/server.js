const express = require('express');
const tf = require('@tensorflow/tfjs');  // Use tfjs-node for server-side
const speechCommands = require('@tensorflow-models/speech-commands');
const cors = require('cors')
const multer = require('multer');
const fs = require('fs-extra');
const path = require('path');
const bodyParser = require('body-parser');
const tfio = require('@tensorflow/tfjs-node');

const app = express();
app.use(express.json());  // Middleware to parse JSON request bodies
app.use(cors());
const port = 5000;
app.use(bodyParser.json({ limit: '50mb' }));

let recognizer;

async function loadModel() {
  // Create the recognizer
  recognizer = speechCommands.create('BROWSER_FFT');
  
  // Ensure the model is loaded
  await recognizer.ensureModelLoaded();
  console.log('Model loaded');
  
}

loadModel().then(() => {
  // After the model is loaded, inspect the sampling frequency and FFT size
  console.log(`Sample Rate Hz: ${recognizer.params().sampleRateHz}`);
  console.log(`FFT Size: ${recognizer.params().fftSize}`);
  console.log("words ",recognizer.wordLabels());

}).catch(err => {
  console.error('Error loading the model:', err);
});

// File upload code 
const uploadFolder = path.join(__dirname, 'uploads');

// Ensure upload folder exists
fs.ensureDirSync(uploadFolder);

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadFolder);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});

const upload = multer({ storage });

app.post('/upload', upload.single('file'), (req, res) => {
  res.send({ message: 'File uploaded successfully', file: req.file });
});

// audio recognizer code 

app.post('/recognize', async (req, res) => {
  try {
    const base64AudioData = req.body.audioData;
    const binaryString = Buffer.from(base64AudioData, 'base64').toString('binary');
    const len = binaryString.length;
    const uint8Array = new Uint8Array(len);

    for (let i = 0; i < len; i++) { 
      uint8Array[i] = binaryString.charCodeAt(i);
    } 

    // Convert Uint8Array to Float32Array
    const float32Array = new Float32Array(uint8Array.buffer);

    // Process the uint8Array with TensorFlow.js
    const spectrogram = await generateSpectrogram(float32Array);

    // Flatten the spectrogram to match model input shape
    const flattenedSpectrogram = spectrogram.flat();

    // Get the model input shape
    const modelInputShape = [1, 43, 232, 1]; // Update this if needed
    const requiredValues = modelInputShape[1] * modelInputShape[2] * modelInputShape[3];

    // Pad or trim the flattened spectrogram to match the required input shape
    if (flattenedSpectrogram.length < requiredValues) {
      while (flattenedSpectrogram.length < requiredValues) {
        flattenedSpectrogram.push(0);
      }
    } else if (flattenedSpectrogram.length > requiredValues) {
      flattenedSpectrogram.length = requiredValues;
    }

    // Normalize the tensor values to avoid NaN values
    const maxVal = Math.max(...flattenedSpectrogram.filter(value => !isNaN(value) && isFinite(value)));
    const normalizedSpectrogram = flattenedSpectrogram.map(value => (isNaN(value) || !isFinite(value) ? 0 : value / maxVal));

    // Create a 4D tensor from normalized spectrogram
    const inputShape = [1, modelInputShape[1], modelInputShape[2], modelInputShape[3]];
    const x = tf.tensor4d(normalizedSpectrogram, inputShape);

    // Log the tensor values
    console.log('Tensor values:', x.dataSync());

    // Make a prediction using the recognizer
    const recognizer = await speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    const output = await recognizer.recognize(x);

    // Dispose tensors
    tf.dispose([x, output]);
        // Map the scores to words
        const words = [
          '_background_noise_', '_unknown_',
          'down',               'eight',
          'five',               'four',
          'go',                 'left',
          'nine',               'no',
          'one',                'right',
          'seven',              'six',
          'stop',               'three',
          'two',                'up',
          'yes',                'zero'
        ];
        const scores = output.scores;
        const recognizedWords = words.map((word, index) => ({
          word,
          score: scores[index]
        }));

    // Find the word with the highest score
    const recognizedWord = recognizedWords.reduce((a, b) => (a.score > b.score ? a : b));

    console.log('Recognized Words:', recognizedWords);
    console.log('Recognized Word:', recognizedWord);
    
    res.json({ words: recognizedWord });
  } catch (err) {
    console.error('Error recognizing speech:', err);
    res.status(500).send('Error recognizing speech');
  }
});

async function generateSpectrogram(audioBuffer) {
  const audioTensor = tf.tensor(audioBuffer, [audioBuffer.length]);

  const nfft = 512;
  const window = 256;
  const stride = 128;

  const frameStep = stride;
  const frameLength = window;

  const numFrames = Math.floor((audioTensor.size - frameLength) / frameStep) + 1;
  const stft = [];

  for (let i = 0; i < numFrames; i++) {
    const frame = audioTensor.slice([i * frameStep], [frameLength]);
    const windowedFrame = frame.mul(tf.signal.hannWindow(frameLength));
    const complexFrame = tf.complex(windowedFrame, tf.zerosLike(windowedFrame));
    const frameFFT = complexFrame.fft().abs();
    stft.push(frameFFT);
  }

  const spectrogramTensor = tf.stack(stft).expandDims(-1); // Expand dims to match model input
  return spectrogramTensor.arraySync();
}



app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

