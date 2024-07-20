const express = require('express');
const tf = require('@tensorflow/tfjs');  // Use tfjs-node for server-side
const speechCommands = require('@tensorflow-models/speech-commands');
const cors = require('cors')
const multer = require('multer');
const fs = require('fs-extra');
const path = require('path');
const bodyParser = require('body-parser');
const swagger = require('swagger-ui-express');
const apiDocs =  require("./swagger.json");


const app = express();
app.use(express.json());  // Middleware to parse JSON request bodies
app.use(cors());
const port = 5000;
app.use(bodyParser.json({ limit: '50mb' }));
app.use('/docs', swagger.serve, swagger.setup(apiDocs));



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
/**
 * Route handler for uploading files.
 * Responds with a message indicating the file was uploaded successfully.
 * @async
 * @name post/upload
 * @function
 * @inner
 * @param {Request} req - The request object.
 * @param {Response} res - The response object.
 * @returns {Promise<void>}
 */
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    const filePath = req.file.path;

    // Read the uploaded file
    const audioData = await fs.readFile(filePath);
    const audioBuffer = new Uint8Array(audioData);

    // Log the audio buffer
    console.log('Audio Buffer:', audioBuffer);

    // Process the audio buffer
    const recognizedWord = await processAudioBuffer(audioBuffer);

    console.log('Recognized Words:',recognizedWord);

    res.json({ word: recognizedWord });
  } catch (err) {
    console.error('Error processing audio:', err);
    res.status(500).send('Error processing audio');
  }
});
/**
 * Generates a spectrogram from an audio buffer.
 *
 * @async
 * @function processAudioBuffer
 * @param {Uint8Array} audioBuffer - The audio buffer to process.
 * @returns {Promise<Array<{ word: string, score: number }>>} The recognized words with their scores.
 */
async function processAudioBuffer(audioBuffer) {
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

  // Log the spectrogram tensor
  console.log('Spectrogram Tensor:', spectrogramTensor.arraySync());

  // Flatten the spectrogram to match model input shape
  const flattenedSpectrogram = spectrogramTensor.flatten().arraySync();

    // Log the flattened spectrogram
    console.log('Flattened Spectrogram:', flattenedSpectrogram);

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

  // Log the input tensor
  console.log('Input Tensor:', x.arraySync());

  // Load the recognizer model
  const recognizer = await speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();

  // Make a prediction using the recognizer
  const output = await recognizer.recognize(x);

  // Dispose tensors
  tf.dispose([x, output]);

    // Log the output tensor
    console.log('Output Tensor:', output.scores);

  // Map the scores to words
  const words = [
    '_background_noise_', '_unknown_',
    'down', 'eight', 'five', 'four', 'go', 'left',
    'nine', 'no', 'one', 'right', 'seven', 'six',
    'stop', 'three', 'two', 'up', 'yes', 'zero'
  ];
  const scores = output.scores;
  const recognizedWords = words.map((word, index) => ({
    word,
    score: scores[index]
  }));
  console.log('Recognized Words:', recognizedWords);

  // Filter words with a certain score threshold
  const scoreThreshold = 0.05; // Adjust this threshold as needed
  const filteredRecognizedWords = recognizedWords.filter(w => w.score >= scoreThreshold);

  console.log('Filtered Recognized Words:', filteredRecognizedWords);

  return filteredRecognizedWords;
}



app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

