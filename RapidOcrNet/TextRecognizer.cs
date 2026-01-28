// Apache-2.0 license
// Adapted from RapidAI / RapidOCR
// https://github.com/RapidAI/RapidOCR/blob/92aec2c1234597fa9c3c270efd2600c83feecd8d/dotnet/RapidOcrOnnxCs/OcrLib/CrnnNet.cs

using System.Diagnostics;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RapidOcrNet
{
    public sealed class TextRecognizer : IDisposable
    {
        private static readonly float[] MeanValues = [127.5F, 127.5F, 127.5F];
        private static readonly float[] NormValues = [1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F];
        private const int CrnnDstHeight = 48;
        //private const int CrnnCols = 6625;

        private InferenceSession _crnnNet;
        private string[] _keys;
        private string _inputName;

        public void InitModel(string path, string? keysPath, int numThread)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Recognizer model file does not exist: '{path}'.");

            var op = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                InterOpNumThreads = numThread,
                IntraOpNumThreads = numThread
            };

            _crnnNet = new InferenceSession(path, op);
            _inputName = _crnnNet.InputMetadata.Keys.First();

            if (!string.IsNullOrEmpty(keysPath))
            {
                if (!File.Exists(keysPath))
                    throw new FileNotFoundException($"Recognizer keys file does not exist: '{keysPath}'.");

                _keys = InitKeys(keysPath);
            }
            else
            {

                var meta = _crnnNet.ModelMetadata.CustomMetadataMap;
                if (meta.TryGetValue("character", out var characterData))
                {
                    _keys = InitKeys(path, characterData);

                    Console.WriteLine($"[ONNX] Loaded {_keys.Length} keys from model metadata.");
                }
            }
        }


        private static string[] InitKeys(string path, string? characterData = null)
        {
            var keys = new List<string> { "#" }; // start with special symbol

            if (!string.IsNullOrEmpty(characterData))
            {
                keys.AddRange(characterData.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries));
            }
            else
            {
                using var sr = new StreamReader(path, Encoding.UTF8);
                while (sr.ReadLine() is { } line)
                {
                    keys.Add(line);
                }
            }

            keys.Add(" ");

            System.Diagnostics.Debug.WriteLine($"keys Size = {keys.Count}");

            return keys.ToArray();
        }


        public TextLine[] GetTextLines(Image<Rgba32>[] partImgs)
        {
            var textLines = new TextLine[partImgs.Length];
            for (var i = 0; i < partImgs.Length; i++)
            {
                textLines[i] = GetTextLine(partImgs[i]);
            }

            return textLines;
        }

        public TextLine GetTextLine(Image<Rgba32> src)
        {
            var sw = Stopwatch.StartNew();
            float scale = CrnnDstHeight / (float)src.Height;
            int dstWidth = (int)(src.Width * scale);

            Tensor<float> inputTensors;

            // Resize using ImageSharp
            using (var srcResize = src.Clone(ctx => ctx.Resize(dstWidth, CrnnDstHeight, KnownResamplers.Bicubic)))
            {
#if DEBUG
                string debugPath = $"Recognizer_{Guid.NewGuid()}.png";
                srcResize.Save(debugPath);
#endif

                // Convert to Tensor<float> (mean subtraction and normalization)
                inputTensors = OcrUtils.SubtractMeanNormalize(srcResize, MeanValues, NormValues);
            }

            IReadOnlyCollection<NamedOnnxValue> inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensors)
            };

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _crnnNet.Run(inputs))
                {
                    var result = results.First();
                    var tl = ScoreToTextLine(result.AsTensor<float>());
                    tl.Time = sw.ElapsedMilliseconds;
                    return tl;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
            }

            return new TextLine() { Time = sw.ElapsedMilliseconds };
        }

        private TextLine ScoreToTextLine(Tensor<float> srcData)
        {
            var dimensions = srcData.Dimensions; // e.g., [1, 80, 6625]
            int frames = dimensions[1]; // Height (Time steps)
            int vocabSize = dimensions[2]; // Width (Characters in dict)

            var scores = new List<float>(frames);
            var chars = new List<string>(frames);

            // Get flat data to avoid multi-dimensional indexer overhead
            var data = srcData.ToArray();

            int lastIndex = 0;

            for (int i = 0; i < frames; i++)
            {
                int maxIndex = 0;
                float maxValue = -1000f;

                // Current frame offset
                int offset = i * vocabSize;

                for (int j = 0; j < vocabSize; j++)
                {
                    float v = data[offset + j];
                    if (v > maxValue)
                    {
                        maxValue = v;
                        maxIndex = j;
                    }
                }

                // CTC Logic: 
                // 1. Skip the "Blank" token (usually index 0)
                // 2. Skip if it's a repeat of the previous frame's character
                // 3. Ensure it's within the dictionary bounds
                if (maxIndex > 0 && maxIndex < _keys.Length && maxIndex != lastIndex)
                {
                    scores.Add(maxValue);
                    chars.Add(_keys[maxIndex]);
                }

                lastIndex = maxIndex;
            }

            return new TextLine
            {
                Chars = chars.ToArray(),
                CharScores = scores.ToArray()
            };
        }

        private TextLine ScoreToTextLine(ReadOnlySpan<float> srcData, int h, int w)
        {
            int lastIndex = 0;
            var scores = new List<float>();
            var chars = new List<string>();

            for (int i = 0; i < h; i++)
            {
                int maxIndex = 0;
                float maxValue = -1000F;
                for (int j = 0; j < w; j++)
                {
                    int idx = i * w + j;
                    if (srcData[idx] > maxValue)
                    {
                        maxIndex = j;
                        maxValue = srcData[idx];
                    }
                }

                if (maxIndex > 0 && maxIndex < _keys.Length && !(i > 0 && maxIndex == lastIndex))
                {
                    scores.Add(maxValue);
                    chars.Add(_keys[maxIndex]);
                }

                lastIndex = maxIndex;
            }

            return new TextLine
            {
                Chars = chars.ToArray(),
                CharScores = scores.ToArray()
            };
        }

        public void Dispose()
        {
            _crnnNet.Dispose();
        }
    }
}