// Apache-2.0 license
// Adapted from RapidAI / RapidOCR
// https://github.com/RapidAI/RapidOCR/blob/92aec2c1234597fa9c3c270efd2600c83feecd8d/dotnet/RapidOcrOnnxCs/OcrLib/AngleNet.cs

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using KnownResamplers = SixLabors.ImageSharp.Processing.KnownResamplers;

namespace RapidOcrNet
{
    public sealed class TextClassifier : IDisposable
    {
        private const int AngleDstWidth = 192;
        private const int AngleDstHeight = 48;
        private const int AngleCols = 2;

        private static readonly float[] MeanValues = [127.5F, 127.5F, 127.5F];
        private static readonly float[] NormValues = [1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F];

        private InferenceSession _angleNet;
        private string _inputName;

        public void InitModel(string path, int numThread)
        {
            System.Diagnostics.Debug.WriteLine("InitModel enter");
            System.Diagnostics.Debug.WriteLine($"InitModel path={path}, numThread={numThread}");

            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Classifier model file does not exist: '{path}'.");
            }

            var op = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                InterOpNumThreads = numThread,
                IntraOpNumThreads = numThread
            };

            System.Diagnostics.Debug.WriteLine("InitModel creating InferenceSession");
            _angleNet = new InferenceSession(path, op);

            _inputName = _angleNet.InputMetadata.Keys.First();
            System.Diagnostics.Debug.WriteLine($"InitModel inputName={_inputName}");

            System.Diagnostics.Debug.WriteLine("InitModel exit");
        }

        public Angle[] GetAngles(Image<Rgba32>[] partImgs, bool doAngle, bool mostAngle)
        {
            System.Diagnostics.Debug.WriteLine("GetAngles enter");
            System.Diagnostics.Debug.WriteLine($"partImgs.Length={partImgs.Length}, doAngle={doAngle}, mostAngle={mostAngle}");

            var angles = new Angle[partImgs.Length];

            if (doAngle)
            {
                for (int i = 0; i < partImgs.Length; i++)
                {
                    System.Diagnostics.Debug.WriteLine($"GetAngles processing index={i}");
                    angles[i] = GetAngle(partImgs[i]);
                    System.Diagnostics.Debug.WriteLine($"GetAngles index={i} result Index={angles[i].Index}, Score={angles[i].Score}, Time={angles[i].Time}");
                }

                if (mostAngle && angles.Length > 0)
                {
                    // Sum the scores for each index
                    double scoreZero = angles.Where(x => x.Index == 0).Sum(x => x.Score);
                    double scoreOne = angles.Where(x => x.Index == 1).Sum(x => x.Score);

                    int mostAngleIndex = scoreZero >= scoreOne ? 0 : 1;

                    foreach (var angle in angles)
                    {
                        angle.Index = mostAngleIndex;
                    }
                }
            }
            else
            {
                System.Diagnostics.Debug.WriteLine("GetAngles doAngle disabled, initializing default angles");

                for (int i = 0; i < partImgs.Length; i++)
                {
                    angles[i] = new Angle
                    {
                        Index = -1,
                        Score = 0F
                    };

                    System.Diagnostics.Debug.WriteLine($"GetAngles index={i} default Index=-1 Score=0");
                }
            }

            System.Diagnostics.Debug.WriteLine("GetAngles exit");
            return angles;
        }


        public Angle GetAngle(Image<Rgba32> src)
        {
            System.Diagnostics.Debug.WriteLine("GetAngle enter");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            Tensor<float> inputTensors;

            System.Diagnostics.Debug.WriteLine($"Cloning and resizing image to {AngleDstWidth}x{AngleDstHeight}");

            using (var angleImg = src.Clone(ctx =>
                ctx.Resize(
                    AngleDstWidth,
                    AngleDstHeight,
                    KnownResamplers.Bicubic)))
            {
#if DEBUG
                var debugPath = $"Classifier_{Guid.NewGuid()}.png";
                System.Diagnostics.Debug.WriteLine($"Saving debug image: {debugPath}");
                angleImg.Save(debugPath);
#endif

                System.Diagnostics.Debug.WriteLine("Running SubtractMeanNormalize");
                inputTensors = OcrUtils.SubtractMeanNormalize(
                    angleImg,
                    MeanValues,
                    NormValues);
            }

            IReadOnlyCollection<NamedOnnxValue> inputs = new NamedOnnxValue[]
            {
        NamedOnnxValue.CreateFromTensor(_inputName, inputTensors)
            };

            System.Diagnostics.Debug.WriteLine("Running angle ONNX session");

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _angleNet.Run(inputs))
                {
                    var outputTensor = results[0];
                    System.Diagnostics.Debug.WriteLine("Received output tensor");

                    ReadOnlySpan<float> outputData;
                    if (outputTensor.AsTensor<float>() is DenseTensor<float> dt)
                    {
                        outputData = dt.Buffer.Span;
                        System.Diagnostics.Debug.WriteLine($"Output tensor DenseTensor span length={outputData.Length}");
                    }
                    else
                    {
                        var arr = outputTensor.AsEnumerable<float>().ToArray();
                        outputData = arr;
                        System.Diagnostics.Debug.WriteLine($"Output tensor enumerated length={outputData.Length}");
                    }

                    var angle = ScoreToAngle(outputData, AngleCols);
                    angle.Time = sw.ElapsedMilliseconds;

                    System.Diagnostics.Debug.WriteLine($"GetAngle exit success: Index={angle.Index}, Score={angle.Score}, Time={angle.Time}");
                    return angle;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"GetAngle exception: {ex.Message}{ex.StackTrace}");
            }

            var fallback = new Angle() { Time = sw.ElapsedMilliseconds };
            System.Diagnostics.Debug.WriteLine($"GetAngle exit fallback: Time={fallback.Time}");
            return fallback;
        }



        private static Angle ScoreToAngle(ReadOnlySpan<float> srcData, int angleColumns)
        {
            System.Diagnostics.Debug.WriteLine($"ScoreToAngle enter: angleColumns={angleColumns}, srcDataLength={srcData.Length}");

            int angleIndex = 0;
            float maxValue = srcData[0];

            System.Diagnostics.Debug.WriteLine($"Initial maxValue={maxValue}, angleIndex={angleIndex}");

            for (int i = 1; i < angleColumns; ++i)
            {
                float current = srcData[i];
                System.Diagnostics.Debug.WriteLine($"Iter i={i}, current={current}, maxValue={maxValue}, angleIndex={angleIndex}");

                if (current > maxValue)
                {
                    angleIndex = i;
                    maxValue = current;
                    System.Diagnostics.Debug.WriteLine($"New max found: maxValue={maxValue}, angleIndex={angleIndex}");
                }
            }

            var result = new Angle
            {
                Index = angleIndex,
                Score = maxValue
            };

            System.Diagnostics.Debug.WriteLine($"ScoreToAngle exit: Index={result.Index}, Score={result.Score}");

            return result;
        }


        public void Dispose()
        {
            _angleNet.Dispose();
        }
    }
}
