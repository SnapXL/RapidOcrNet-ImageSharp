// Apache-2.0 license
// Adapted from RapidAI / RapidOCR
// https://github.com/RapidAI/RapidOCR/blob/92aec2c1234597fa9c3c270efd2600c83feecd8d/dotnet/RapidOcrOnnxCs/OcrLib/OcrLite.cs

using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace RapidOcrNet
{
    public sealed class RapidOcr : IDisposable
    {
        private readonly TextDetector _textDetector = new TextDetector();
        private readonly TextClassifier _textClassifier = new TextClassifier();
        private readonly TextRecognizer _textRecognizer = new TextRecognizer();

        /// <summary>
        /// Initialize using default models (latin).
        /// </summary>
        public void InitModels(int numThread = 0)
        {
            const string modelsFolderName = "models";
            const string modelsVersion = "v5";

            string detPath = Path.Combine(modelsFolderName, modelsVersion, "ch_PP-OCRv5_mobile_det.onnx");
            string clsPath = Path.Combine(modelsFolderName, modelsVersion, "ch_ppocr_mobile_v2.0_cls_infer.onnx");
            string recPath = Path.Combine(modelsFolderName, modelsVersion, "latin_PP-OCRv5_rec_mobile_infer.onnx");
            string keysPath = Path.Combine(modelsFolderName, modelsVersion, "ppocrv5_latin_dict.txt");

            InitModels(detPath, clsPath, recPath, keysPath, numThread);
        }

        public void InitModels(string detPath, string clsPath, string recPath, string keysPath, int numThread)
        {
            _textDetector.InitModel(detPath, numThread);
            _textClassifier.InitModel(clsPath, numThread);
            _textRecognizer.InitModel(recPath, keysPath, numThread);
        }
        public async Task LoadModelAsync(OcrModel model, string modelDir, HttpClient client, int threads = 0, CancellationToken ct = default)
        {
            var targetThreads = threads;
            Directory.CreateDirectory(modelDir);

            var detPath = GetLocalPath(modelDir, model.Detector.Url);
            var clsPath = GetLocalPath(modelDir, model.Classifier.Url);
            var recPath = GetLocalPath(modelDir, model.Recognizer.Url);
            var keysPath = model.Recognizer.KeyUrl is not null
                ? GetLocalPath(modelDir, model.Recognizer.KeyUrl)
                : string.Empty;

            // Collect missing files for batch download
            var missingFiles = new List<(Uri Uri, string TargetPath)>();
            ValidateFile(model.Detector.Url, detPath, missingFiles);
            ValidateFile(model.Classifier.Url, clsPath, missingFiles);
            ValidateFile(model.Recognizer.Url, recPath, missingFiles);

            if (model.Recognizer.KeyUrl is not null)
                ValidateFile(model.Recognizer.KeyUrl, keysPath, missingFiles);

            if (missingFiles.Count > 0)
            {
                foreach (var (uri, path) in missingFiles)
                {
                    var folder = Path.GetDirectoryName(path)!;
                    await OcrUtils.DownloadAndProcessFilesAsync([uri], folder, client, ct).ConfigureAwait(false);
                }
            }

            InitModels(detPath, clsPath, recPath, keysPath, targetThreads);
        }

        private static string GetLocalPath(string root, string url)
        {
            var fileName = Path.GetFileName(url);
            // If the URL suggests it was brotli compressed, we expect the decompressed file locally
            if (fileName.EndsWith(".br", StringComparison.OrdinalIgnoreCase))
            {
                fileName = fileName[..^3];
            }
            return Path.Combine(root, fileName);
        }

        private static void ValidateFile(string url, string localPath, List<(Uri, string)> missing)
        {
            if (!File.Exists(localPath))
            {
                missing.Add((new Uri(url), localPath));
                return;
            }

            var info = new FileInfo(localPath);

            if (info.Length >= 1) return;
            File.Delete(localPath);
            missing.Add((new Uri(url), localPath));
        }
        public OcrResult Detect(string path, RapidOcrOptions options)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Could not find image to process: '{path}'.", path);
            }

            var originSrc = Image.Load<Rgba32>(path);
            return Detect(originSrc, options);
        }

        public OcrResult Detect(Image originSrc, RapidOcrOptions options)
        {
            return Detect(originSrc.CloneAs<Rgba32>(), options);
        }

        public OcrResult Detect(Image<Rgba32> originSrc, RapidOcrOptions options)
        {
            int originMaxSide = Math.Max(originSrc.Width, originSrc.Height);

            int resize;
            if (options.ImgResize <= 0 || options.ImgResize > originMaxSide)
            {
                resize = originMaxSide;
            }
            else
            {
                resize = options.ImgResize;
            }

            resize += 2 * options.Padding;
            var paddingRect = new Rectangle(options.Padding, options.Padding, originSrc.Width, originSrc.Height);
            using (var paddingSrc = OcrUtils.MakePadding(originSrc, options.Padding))
            {
                return DetectOnce(
                    paddingSrc,
                    paddingRect,
                    ScaleParam.GetScaleParam(paddingSrc, resize),
                    options.BoxScoreThresh,
                    options.BoxThresh,
                    options.UnClipRatio,
                    options.DoAngle,
                    options.MostAngle
                );
            }
        }

        private OcrResult DetectOnce(
    Image<Rgba32> src,
    Rectangle originRect,
    ScaleParam scale,
    float boxScoreThresh,
    float boxThresh,
    float unClipRatio,
    bool doAngle,
    bool mostAngle)
        {
            System.Diagnostics.Debug.WriteLine($"[OCR] Starting detection: Image={src.Width}x{src.Height}, BoxScoreThresh={boxScoreThresh}, UnClip={unClipRatio}");

            // Start detect
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // step: dbNet getTextBoxes
            var textBoxes = _textDetector.GetTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio) ?? [];
            var dbNetTime = sw.ElapsedMilliseconds;
            System.Diagnostics.Debug.WriteLine($"[OCR] DBNet found {textBoxes.Count} text boxes in {dbNetTime}ms");

            // getPartImages
            var partImages = OcrUtils.GetPartImages(src, textBoxes).ToArray();

            // step: angleNet getAngles
            Angle[] angles = _textClassifier.GetAngles(partImages, doAngle, mostAngle);
            System.Diagnostics.Debug.WriteLine($"[OCR] AngleNet processed {angles.Length} parts");

            // Rotate partImgs
            int rotateCount = 0;
            for (int i = 0; i < partImages.Length; ++i)
            {
                if (angles[i].Index == 1)
                {
                    partImages[i] = OcrUtils.BitmapRotateClockWise180(partImages[i]);
                    rotateCount++;
                }
            }
            System.Diagnostics.Debug.WriteLine($"[OCR] Rotated {rotateCount} images");

            // step: crnnNet getTextLines
            TextLine[] textLines = _textRecognizer.GetTextLines(partImages);
            System.Diagnostics.Debug.WriteLine($"[OCR] CRNN finished recognizing {textLines.Length} lines");

            foreach (var img in partImages)
            {
                img.Dispose();
            }

            var textBlocks = new TextBlock[textLines.Length];
            for (int i = 0; i < textLines.Length; ++i)
            {
                var textBox = textBoxes[i];
                var angle = angles[i];
                var textLine = textLines[i];

                for (int p = 0; p < textBox.Points.Length; ++p)
                {
                    ref PointF point = ref textBox.Points[p];
                    point.X -= originRect.Left;
                    point.Y -= originRect.Top;
                }

                textBlocks[i] = new TextBlock
                {
                    BoxPoints = textBox.Points,
                    BoxScore = textBox.Score,
                    AngleIndex = angle.Index,
                    AngleScore = angle.Score,
                    AngleTime = angle.Time,
                    Chars = textLine.Chars,
                    CharScores = textLine.CharScores,
                    CrnnTime = textLine.Time,
                    BlockTime = angle.Time + textLine.Time
                };
            }

            var fullDetectTime = sw.ElapsedMilliseconds;

            var strRes = new StringBuilder();
            foreach (var x in textBlocks)
            {
                strRes.AppendLine(x.GetText());
            }

            System.Diagnostics.Debug.WriteLine($"[OCR] Detection complete. Total Time: {fullDetectTime}ms, Results: {textBlocks.Length} blocks");

            return new OcrResult
            {
                TextBlocks = textBlocks,
                DbNetTime = dbNetTime,
                DetectTime = fullDetectTime,
                StrRes = strRes.ToString()
            };
        }


        public void Dispose()
        {
            _textClassifier.Dispose();
            _textRecognizer.Dispose();
            _textDetector.Dispose();
        }
    }
}
