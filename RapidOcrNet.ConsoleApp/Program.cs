using System.Diagnostics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RapidOcrNet.ConsoleApp
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                var sourceExtensions = new[] { ".jpg", ".jpeg", ".png", ".webp" };
                var exclusionPatterns = new[] { "_ocr", "Detector_", "Padding", "Recognizer", "Classifier" };

                args = Directory.EnumerateFiles(Directory.GetCurrentDirectory())
                    .Where(file =>
                    {
                        var name = Path.GetFileName(file);
                        var ext = Path.GetExtension(file).ToLower();

                        // Must be an image extension AND NOT match our "debug" patterns
                        return sourceExtensions.Contains(ext) &&
                               !exclusionPatterns.Any(p => name.Contains(p, StringComparison.OrdinalIgnoreCase));
                    })
                    .ToArray();

                if (args.Length > 0)
                {
                    Console.WriteLine($"No args provided. Auto-detected {args.Length} images to process.");
                }
                else
                {
                    Console.WriteLine("[CRITICAL] No images found in the current directory.");
                    return;
                }
            }
            Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));
            Debug.AutoFlush = true;
            using var ocrEngin = new RapidOcr();
            ocrEngin.InitModels();

            foreach (var path in args)
            {
                ProcessImage(ocrEngin, path);
            }
            var currentDir = Directory.GetCurrentDirectory();
            var debugDir = "debug";
            if (!Directory.Exists(debugDir)) Directory.CreateDirectory(debugDir);

            // The ocrEngine litters pngs in the current dir. Move them all to the out dir
            var patterns = new[] { "Detector_*.png", "Padding*.png", "Recognizer*.png", "Classifier*.png", "*_ocr.png" };

            var filesToMove = patterns.SelectMany(p => Directory.GetFiles(currentDir, p)).Distinct();

            foreach (var filePath in filesToMove)
            {
                try
                {
                    var fileName = Path.GetFileName(filePath);
                    var destination = Path.Combine(debugDir, fileName);

                    File.Move(filePath, destination, overwrite: true);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Could not move {filePath}: {ex.Message}");
                }
            }

            Console.WriteLine("Bye, RapidOcrNet!");
        }

        static void ProcessImage(RapidOcr ocrEngine, string targetImg)
        {
            Console.WriteLine($"Processing {targetImg}");

            using var originSrc = Image.Load<Rgba32>(targetImg);
            var ocrResult = ocrEngine.Detect(originSrc, RapidOcrOptions.Default);
            Console.WriteLine(ocrResult.ToString());
            Console.WriteLine(ocrResult.StrRes);
            Console.WriteLine();

            foreach (var block in ocrResult.TextBlocks)
            {
                var points = block.BoxPoints;
                var scoreColor = GetColorFromScore(block.BoxScore);

                var secondaryColor = Color.FromRgb(0, 255, 255).WithAlpha(0.5f);

                var start = points[0];
                var end = points[2];

                var brush = new LinearGradientBrush(
                    start,
                    end,
                    GradientRepetitionMode.Reflect,
                    new ColorStop(0f, scoreColor),
                    new ColorStop(1f, secondaryColor));

                var pen = Pens.DashDotDot(brush, 4F);

                originSrc.Mutate(ctx =>
                {
                    ctx.DrawPolygon(pen, points);
                });
            }
            if (!Directory.Exists("out")) Directory.CreateDirectory("out");
            var pureName = Path.GetFileNameWithoutExtension(targetImg);

            var outPath = Path.Combine(Directory.GetCurrentDirectory(), "out", $"{pureName}_ocr.png");

            originSrc.SaveAsPng(outPath);
        }
        private static Color GetColorFromScore(float score)
        {
            score = Math.Clamp(score, 0f, 1f);

            // Transition from Red (0.0) to Yellow (0.5) to Green (1.0)
            var r = (byte)(255 * Math.Clamp(2.0 * (1.0 - score), 0.0, 1.0));
            var g = (byte)(255 * Math.Clamp(2.0 * score, 0.0, 1.0));
            const byte b = 0;

            return Color.FromRgb(r, g, b);
        }
    }
}
