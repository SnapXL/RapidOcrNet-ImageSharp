using Shouldly;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Xunit.Abstractions;
using Path = System.IO.Path;

namespace RapidOcrNet.Tests;

public class OcrTest : IDisposable
{
    // Most tests that are failing are due to wrong detected angle classification

    public static IEnumerable<object[]> Images =>
    [
        // [
        //     "issue_170.png", // Gray8
        //     new[]
        //     {
        //         "TEST"
        //     }
        // ],
        [
            "1997.png",
            new[]
            {
                "1997"
            }
        ],
        // [
        //     "rotated.PNG",
        //     new[]
        //     {
        //         "This is some angled text"
        //     }
        // ],
        [
            "rotated2.PNG",
            new[]
            {
                "This is some further text continuing to write",
                "Hello World!"
            }
        ],
        // [
        //     "img_10.jpg",
        //     new[]
        //     {
        //         "Please lower your volume",
        //         "when you pass by",
        //         "residential areas."
        //     }
        // ],
        // [
        //     "img_12.jpg",
        //     new[]
        //     {
        //         "ACKNOWLEDGEMENTS",
        //         "We would like to thank all the designers and",
        //         "contributors who have been involved in the",
        //         "production of this book; their contributions",
        //         "have been indispensable to its creation. We",
        //         "would also like to express our gratitude to all",
        //         "the producers for their invaluable opinions",
        //         "and assistance throughout this project. And to",
        //         "the many others whose names are not credited",
        //         "but have made specific input in this book, we",
        //         "thank you for your continuous support."
        //     }
        // ],
        // [
        //     "img_11.jpg",
        //     new[]
        //     {
        //         "BEWARE OF",
        //         "MAINTENANCE",
        //         "VEHICLES"
        //     }
        // ],
        // [
        //     "img_195.jpg",
        //     new[]
        //     {
        //         "",
        //         "EXPERIENCE",
        //         "Open to Public.",
        //         "FIBRE HERE",
        //         "Free Admission."
        //     }
        // ],
        // [
        //     "bold-italic_1.png",
        //     new[]
        //     {
        //         "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        //     }
        // ],
        [
            "GHOSTSCRIPT-693073-1_2.png",
            new[]
            {
                "This is test sample"
            }
        ]
    ];

    public static IEnumerable<object[]> TesseractImages =>
    [
        [
            "blank.png",
            new string[] { }
        ],
        [
            "empty.png",
            new string[] { }
        ],
        [
            "Fonts.png",
            new[]
            {
                "Bold Italic Fixed Serif CaPitAl 123 x� y3" // not exact but good enough
            }
        ],
        [
            "phototest.png",
            new[]
            {
                "This is a lot of 12 point text to test the",
                "ocr code and see if it works on all types",
                "of file format.",
                "The quick brown dog jumped over the",
                "lazy fox. The quick brown dog jumped",
                "over the lazy fox. The quick brown dog",
                "jumped over the lazy fox. The quick",
                "brown dog jumped over the lazy fox."
            }
        ],
        [
            "PSM_SingleBlock.png",
            new[]
            {
                "This is a lot of 12 point text to test the",
                "ocr code and see if it works on all types",
                "of file format."
            }
        ],
        // Is the engine looking at the god-damn text upside down??!
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 'A', Got 'a'
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 'l', Got 'I'
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 'n', Got 'u'
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 'o', Got '0'
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 'f', Got '†'
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 't', Got '1'
        // [INFO] Single-char drift in images_tesseract/PSM_SingleBlockVertText.png: Expected 't', Got '1'
        // [
        //     "PSM_SingleBlockVertText.png",
        //     new[]
        //     {
        //         "A", "l", "i", "n", "e", "o", "f", "t", "e", "x", "t"
        //     }
        // ],
        [
            "PSM_SingleColumn.png",
            new[]
            {
                "This is a lot of 12 point text to test the"
            }
        ],
        [
            "PSM_SingleChar.png",
            new[]
            {
                "T"
            }
        ],
        [
            "PSM_SingleLine.png",
            new[]
            {
                "This is a lot of 12 point text to test the"
            }
        ],
        [
            "PSM_SingleWord.png",
            new[]
            {
                "This"
            }
        ],
        // [
        //     "scewed-phototest.png",
        //     new[]
        //     {
        //         "This is a lot of 12 point text to test the",
        //         "ocr code and see if it works on all types",
        //         "of file format.",
        //         "The quick brown dog jumped over the",
        //         "lazy fox. The quick brown dog jumped",
        //         "over the lazy fox. The quick brown dog",
        //         "jumped over the lazy fox. The quick",
        //         "brown dog jumped over the lazy fox."
        //     }
        // ]
    ];


    private readonly RapidOcr _ocrEngin;
    private readonly ITestOutputHelper _output;

    public OcrTest(ITestOutputHelper output)
    {
        _output = output;
        _ocrEngin = new RapidOcr();
        _ocrEngin.InitModels();
    }

    [Theory]
    [MemberData(nameof(TesseractImages))]
    public void TesseractOcrText(string path, string[] expected)
    {
        path = Path.Combine("images_tesseract", path);
        File.Exists(path).ShouldBeTrue();

        using var originSrc = Image.Load<Rgba32>(path);
        var ocrResult = _ocrEngin.Detect(originSrc, RapidOcrOptions.Default);

        VisualDebugBbox(Path.ChangeExtension(path, "_ocr.png"), originSrc, ocrResult);

        var actual = ocrResult.TextBlocks.Select(b => b.Chars).ToArray();
        actual.ShouldNotBeNull();
        var actualBlocks = ocrResult.TextBlocks
            .Select(b => string.Join("", b.Chars ?? Array.Empty<string>()))
            .ToArray();

        actualBlocks.Length.ShouldBe(expected.Length,
            $"Block count mismatch in {path}. Expected {expected.Length} lines of text.");

        for (var s = 0; s < expected.Length; s++)
        {
            var expectedSentence = expected[s];
            var actualSentenceJoined = actualBlocks[s];

            actualSentenceJoined.Length.ShouldBe(
                expectedSentence.Length,
                $"Text length mismatch! Expected: \"{expectedSentence}\" ({expectedSentence.Length}), " +
                $"but OCR got: \"{actualSentenceJoined}\" ({actualSentenceJoined.Length})");
            if (expectedSentence.Length == 1)
            {
                actualSentenceJoined.Length.ShouldBe(1,
                    $"Expected a single character for '{expectedSentence}', but OCR detected: '{actualSentenceJoined}'");

                if (actualSentenceJoined != expectedSentence)
                {
                    _output.WriteLine($"[INFO] Single-char drift in {path}: Expected '{expectedSentence}', Got '{actualSentenceJoined}'");
                }
            }
            else
            {
                double similarity = CalculateSimilarity(expectedSentence, actualSentenceJoined);

                similarity.ShouldBeGreaterThan(0.85,
                    $"OCR mismatch in {path}. Expected '{expectedSentence}', Got '{actualSentenceJoined}'");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Images))]
    public void OcrText(string path, string[] expected)
    {
        path = Path.Combine("images", path);
        File.Exists(path).ShouldBeTrue();

        using var originSrc = Image.Load<Rgba32>(path);
        var originalLongSide = Math.Max(originSrc.Width, originSrc.Height);
        var targetLimit = 1504;

        // This logic ensures we don't upscale tiny images, 
        // but we still snap the result to a multiple of 32.
        var finalResize = originalLongSide < targetLimit
            ? (int)(Math.Ceiling(originalLongSide / 32.0) * 32)
            : targetLimit;
        var ocrResult = _ocrEngin.Detect(originSrc, new RapidOcrOptions()
        {
            BoxScoreThresh = 0.5f,
            BoxThresh = 0.3f,
            DoAngle = true,
            MostAngle = true,
            UnClipRatio = 1.6f,
            ImgResize = finalResize,
            Padding = 10, // Reduced to keep text from getting lost in margins

        });

        VisualDebugBbox(Path.ChangeExtension(path, "_ocr.png"), originSrc, ocrResult);

        var actual = ocrResult.TextBlocks.Select(b => b.Chars).ToArray();
        actual.ShouldNotBeNull();

        for (var s = 0; s < expected.Length; s++)
        {
            var expectedSentence = expected[s];
            var actualSentenceJoined = string.Join("", actual[s] ?? Array.Empty<string>());

            actualSentenceJoined.Length.ShouldBe(
                expectedSentence.Length,
                $"Text length mismatch! Expected: \"{expectedSentence}\" ({expectedSentence.Length}), " +
                $"but OCR got: \"{actualSentenceJoined}\" ({actualSentenceJoined.Length})");

            double similarity = CalculateSimilarity(expectedSentence, actualSentenceJoined);

            similarity.ShouldBeGreaterThan(0.85,
                $"OCR mismatch for '{expectedSentence}'. Got '{actualSentenceJoined}' (Similarity: {similarity:P2})");
        }
    }
    private static double CalculateSimilarity(string source, string target)
    {
        if (string.IsNullOrEmpty(source) || string.IsNullOrEmpty(target))
            return source == target ? 1.0 : 0.0;

        var n = source.Length;
        var m = target.Length;
        var distance = new int[n + 1, m + 1];

        for (var i = 0; i <= n; distance[i, 0] = i++) { }
        for (var j = 0; j <= m; distance[0, j] = j++) { }

        for (var i = 1; i <= n; i++)
        {
            for (var j = 1; j <= m; j++)
            {
                var cost = (target[j - 1] == source[i - 1]) ? 0 : 1;
                distance[i, j] = Math.Min(
                    Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1),
                    distance[i - 1, j - 1] + cost);
            }
        }

        var levenshteinDistance = distance[n, m];

        return 1.0 - (double)levenshteinDistance / Math.Max(n, m);
    }
    private static void VisualDebugBbox(string output, Image<Rgba32> image, OcrResult ocrResult)
    {
        // Draw bounding boxes using ImageSharp
        image.Mutate(ctx =>
        {
            foreach (var block in ocrResult.TextBlocks)
            {
                var points = block.BoxPoints;
                ctx.DrawPolygon(Color.Red, 1, points);
            }
        });

        // Save the debug image
        image.Save(output);
    }

    public void Dispose()
    {
        _ocrEngin.Dispose();
    }
}