# RapidOcrNet, ImageSharp flavor
Cross-platform OCR processing library using PaddleOCR ONNX models, and based on original code from RapidAI's [RapidOCR](https://github.com/RapidAI/RapidOCR).

Available as NuGet package here https://www.nuget.org/packages/BrycensRanch.RapidOcrNet/

The code was optimised to remove dependencies on `System.Drawing`, `OpenCV` and `SkiaSharp`. The image processing is now done only using `ImageSharp` and `PContourNet`.

The project now uses PP-OCR v5 models, but v4 and v3 models are also supported (see [here](https://github.com/BobLd/RapidOcrNet/issues/3)).

All ONNX models and files and can be downloaded from: https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
You will need 4 different files for the code to work. Example below for PP-OCR v5 with latin language:
- Detection: `ch_PP-OCRv5_mobile_det.onnx`
- Classification: `ch_ppocr_mobile_v2.0_cls_infer.onnx`
- Recognition: `latin_PP-OCRv5_rec_mobile_infer.onnx`
- Model dictionary: `ppocrv5_latin_dict.txt`

## Usage
```csharp
string targetImg = "image.png";
// Where ONNX models downloaded from GitHub live
// The ONNX models are no longer copied to your publish directory.
string modelDir = Path.Combine(AppContext.BaseDirectory, "models");

using (var ocrEngine = new RapidOcr())
{
	await ocrEngine.LoadModelAsync(OnnxModels.V5.EnglishMobile, modelDir, new HttpClient());
    
	using (Image originSrc = Image.Load<Rgba32>(targetImg))
	{
		OcrResult ocrResult = ocrEngine.Detect(originSrc, RapidOcrOptions.Default);
		Console.WriteLine(ocrResult.ToString());
		Console.WriteLine(ocrResult.StrRes);
		Console.WriteLine();

		// Draw bounding boxes
        foreach (var block in ocrResult.TextBlocks)
        {
            var points = block.BoxPoints;
    
            originSrc.Mutate(ctx =>
            {
                ctx.DrawPolygon(Color.Red, 2f, points);
            });
        }

		using (var fs = new FileStream(Path.ChangeExtension(targetImg, "_ocr.png"), FileMode.Create))
		{
			originSrc.SaveAsPng(fs);
		}
	}
}
```
## Notice
Based on source code originally developed in the RapidOCR project (Apache-2.0 license).
- https://github.com/RapidAI/RapidOCR

Uses parts of source code originally developed in the PdfPig project (Apache-2.0 license).
- https://github.com/UglyToad/PdfPig

The dependency on OpenCV was removed thanks to the PContour library and its C# port.
- https://github.com/LingDong-/PContour
- https://github.com/BobLd/PContourNet

The models made available are from the PaddleOCR project (Apache-2.0 license) and were downloaded from https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
