// Apache-2.0 license
// Adapted from RapidAI / RapidOCR
// https://github.com/RapidAI/RapidOCR/blob/92aec2c1234597fa9c3c270efd2600c83feecd8d/dotnet/RapidOcrOnnxCs/OcrLib/OcrUtils.cs

using System.IO.Compression;
using System.Numerics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RapidOcrNet;

internal static class OcrUtils
{
    public static Tensor<float> SubtractMeanNormalize(Image<Rgba32> src, float[] meanVals, float[] normVals)
    {
        var cols = src.Width;
        var rows = src.Height;
        const int channels = 3;

        var inputTensor = new DenseTensor<float>(new[] { 1, channels, rows, cols });

        // We get the underlying array or memory from the Tensor.
        // Since DenseTensor uses a flat array internally, we can manipulate it directly.
        var dataSpan = inputTensor.Buffer.Span;
        var channelSize = rows * cols;

        for (var r = 0; r < rows; r++)
        {
            // DangerousGetPixelRowMemory returns Memory<Rgba32>
            // Memory<T> is safe to use in various contexts where Span<T> is not.
            var rowMemory = src.DangerousGetPixelRowMemory(r);
            var rowSpan = rowMemory.Span;
            var rowOffset = r * cols;

            for (var c = 0; c < cols; c++)
            {
                var pixel = rowSpan[c];
                var pixelIdx = rowOffset + c;

                // RGB
                dataSpan[0 * channelSize + pixelIdx] = (pixel.R - meanVals[0]) * normVals[0];
                dataSpan[1 * channelSize + pixelIdx] = (pixel.G - meanVals[1]) * normVals[1];
                dataSpan[2 * channelSize + pixelIdx] = (pixel.B - meanVals[2]) * normVals[2];
            }
        }

        return inputTensor;
    }
    public static async Task DownloadAndProcessFilesAsync(IEnumerable<Uri> uris, string destinationFolder, HttpClient? client = null, CancellationToken ct = default)
    {
        var httpClient = client ?? new HttpClient();
        var logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger("FileProcessor");

        Directory.CreateDirectory(destinationFolder);

        foreach (var uri in uris)
        {
            try
            {
                using var response = await httpClient.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();

                var fileName = Path.GetFileName(uri.LocalPath);
                if (string.IsNullOrWhiteSpace(fileName)) fileName = Guid.NewGuid().ToString();

                var destinationPath = Path.Combine(destinationFolder, fileName);
                await using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);

                if (response.Content.Headers.ContentEncoding.Contains("br") || fileName.EndsWith(".br", StringComparison.OrdinalIgnoreCase))
                {
                    if (fileName.EndsWith(".br")) destinationPath = destinationPath[..^3];

                    await using var decompressionStream = new BrotliStream(responseStream, CompressionMode.Decompress);
                    await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
                    await decompressionStream.CopyToAsync(fileStream, ct).ConfigureAwait(false);
                }
                else
                {
                    await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
                    await responseStream.CopyToAsync(fileStream, ct).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                logger.LogWarning("Download cancelled for {Uri}", uri);
                throw;
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to process {Uri}", uri);
            }
        }
    }
    public static Image<Rgba32> MakePadding(Image<Rgba32> src, int padding)
    {
        if (padding <= 0)
        {
            System.Diagnostics.Debug.WriteLine($"[Padding] Padding is {padding}, returning original image.");
            return src;
        }

        var width = src.Width + 2 * padding;
        var height = src.Height + 2 * padding;
        System.Diagnostics.Debug.WriteLine(
            $"[Padding] Resizing image: {src.Width}x{src.Height} -> {width}x{height} (Padding: {padding})");

        var newImg = new Image<Rgba32>(width, height);

        newImg.Mutate(ctx =>
        {
            ctx.Fill(Color.White);
            ctx.DrawImage(src, new Point(padding, padding), 1f);
        });

#if DEBUG
        var fileName = $"Padding_{Guid.NewGuid()}.png";
        newImg.Save(fileName);
        System.Diagnostics.Debug.WriteLine($"[Padding] Debug image saved to: {Path.GetFullPath(fileName)}");
#endif

        return newImg;
    }


    public static int GetThickness(Image boxImg)
    {
        var minSize = boxImg.Width > boxImg.Height ? boxImg.Height : boxImg.Width;
        return minSize / 1000 + 2;
    }

    public static IEnumerable<Image<Rgba32>> GetPartImages(Image<Rgba32> src, IReadOnlyList<TextBox>? textBoxes)
    {
        if (textBoxes is null || textBoxes.Count == 0) yield break;

        foreach (var t in textBoxes)
            yield return GetRotateCropImage(src, t.Points);
    }

    public static Matrix4x4 GetPerspectiveTransform(
        PointF p0, PointF p1, PointF p2, PointF p3, // Source corners
        float w, float h) // Destination dimensions
    {
        // Solving for the perspective transform matrix that maps:
        // (0,0) -> p0
        // (w,0) -> p1
        // (w,h) -> p2
        // (0,h) -> p3
        // This "backward" mapping is what ImageSharp needs to sample pixels.

        float x0 = p0.X, y0 = p0.Y;
        float x1 = p1.X, y1 = p1.Y;
        float x2 = p2.X, y2 = p2.Y;
        float x3 = p3.X, y3 = p3.Y;

        var dx1 = x1 - x2;
        var dx2 = x3 - x2;
        var dx3 = x0 - x1 + x2 - x3;
        var dy1 = y1 - y2;
        var dy2 = y3 - y2;
        var dy3 = y0 - y1 + y2 - y3;

        var det = dx1 * dy2 - dx2 * dy1;

        // Safety check: if det is 0, the points are collinear (not a box)
        if (Math.Abs(det) < 1e-10) return Matrix4x4.Identity;

        var a13 = (dx3 * dy2 - dx2 * dy3) / det;
        var a23 = (dx1 * dy3 - dx3 * dy1) / det;

        // This creates the 3x3 Homography matrix inside a 4x4 structure
        // We map a unit square to the quad, then scale it to our target width/height
        var result = new Matrix4x4(
            x1 - x0 + a13 * x1, y1 - y0 + a13 * y1, 0, a13,
            x3 - x0 + a23 * x3, y3 - y0 + a23 * y3, 0, a23,
            0, 0, 1, 0,
            x0, y0, 0, 1
        );

        // Final step: Adjust for the fact that our destination isn't a 1x1 square,
        // but a width x height rectangle.
        return Matrix4x4.CreateScale(1.0f / w, 1.0f / h, 1.0f) * result;
    }

    public static Image<Rgba32> BitmapRotateClockWise180(Image<Rgba32> src)
    {
        var rotated = src.Clone(ctx => { ctx.Rotate(RotateMode.Rotate180); });

        return rotated;
    }

    public static Image<Rgba32> GetRotateCropImage(Image<Rgba32> src, PointF[] box)
    {
        if (box.Length != 4) throw new ArgumentException("Box must have exactly 4 points.", nameof(box));

        // Calculate dimensions of the target "flat" rectangle
        var width = (int)MathF.Max(
            float.Hypot(box[0].X - box[1].X, box[0].Y - box[1].Y),
            float.Hypot(box[2].X - box[3].X, box[2].Y - box[3].Y)
        );
        var height = (int)MathF.Max(
            float.Hypot(box[0].X - box[3].X, box[0].Y - box[3].Y),
            float.Hypot(box[1].X - box[2].X, box[1].Y - box[2].Y)
        );

        // If the box is already a perfect axis-aligned rectangle, just crop it
        if (box[0].Y == box[1].Y && box[1].X == box[2].X)
        {
            var rect = new Rectangle((int)box[0].X, (int)box[0].Y, width, height);
            var simpleCrop = src.Clone(ctx => ctx.Crop(rect));
            return FinalOrientationCheck(simpleCrop);
        }

        // Advanced: Perspective Warp
        // ImageSharp's Transform works best when applied to a crop that encompasses the polygon
        float minX = box[0].X, maxX = box[0].X, minY = box[0].Y, maxY = box[0].Y;
        for (var i = 1; i < 4; i++)
        {
            if (box[i].X < minX) minX = box[i].X;
            if (box[i].X > maxX) maxX = box[i].X;
            if (box[i].Y < minY) minY = box[i].Y;
            if (box[i].Y > maxY) maxY = box[i].Y;
        }

        var left = (int)MathF.Floor(minX);
        var top = (int)MathF.Floor(minY);
        var right = (int)MathF.Ceiling(maxX);
        var bottom = (int)MathF.Ceiling(maxY);

        var bounds = new Rectangle(left, top, right - left, bottom - top);
        // Ensure bounds are within source image
        bounds.Intersect(new Rectangle(0, 0, src.Width, src.Height));

        if (bounds.Width <= 0 || bounds.Height <= 0) return new Image<Rgba32>(1, 1);

        var imgCrop = src.Clone(ctx => ctx.Crop(bounds));

        // Adjust box points to be local to the imgCrop
        var matrix = GetPerspectiveTransform(
            new PointF(box[0].X - bounds.X, box[0].Y - bounds.Y),
            new PointF(box[1].X - bounds.X, box[1].Y - bounds.Y),
            new PointF(box[2].X - bounds.X, box[2].Y - bounds.Y),
            new PointF(box[3].X - bounds.X, box[3].Y - bounds.Y),
            width, height);

        // ImageSharp doesn't have a native "Cv2.GetPerspectiveTransform" 
        // You must ensure your Matrix from 'GetPerspectiveTransform' is a 3x3 
        // that maps SRC -> DEST.
        imgCrop.Mutate(ctx =>
        {
            // Set the background so we don't get black fringes
            ctx.BackgroundColor(Color.White);

            var builder = new ProjectiveTransformBuilder();
            builder.AppendMatrix(matrix);

            // This applies the transformation to the source image during the draw
            ctx.Transform(builder);
        });

        return FinalOrientationCheck(imgCrop);
    }

    private static Image<Rgba32> FinalOrientationCheck(Image<Rgba32> img)
    {
        if (img.Height >= img.Width * 1.5) img.Mutate(x => x.Rotate(RotateMode.Rotate90));
        return img;
    }
}