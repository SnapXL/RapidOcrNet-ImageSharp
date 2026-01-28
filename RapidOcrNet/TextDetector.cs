// Apache-2.0 license
// Adapted from RapidAI / RapidOCR
// https://github.com/RapidAI/RapidOCR/blob/92aec2c1234597fa9c3c270efd2600c83feecd8d/dotnet/RapidOcrOnnxCs/OcrLib/DbNet.cs

using System.Buffers;
using Clipper2Lib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using KnownResamplers = SixLabors.ImageSharp.Processing.KnownResamplers;

namespace RapidOcrNet
{
    public sealed class TextDetector : IDisposable
    {

        private static readonly float[] MeanValues = [0.485F * 255F, 0.456F * 255F, 0.406F * 255F];
        private static readonly float[] NormValues = [1.0F / 0.229F / 255.0F, 1.0F / 0.224F / 255.0F, 1.0F / 0.225F / 255.0F];

        private InferenceSession _dbNet;
        private string _inputName;



        public void InitModel(string path, int numThread)
        {
            System.Diagnostics.Debug.WriteLine("InitModel (Detector) enter");
            System.Diagnostics.Debug.WriteLine($"InitModel path={path}, numThread={numThread}");

            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Detector model file does not exist: '{path}'.");
            }

            var op = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                InterOpNumThreads = numThread,
                IntraOpNumThreads = numThread
            };

            System.Diagnostics.Debug.WriteLine("InitModel (Detector) creating InferenceSession");
            _dbNet = new InferenceSession(path, op);

            _inputName = _dbNet.InputMetadata.Keys.First();
            System.Diagnostics.Debug.WriteLine($"InitModel (Detector) inputName={_inputName}");

            System.Diagnostics.Debug.WriteLine("InitModel (Detector) exit");
        }


        public IReadOnlyList<TextBox>? GetTextBoxes(
    Image<Rgba32> src,
    ScaleParam scale,
    float boxScoreThresh,
    float boxThresh,
    float unClipRatio)
        {
            System.Diagnostics.Debug.WriteLine("GetTextBoxes enter");
            System.Diagnostics.Debug.WriteLine($"GetTextBoxes srcSize={src.Width}x{src.Height}");
            System.Diagnostics.Debug.WriteLine($"GetTextBoxes scale=({scale.DstWidth}x{scale.DstHeight}), boxScoreThresh={boxScoreThresh}, boxThresh={boxThresh}, unClipRatio={unClipRatio}");

            Tensor<float> inputTensors;

            using (var srcResize = src.Clone(ctx =>
                ctx.Resize(
                    scale.DstWidth,
                    scale.DstHeight,
                    KnownResamplers.Bicubic)))
            {
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes resized image to {scale.DstWidth}x{scale.DstHeight}");

#if DEBUG
                var debugPath = $"Detector_{Guid.NewGuid()}.png";
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes saving debug image {debugPath}");
                srcResize.Save(debugPath);
#endif

                inputTensors = OcrUtils.SubtractMeanNormalize(srcResize, MeanValues, NormValues);
                System.Diagnostics.Debug.WriteLine("GetTextBoxes input tensor created");
            }

            IReadOnlyCollection<NamedOnnxValue> inputs = new NamedOnnxValue[]
            {
        NamedOnnxValue.CreateFromTensor(_inputName, inputTensors)
            };

            System.Diagnostics.Debug.WriteLine($"GetTextBoxes running inference, inputName={_inputName}");

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _dbNet.Run(inputs))
                {
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes inference completed");
                    var output = results[0];
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes processing output tensor");

                    var boxes = GetTextBoxes(
                        output,
                        scale.DstHeight,
                        scale.DstWidth,
                        scale,
                        boxScoreThresh,
                        boxThresh,
                        unClipRatio);

                    System.Diagnostics.Debug.WriteLine($"GetTextBoxes produced {(boxes == null ? 0 : boxes.Count)} boxes");
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes exit");
                    return boxes;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("GetTextBoxes exception");
                System.Diagnostics.Debug.WriteLine(ex.Message);
                System.Diagnostics.Debug.WriteLine(ex.StackTrace);
            }

            System.Diagnostics.Debug.WriteLine("GetTextBoxes exit with null");
            return null;
        }


        private static PointF[][] FindContours(ReadOnlySpan<L8> array, int rows, int cols)
        {
            System.Diagnostics.Debug.WriteLine("FindContours enter");
            System.Diagnostics.Debug.WriteLine($"FindContours arrayLength={array.Length}, rows={rows}, cols={cols}");

            int[]? vPool = null;
            try
            {
                Span<int> v = array.Length <= 256
                    ? stackalloc int[array.Length]
                    : vPool = ArrayPool<int>.Shared.Rent(array.Length);

                System.Diagnostics.Debug.WriteLine($"FindContours using {(vPool is null ? "stackalloc" : "ArrayPool")} buffer");

                for (int i = 0; i < array.Length; i++)
                {
                    v[i] = array[i].PackedValue;
                }

                System.Diagnostics.Debug.WriteLine("FindContours input buffer populated");

                var contours = PContour.FindContours(v, cols, rows);
                System.Diagnostics.Debug.WriteLine($"FindContours rawContours={contours.Count}");

                var result = contours
                    .Where(c => !c.isHole)
                    .Select(c =>
                    {
                        var approx = PContour.ApproxPolyDP(c.GetSpan(), 1).ToArray();
                        System.Diagnostics.Debug.WriteLine($"FindContours contour approximated, points={approx.Length}");
                        return approx;
                    })
                    .ToArray();

                System.Diagnostics.Debug.WriteLine($"FindContours finalContours={result.Length}");
                System.Diagnostics.Debug.WriteLine("FindContours exit");
                return result;
            }
            finally
            {
                if (vPool is not null)
                {
                    ArrayPool<int>.Shared.Return(vPool);
                    System.Diagnostics.Debug.WriteLine("FindContours returned ArrayPool buffer");
                }
            }
        }



        private static bool TryFindIndex(Dictionary<int, int> link, int offset, out int index)
        {
            bool found = false;
            index = offset;
            while (link.TryGetValue(index, out int newIndex))
            {
                found = true;
                if (index == newIndex) break;
                index = newIndex;
            }
            return found;
        }
        private IReadOnlyList<TextBox> GetTextBoxes(
            DisposableNamedOnnxValue outputTensor,
            int rows,
            int cols,
            ScaleParam s,
            float boxScoreThresh,
            float boxThresh,
            float unClipRatio)
        {
            System.Diagnostics.Debug.WriteLine("GetTextBoxes enter");
            System.Diagnostics.Debug.WriteLine($"GetTextBoxes rows={rows}, cols={cols}, boxScoreThresh={boxScoreThresh}, boxThresh={boxThresh}, unClipRatio={unClipRatio}");

            const float maxSideThresh = 3.0f;
            var rsBoxes = new List<TextBox>();

            ReadOnlySpan<float> predData;
            if (outputTensor.AsTensor<float>() is DenseTensor<float> dt)
            {
                predData = dt.Buffer.Span;
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes tensor DenseTensor spanLength={predData.Length}");
            }
            else
            {
                var tmp = outputTensor.AsEnumerable<float>().ToArray();
                predData = tmp;
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes tensor enumerable spanLength={predData.Length}");
            }

            using var predImage = new Image<Rgba32>(cols, rows);
            using var thresholdMatBitmap = new Image<L8>(cols, rows);

            System.Diagnostics.Debug.WriteLine("GetTextBoxes filling prediction and threshold buffers");

            // FIX: Access internal image memory directly using row spans instead of the extension method copy
            for (int y = 0; y < rows; y++)
            {
                var pixelRow = predImage.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y);
                var thresholdRow = thresholdMatBitmap.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y);

                for (int x = 0; x < cols; x++)
                {
                    int i = (y * cols) + x;
                    float f = predData[i];
                    byte v = (byte)(f * 255f);

                    pixelRow[x] = new Rgba32(v, v, v, 255);
                    thresholdRow[x] = new L8(f > boxThresh ? (byte)1 : (byte)0);
                }
            }

            System.Diagnostics.Debug.WriteLine("GetTextBoxes applying dilation");

            thresholdMatBitmap.Mutate(ctx =>
            {
                // Simple internal operation to ensure the image state is updated
                ctx.DrawImage(thresholdMatBitmap, 1f);
            });

            System.Diagnostics.Debug.WriteLine("GetTextBoxes finding contours");

            // Since thresholdMat was originally a span from your extension, 
            // we need to pass the actual pixels from the bitmap to FindContours
            // Assuming FindContours takes a Span<L8>, we get it from the frame:
            var thresholdPixels = new L8[cols * rows];
            for (int y = 0; y < rows; y++)
            {
                thresholdMatBitmap.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y).CopyTo(thresholdPixels.AsSpan(y * cols, cols));
            }

            var contours = FindContours(thresholdPixels, rows, cols);
            System.Diagnostics.Debug.WriteLine($"GetTextBoxes contoursCount={contours.Length}");

            for (int i = 0; i < contours.Length; i++)
            {
                var contour = contours[i];
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes contourIndex={i}, points={contour.Length}");

                // FIX: Must have at least 4 points for GetSize to access index [3]
                if (contour.Length < 4)
                {
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes contour skipped: too few points (need 4 for quad)");
                    continue;
                }

                PointF[] minBox = GetMiniBox(contour, out float maxSide);
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes minBox maxSide={maxSide}");

                if (maxSide < maxSideThresh)
                {
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes contour skipped: maxSide below threshold");
                    continue;
                }

                double score = GetScore(contour, predImage);
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes contour score={score}");

                if (score < boxScoreThresh)
                {
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes contour skipped: score below threshold");
                    continue;
                }

                var clipBox = Unclip(minBox, unClipRatio);
                if (clipBox is null)
                {
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes contour skipped: unclip returned null");
                    continue;
                }

                PointF[] clipMinBox = GetMiniBox(clipBox, out maxSide);
                System.Diagnostics.Debug.WriteLine($"GetTextBoxes clipMinBox maxSide={maxSide}");

                if (maxSide < maxSideThresh + 2)
                {
                    System.Diagnostics.Debug.WriteLine("GetTextBoxes contour skipped: clip maxSide below threshold");
                    continue;
                }

                var finalPoints = new PointF[clipMinBox.Length];

                for (int j = 0; j < clipMinBox.Length; j++)
                {
                    var item = clipMinBox[j];

                    int x = (int)(item.X / s.ScaleWidth);
                    int y = (int)(item.Y / s.ScaleHeight);

                    int ptx = Math.Min(Math.Max(x, 0), s.SrcWidth);
                    int pty = Math.Min(Math.Max(y, 0), s.SrcHeight);

                    finalPoints[j] = new PointF(ptx, pty);

                    System.Diagnostics.Debug.WriteLine($"GetTextBoxes point[{j}] raw=({item.X},{item.Y}) scaled=({ptx},{pty})");
                }

                var textBox = new TextBox
                {
                    Score = (float)score,
                    Points = finalPoints
                };

                rsBoxes.Add(textBox);
                System.Diagnostics.Debug.WriteLine("GetTextBoxes TextBox added");
            }

            System.Diagnostics.Debug.WriteLine($"GetTextBoxes exit, totalBoxes={rsBoxes.Count}");
            return rsBoxes;
        }


        private static PointF[] GetMiniBox(PointF[] contours, out float minEdgeSize)
        {
            PointF[] points = GeometryExtensions.MinimumAreaRectangle(contours);

            GeometryExtensions.GetSize(points, out float width, out float height);
            minEdgeSize = MathF.Min(width, height);

            Array.Sort(points, CompareByX);

            int index1 = 0;
            int index2 = 1;
            int index3 = 2;
            int index4 = 3;

            if (points[1].Y > points[0].Y)
            {
                index1 = 0;
                index4 = 1;
            }
            else
            {
                index1 = 1;
                index4 = 0;
            }

            if (points[3].Y > points[2].Y)
            {
                index2 = 2;
                index3 = 3;
            }
            else
            {
                index2 = 3;
                index3 = 2;
            }

            return [points[index1], points[index2], points[index3], points[index4]];
        }

        public static int CompareByX(PointF left, PointF right)
        {
            if (left.X > right.X)
            {
                return 1;
            }

            if (left.X == right.X)
            {
                return 0;
            }

            return -1;
        }
        private static double GetScore(PointF[] contours, Image<Rgba32> fMapMat)
        {
            float xmin = contours.Min(p => p.X);
            float xmax = contours.Max(p => p.X);
            float ymin = contours.Min(p => p.Y);
            float ymax = contours.Max(p => p.Y);

            int iXmin = Math.Max(0, (int)Math.Floor(xmin));
            int iXmax = Math.Min(fMapMat.Width - 1, (int)Math.Ceiling(xmax));
            int iYmin = Math.Max(0, (int)Math.Floor(ymin));
            int iYmax = Math.Min(fMapMat.Height - 1, (int)Math.Ceiling(ymax));

            int width = iXmax - iXmin + 1;
            int height = iYmax - iYmin + 1;

            System.Diagnostics.Debug.WriteLine($"[ScoreCalc] ROI: x={iXmin}, y={iYmin}, w={width}, h={height} (Original: xmin={xmin:F2}, ymin={ymin:F2})");

            if (width <= 0 || height <= 0)
            {
                System.Diagnostics.Debug.WriteLine("[ScoreCalc] ROI has no area, returning 0");
                return 0;
            }

            try
            {
                var roiRect = new Rectangle(iXmin, iYmin, width, height);
                using var imgCrop = fMapMat.Clone(ctx => ctx.Crop(roiRect));

                var localPoints = contours.Select(p => new PointF(p.X - iXmin, p.Y - iYmin)).ToArray();

                double sum = 0;
                int count = 0;

                using var mask = new Image<L8>(width, height);
                mask.Mutate(ctx =>
                {
                    ctx.Fill(Color.Black);
                    var polygon = new Polygon(new LinearLineSegment(localPoints));
                    ctx.Fill(Color.White, polygon);
                });

                // FIX: Instead of using the faulty GetPixelSpan copy, access the rows directly
                for (int y = 0; y < height; y++)
                {
                    var maskRow = mask.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y);
                    var cropRow = imgCrop.Frames.RootFrame.PixelBuffer.DangerousGetRowSpan(y);

                    // Sample data for the first valid row to verify fixing the "all zero" bug
                    if (y == 0 && cropRow.Length > 0)
                    {
                        var p = cropRow[0];
                        System.Diagnostics.Debug.WriteLine($"[ScoreCalc] Sample Pixel Data at ROI start: R={p.R}, G={p.G}, B={p.B}, A={p.A}");
                    }

                    for (int x = 0; x < width; x++)
                    {
                        // L8 PackedValue 0 is Black, > 0 is White (inside polygon)
                        if (maskRow[x].PackedValue > 0)
                        {
                            sum += cropRow[x].R;
                            count++;
                        }
                    }
                }

                if (count == 0)
                {
                    System.Diagnostics.Debug.WriteLine("[ScoreCalc] Mask produced 0 pixels inside polygon");
                    return 0;
                }

                double finalScore = sum / count / 255.0;
                System.Diagnostics.Debug.WriteLine($"[ScoreCalc] Stats: Counted={count}, RawSum={sum}, AvgByte={sum / count:F2}, FinalScore={finalScore:F4}");

                return finalScore;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ScoreCalc ERROR] {ex.Message}\n{ex.StackTrace}");
                return 0;
            }
        }

        private static PointF[]? Unclip(PointF[] box, float unclipRatio)
        {
            PointF[] points = GeometryExtensions.MinimumAreaRectangle(box).Select(p => new PointF(p.X, p.Y)).ToArray();
            GeometryExtensions.GetSize(points, out float width, out float height);

            if (height < 1.001 && width < 1.001)
            {
                return null;
            }

            var theClipperPts = new Path64(box.Select(pt => new Point64(pt.X, pt.Y)));

            float area = MathF.Abs(SignedPolygonArea(box));
            double length = LengthOfPoints(box);
            double distance = area * unclipRatio / length;

            var co = new ClipperOffset();
            co.AddPath(theClipperPts, JoinType.Round, EndType.Polygon);
            var solution = new Paths64();
            co.Execute(distance, solution);
            if (solution.Count == 0)
            {
                return null;
            }

            var unclipped = solution[0];

            var retPts = new PointF[unclipped.Count];
            for (int i = 0; i < unclipped.Count; ++i)
            {
                var ip = unclipped[i];
                retPts[i] = new Point((int)ip.X, (int)ip.Y);
            }

            return retPts;
        }

        private static float SignedPolygonArea(PointF[] points)
        {
            float area = 0;
            for (int i = 0; i < points.Length - 1; i++)
            {
                area +=
                    (points[i + 1].X - points[i].X) *
                    (points[i + 1].Y + points[i].Y) / 2;
            }

            area +=
                (points[0].X - points[points.Length - 1].X) *
                (points[0].Y + points[points.Length - 1].Y) / 2;

            return area;
        }

        private static double LengthOfPoints(PointF[] box)
        {
            double length = 0;

            PointF pt = box[0];
            double x0 = pt.X;
            double y0 = pt.Y;

            for (int idx = 1; idx < box.Length; idx++)
            {
                PointF pts = box[idx];
                double x1 = pts.X;
                double y1 = pts.Y;
                double dx = x1 - x0;
                double dy = y1 - y0;

                length += Math.Sqrt(dx * dx + dy * dy);

                x0 = x1;
                y0 = y1;
            }

            var dxL = pt.X - x0;
            var dyL = pt.Y - y0;
            length += Math.Sqrt(dxL * dxL + dyL * dyL);

            return length;
        }


        public void Dispose()
        {
            _dbNet.Dispose();
        }
    }
}
