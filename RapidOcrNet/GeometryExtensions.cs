// Apache-2.0 license
// Adapted from PdfPig
// https://github.com/UglyToad/PdfPig/blob/master/src/UglyToad.PdfPig/Geometry/GeometryExtensions.cs

using SixLabors.ImageSharp;

namespace RapidOcrNet
{
    internal static class GeometryExtensions
    {
        /// <summary>
        /// Return true if the points are in counter-clockwise order.
        /// </summary>
        /// <param name="point1">The first point.</param>
        /// <param name="point2">The second point.</param>
        /// <param name="point3">The third point.</param>
        private static bool ccw(in PointF point1, in PointF point2, in PointF point3)
        {
            return (point2.X - point1.X) * (point3.Y - point1.Y) > (point2.Y - point1.Y) * (point3.X - point1.X);
        }

        private sealed class PdfPointXYComparer : IComparer<PointF>
        {
            public static readonly PdfPointXYComparer Instance = new PdfPointXYComparer();

            public int Compare(PointF p1, PointF p2)
            {
                int comp = p1.X.CompareTo(p2.X);
                return comp == 0 ? p1.Y.CompareTo(p2.Y) : comp;
            }
        }

        private static float polarAngle(PointF point1, PointF point2)
        {
            // This is used for grouping, we could use Math.Round()
            return MathF.Atan2(point2.Y - point1.Y, point2.X - point1.X) % MathF.PI;
        }
        /// <summary>
        /// Computes the convex hull of a set of points using Andrew's Monotone Chain algorithm.
        /// This algorithm has a time complexity of O(n log n) and is generally faster and 
        /// more numerically stable than Graham Scan as it avoids trigonometric calculations.
        /// </summary>
        /// <param name="points">The array of points to process.</param>
        /// <returns>A collection of points representing the convex hull in counter-clockwise order.</returns>
        public static IReadOnlyCollection<PointF> MonotoneChain(PointF[] points)
        {
            int n = points.Length;
            if (n <= 2) return points;

            // Sort points lexicographically (by X, then Y)
            Array.Sort(points, (a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));

            PointF[] hull = new PointF[2 * n];
            int k = 0;

            // Build Lower Hull
            for (int i = 0; i < n; ++i)
            {
                while (k >= 2 && CrossProduct(hull[k - 2], hull[k - 1], points[i]) <= 0)
                {
                    k--;
                }
                hull[k++] = points[i];
            }

            // Build Upper Hull
            for (int i = n - 2, t = k + 1; i >= 0; i--)
            {
                while (k >= t && CrossProduct(hull[k - 2], hull[k - 1], points[i]) <= 0)
                {
                    k--;
                }
                hull[k++] = points[i];
            }

            // Result includes the start point twice (as the end), so we take k-1
            return hull.Take(k - 1).ToList();
        }

        private static float CrossProduct(PointF a, PointF b, PointF c)
        {
            return (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
        }

        /// <summary>
        /// Algorithm to find the (oriented) minimum area rectangle (MAR) by first finding the convex hull of the points
        /// and then finding its MAR.
        /// </summary>
        /// <param name="points">The points.</param>
        public static PointF[] MinimumAreaRectangle(PointF[] points)
        {
            if (points is null || points.Length == 0)
            {
                throw new ArgumentException("MinimumAreaRectangle(): points cannot be null and must contain at least one point.", nameof(points));
            }

            return ParametricPerpendicularProjection(MonotoneChain(points.Distinct().ToArray()).ToArray());
        }

        /// <summary>
        /// Algorithm to find a minimal bounding rectangle (MBR) such that the MBR corresponds to a rectangle
        /// with smallest possible area completely enclosing the polygon.
        /// <para>From 'A Fast Algorithm for Generating a Minimal Bounding Rectangle' by Lennert D. Den Boer.</para>
        /// </summary>
        /// <param name="polygon">
        /// Polygon P is assumed to be both simple and convex, and to contain no duplicate (coincident) vertices.
        /// The vertices of P are assumed to be in strict cyclic sequential order, either clockwise or
        /// counter-clockwise relative to the origin P0.
        /// </param>
        private static PointF[] ParametricPerpendicularProjection(ReadOnlySpan<PointF> polygon)
        {
            if (polygon.Length == 0)
            {
                throw new ArgumentException("ParametricPerpendicularProjection(): polygon cannot be null and must contain at least one point.", nameof(polygon));
            }

            if (polygon.Length == 1)
            {
                return [polygon[0], polygon[0]];
            }

            if (polygon.Length == 2)
            {
                return [polygon[0], polygon[1]];
            }

            Span<float> mrb = stackalloc float[8];

            float Amin = float.PositiveInfinity;
            int j = 1;
            int k = 0;

            float QX = float.NaN;
            float QY = float.NaN;
            float R0X = float.NaN;
            float R0Y = float.NaN;
            float R1X = float.NaN;
            float R1Y = float.NaN;

            while (true)
            {
                PointF Pk = polygon[k];
                PointF Pj = polygon[j];

                float vX = Pj.X - Pk.X;
                float vY = Pj.Y - Pk.Y;
                float r = 1.0f / (vX * vX + vY * vY);

                float tmin = 1;
                float tmax = 0;
                float smax = 0;
                int l = -1;
                float uX;
                float uY;

                for (j = 0; j < polygon.Length; j++)
                {
                    Pj = polygon[j];
                    uX = Pj.X - Pk.X;
                    uY = Pj.Y - Pk.Y;
                    float t = (uX * vX + uY * vY) * r;

                    float PtX = t * vX + Pk.X;
                    float PtY = t * vY + Pk.Y;
                    uX = PtX - Pj.X;
                    uY = PtY - Pj.Y;

                    float s = uX * uX + uY * uY;

                    if (t < tmin)
                    {
                        tmin = t;
                        R0X = PtX;
                        R0Y = PtY;
                    }

                    if (t > tmax)
                    {
                        tmax = t;
                        R1X = PtX;
                        R1Y = PtY;
                    }

                    if (s > smax)
                    {
                        smax = s;
                        QX = PtX;
                        QY = PtY;
                        l = j;
                    }
                }

                if (l != -1)
                {
                    PointF Pl = polygon[l];
                    float PlMinusQX = Pl.X - QX;
                    float PlMinusQY = Pl.Y - QY;

                    float R2X = R1X + PlMinusQX;
                    float R2Y = R1Y + PlMinusQY;

                    float R3X = R0X + PlMinusQX;
                    float R3Y = R0Y + PlMinusQY;

                    uX = R1X - R0X;
                    uY = R1Y - R0Y;

                    float A = (uX * uX + uY * uY) * smax;

                    if (A < Amin)
                    {
                        Amin = A;

                        mrb[0] = R0X;
                        mrb[1] = R0Y;
                        mrb[2] = R1X;
                        mrb[3] = R1Y;
                        mrb[4] = R2X;
                        mrb[5] = R2Y;
                        mrb[6] = R3X;
                        mrb[7] = R3Y;
                    }
                }

                k++;
                j = k + 1;

                if (j == polygon.Length) j = 0;
                if (k == polygon.Length) break;
            }

            return
            [
                new PointF(mrb[4], mrb[5]),
                new PointF(mrb[6], mrb[7]),
                new PointF(mrb[2], mrb[3]),
                new PointF(mrb[0], mrb[1])
            ];
        }

        public static void GetSize(PointF[] points, out float width, out float height)
        {
            PointF topLeft = points[0];
            PointF bottomLeft = points[2];
            PointF bottomRight = points[3];

            width = MathF.Sqrt((bottomLeft.X - bottomRight.X) * (bottomLeft.X - bottomRight.X) + (bottomLeft.Y - bottomRight.Y) * (bottomLeft.Y - bottomRight.Y));
            height = MathF.Sqrt((bottomLeft.X - topLeft.X) * (bottomLeft.X - topLeft.X) + (bottomLeft.Y - topLeft.Y) * (bottomLeft.Y - topLeft.Y));
        }

    }
}
